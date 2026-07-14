"""Deep Q-Network agent for ED orchestration."""

from __future__ import annotations

import copy
import random
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.mdp import ACTION_DIM, STATE_DIM, EDOrchestrationEnv, get_action_policies
from ed_digital_twin.distributions import CONFIG
from ed_digital_twin.twin import EmergencyDepartmentTwin


class DQNetwork(nn.Module):
    """Q(s,a; theta) with architecture [256, 256, 128]."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden: Optional[List[int]] = None,
    ):
        super().__init__()
        hidden = hidden or CONFIG["dqn"]["hidden_layers"]
        layers: List[nn.Module] = []
        in_dim = state_dim
        for h in hidden:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: Deque[Tuple] = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN with experience replay and target network."""

    def __init__(self, seed: int = 42):
        self.cfg = CONFIG["dqn"]
        torch.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQNetwork().to(self.device)
        self.target_net = DQNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=self.cfg["learning_rate"]
        )
        self.replay = ReplayBuffer(self.cfg["replay_buffer_size"])
        self.gamma = self.cfg["gamma"]
        self.batch_size = self.cfg["batch_size"]
        self.target_update_freq = self.cfg["target_update_freq"]
        self.training_steps = 0

        self.epsilon = self.cfg["epsilon_start"]
        self.epsilon_end = self.cfg["epsilon_end"]
        self.epsilon_decay = self.cfg["epsilon_decay_episodes"]

        self.reward_history: List[float] = []
        self.loss_history: List[float] = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(ACTION_DIM)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q = self.policy_net(s)
            return int(q.argmax(dim=1).item())

    def optimize(self) -> Optional[float]:
        if len(self.replay) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(
            self.batch_size
        )

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1)[0]
            target = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = nn.functional.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def decay_epsilon(self, episode: int):
        decay = min(1.0, episode / max(self.epsilon_decay, 1))
        self.epsilon = self.epsilon_end + (self.cfg["epsilon_start"] - self.epsilon_end) * (
            1 - decay
        )

    def train(
        self,
        n_episodes: Optional[int] = None,
        seed_offset: int = 0,
    ) -> dict:
        n_episodes = n_episodes or self.cfg["training_episodes"]
        env = EDOrchestrationEnv(seed=seed_offset)

        for ep in range(n_episodes):
            state = env.reset(seed=seed_offset + ep)
            total_reward = 0.0
            done = False

            while not done:
                action = self.select_action(state, training=True)
                next_state, reward, done, _ = env.step(action)
                self.replay.push(state, action, reward, next_state, done)
                loss = self.optimize()
                if loss is not None:
                    self.loss_history.append(loss)
                state = next_state
                total_reward += reward

            self.reward_history.append(total_reward)
            self.decay_epsilon(ep)

            if (ep + 1) % 500 == 0:
                avg_r = np.mean(self.reward_history[-500:])
                print(f"Episode {ep+1}/{n_episodes} | avg_reward={avg_r:.3f} | eps={self.epsilon:.3f}")

        return {
            "reward_history": self.reward_history,
            "loss_history": self.loss_history,
        }

    def get_best_action(self, state: np.ndarray) -> int:
        return self.select_action(state, training=False)

    def save(self, path: Path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: Path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())


class AdaptiveDQNPolicy:
    """Runtime policy: DQN selects dispatch mode each sync cycle."""

    def __init__(self, agent: DQNAgent):
        self.agent = agent
        self.policies = get_action_policies()
        self.current_action = 0
        self.last_state: Optional[np.ndarray] = None

    def update_from_twin(self, twin: EmergencyDepartmentTwin):
        env = EDOrchestrationEnv()
        self.last_state = env._encode_state(twin)
        self.current_action = self.agent.get_best_action(self.last_state)

    def __call__(self, queue, now):
        policy = self.policies[self.current_action]
        return policy(queue, now)

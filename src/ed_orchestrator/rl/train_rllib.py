"""Ray RLlib DQN training with MLflow tracking."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_SIM_PATH = Path(__file__).resolve().parents[3] / "simulation"
if str(_SIM_PATH) not in sys.path:
    sys.path.insert(0, str(_SIM_PATH))

from ed_orchestrator.config import CONFIG  # noqa: E402


def train_with_pytorch_fallback(output_dir: Path, episodes: int) -> Path:
    """Fallback trainer when Ray is unavailable."""
    from agents.dqn import DQNAgent

    output_dir.mkdir(parents=True, exist_ok=True)
    agent = DQNAgent(seed=42)
    history = agent.train(n_episodes=episodes, seed_offset=0)
    model_path = output_dir / "dqn_model.pt"
    agent.save(model_path)
    return model_path, history


def train_with_rllib(output_dir: Path, episodes: int) -> Path:
    import mlflow
    from ray import tune
    from ray.rllib.algorithms.dqn import DQNConfig

    from ed_orchestrator.rl.env import EDOrchestrationGymEnv, register_env

    register_env()
    output_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "ed-orchestrator"))

    cfg = CONFIG["dqn"]
    config = (
        DQNConfig()
        .environment("EDOrchestration-v0")
        .training(
            lr=cfg["learning_rate"],
            gamma=cfg["gamma"],
            train_batch_size=cfg["batch_size"],
            replay_buffer_config={"capacity": cfg["replay_buffer_size"]},
        )
        .resources(num_gpus=int(os.getenv("NUM_GPUS", "0")))
    )

    with mlflow.start_run(run_name="dqn-rllib"):
        mlflow.log_params({"episodes": episodes, **cfg})
        algo = config.build()
        for i in range(max(1, episodes // 10)):
            result = algo.train()
            if i % 5 == 0:
                mlflow.log_metric("episode_reward_mean", result.get("episode_reward_mean", 0), step=i)
        checkpoint = algo.save(str(output_dir / "rllib_checkpoint"))
        mlflow.log_artifact(str(checkpoint))
        return Path(checkpoint)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=CONFIG["dqn"]["training_episodes"])
    parser.add_argument("--output", type=Path, default=Path("models"))
    args = parser.parse_args()

    try:
        import ray  # noqa: F401

        path = train_with_rllib(args.output, args.episodes)
        print(f"RLlib training complete: {path}")
    except ImportError:
        path, history = train_with_pytorch_fallback(args.output, args.episodes)
        print(f"PyTorch fallback training complete: {path}")


if __name__ == "__main__":
    main()

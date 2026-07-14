"""Stochastic distributions for ED simulation (literature-calibrated)."""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
import yaml
from pathlib import Path


def load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config" / "ed_params.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = load_config()


def sample_esi(rng: np.random.Generator) -> int:
    """Sample ESI level from documented distribution (Green 2006)."""
    dist = CONFIG["esi_distribution"]
    levels = sorted(int(k) for k in dist.keys())
    probs = [float(dist.get(k, dist.get(str(k), 0))) for k in levels]
    total = sum(probs)
    probs = [p / total for p in probs]
    return int(rng.choice(levels, p=probs))


def sample_service_time(
    station: str, esi: int, rng: np.random.Generator
) -> float:
    """Sample lognormal service time with ESI multiplier."""
    params = CONFIG["service_times"][station]
    mu = params["mu"]
    sigma = params["sigma"]
    multiplier = CONFIG["esi_service_multipliers"].get(esi, 1.0)
    base = rng.lognormal(mean=math.log(mu), sigma=sigma / mu)
    return max(1.0, base * multiplier)


def nhpp_arrival_rate(hour: float) -> float:
    """Non-homogeneous Poisson arrival rate (peak 10:00-14:00)."""
    cfg = CONFIG["arrival"]
    base = cfg["base_rate_per_hour"]
    peak_start = cfg["peak_start_hour"]
    peak_end = cfg["peak_end_hour"]
    multiplier = cfg["peak_multiplier"]

    if peak_start <= hour < peak_end:
        mid = (peak_start + peak_end) / 2
        width = (peak_end - peak_start) / 2
        factor = 1 + (multiplier - 1) * math.exp(
            -0.5 * ((hour - mid) / width) ** 2
        )
        return base * factor
    return base


def inter_arrival_time(current_hour: float, rng: np.random.Generator) -> float:
    """Generate inter-arrival time from NHPP (minutes)."""
    rate = nhpp_arrival_rate(current_hour)
    return rng.exponential(60.0 / rate)


def needs_imaging(esi: int, rng: np.random.Generator) -> bool:
    prob = CONFIG["imaging_probability"].get(esi, 0.3)
    return rng.random() < prob


def needs_admission(esi: int, rng: np.random.Generator) -> bool:
    prob = CONFIG["admission_probability"].get(esi, 0.1)
    return rng.random() < prob


def generate_reference_service_samples(
    n: int = 10000, seed: int = 42
) -> Dict[str, np.ndarray]:
    """Generate reference samples for KS validation."""
    rng = np.random.Generator(np.random.PCG64(seed))
    samples: Dict[str, np.ndarray] = {}
    for station in ["triage", "consultation", "imaging", "disposition"]:
        samples[station] = np.array(
            [sample_service_time(station, 3, rng) for _ in range(n)]
        )
    return samples

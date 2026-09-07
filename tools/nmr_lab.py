"""Self-contained teaching models for the NMR-AI labs.

The spin-1 Dulya/Pake powder lineshape is theoretical. Instrument gain, baseline,
noise, and parameter distributions remain teaching assumptions, not fitted data.
"""

import numpy as np
import torch
from torch import nn
from lineshape import pake_doublet

FREQUENCY = np.linspace(32.3, 33.1, 512)


def sample_configurations(count, seed=17, coverage="narrow", cc=-1.39):
    rng = np.random.default_rng(seed)
    spread = 1.0 if coverage == "broad" else 0.25
    return [{
        "center_mhz": float(32.7 + rng.uniform(-0.025, 0.025) * spread),
        "split_mhz": float(0.080 + rng.uniform(-0.01, 0.01) * spread),
        "g": float(0.080 + rng.uniform(-0.02, 0.02) * spread),
        "eta": float(0.030 + rng.uniform(-0.02, 0.02) * spread),
        "gain_slope": float(rng.uniform(-0.15, 0.15) * spread),
        "baseline": [float(-0.2 + rng.uniform(-0.03, 0.03) * spread),
                     float(rng.uniform(-0.02, 0.02) * spread),
                     float(rng.uniform(-0.02, 0.02) * spread),
                     float(rng.uniform(-0.008, 0.008) * spread)],
        "cc": float(cc),
    } for _ in range(count)]


def validate_configurations(configurations):
    if not isinstance(configurations, list) or len(configurations) < 3:
        raise ValueError("Supply a list of at least three configurations")
    for config in configurations:
        required = ("center_mhz", "split_mhz", "g", "eta", "gain_slope", "cc", "baseline")
        if not isinstance(config, dict) or any(key not in config for key in required):
            raise ValueError("Pake configurations need center_mhz, split_mhz, g, eta, gain_slope, cc, baseline")
        values = [config[key] for key in required[:-1]]
        if not np.isfinite(values).all() or config["cc"] == 0:
            raise ValueError("Configuration values must be finite, with nonzero cc")
        if config["g"] <= 0 or config["split_mhz"] <= 0 or abs(config["gain_slope"]) >= 1:
            raise ValueError("Widths and splitting must be positive; |gain_slope| must be below 1")
        if not 0 <= config["eta"] <= 1:
            raise ValueError("Quadrupole asymmetry eta must be between 0 and 1")
        if len(config["baseline"]) != 4 or not np.isfinite(config["baseline"]).all():
            raise ValueError("Each baseline needs four finite coefficients, constant through cubic")


def simulate(p, config, rng, noise_level=2.7e-5, noise_correlation=0.0, center_jitter=0.0):
    axis = np.linspace(-1.0, 1.0, len(FREQUENCY))
    center = config["center_mhz"] + rng.uniform(-center_jitter, center_jitter)
    x = (FREQUENCY - center) / config["split_mhz"]
    lineshape, _, _ = pake_doublet(x, p, config["cc"], config["eta"], config["g"])
    lineshape *= 1.0 + config["gain_slope"] * axis
    baseline = np.polynomial.polynomial.polyval(axis, config["baseline"])
    noise = rng.normal(0.0, noise_level, len(axis))
    if noise_correlation:
        for i in range(1, len(noise)):
            noise[i] = noise_correlation * noise[i - 1] + np.sqrt(1 - noise_correlation**2) * noise[i]
    return {"signal": lineshape + baseline + noise, "lineshape": lineshape,
            "baseline": baseline, "noise": noise}


def make_features(signals, cc):
    """Cubic wing fit, followed by known-calibration scaling of the residual."""
    signals = np.asarray(signals, dtype=np.float64)
    cc = np.asarray(cc, dtype=np.float64)
    if signals.ndim != 2 or signals.shape[1] != 512 or cc.shape != (len(signals),):
        raise ValueError("Expected (events, 512) signals and one calibration value per event")
    if not np.isfinite(signals).all() or not np.isfinite(cc).all() or np.any(cc == 0):
        raise ValueError("Signals and calibrations must be finite, with nonzero cc")
    axis = np.linspace(-1.0, 1.0, signals.shape[1])
    design = np.stack([axis**power for power in range(4)], axis=1)
    wings = np.abs(axis) >= 0.65
    coefficients = signals[:, wings] @ np.linalg.pinv(design[wings]).T
    residual = (signals - coefficients @ design.T) * cc[:, None]
    return np.stack([signals, residual], axis=1).astype(np.float32)


def group_split(groups, seed):
    unique = np.unique(groups)
    if len(unique) < 3:
        raise ValueError("At least three sampled configurations are needed for grouped evaluation")
    np.random.default_rng(seed).shuffle(unique)
    n_test = max(1, int(0.1 * len(unique)))
    n_val = max(1, int(0.1 * len(unique)))
    partitions = [unique[:len(unique) - n_test - n_val], unique[-n_test - n_val:-n_test], unique[-n_test:]]
    return [np.flatnonzero(np.isin(groups, partition)) for partition in partitions]


class SmallPolarizationCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(2, 16, 7, padding=3), nn.GELU(), nn.AvgPool1d(2),
                                     nn.Conv1d(16, 32, 7, padding=3), nn.GELU(), nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(32, 1)

    def forward(self, x):
        return self.head(self.encoder(x).squeeze(-1))


class SpectralBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(6, channels), nn.GELU(),
            nn.Conv1d(channels, channels, 5, padding=2 * dilation, dilation=dilation),
            nn.GroupNorm(6, channels), nn.GELU(),
            nn.Conv1d(channels, channels, 5, padding=2 * dilation, dilation=dilation),
        )

    def forward(self, x):
        return x + self.net(x)


class PhysicsMultiscaleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.ModuleList([nn.Conv1d(2, 12, k, padding=k // 2) for k in (5, 15, 31)])
        self.encoder = nn.Sequential(nn.Conv1d(36, 48, 1), nn.GELU(), SpectralBlock(48, 1),
                                     nn.AvgPool1d(2), SpectralBlock(48, 2), nn.AvgPool1d(2),
                                     SpectralBlock(48, 4))
        self.register_buffer("frequency", torch.linspace(-1.0, 1.0, 512))
        self.register_buffer("physics_mean", torch.zeros(1, 25))
        self.register_buffer("physics_std", torch.ones(1, 25))
        self.physics_head = nn.Linear(25, 1)
        self.correction = nn.Sequential(nn.Linear(144, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def physics_features(self, x):
        residual, raw = x[:, 1], x[:, 0]
        center = (residual.abs() * self.frequency).sum(-1) / residual.abs().sum(-1).clamp_min(1e-6)
        axis = self.frequency[None, :] - center[:, None]
        signed = [(residual * axis**power).mean(-1, keepdim=True) for power in range(11)]
        absolute = [(residual.abs() * axis**power).mean(-1, keepdim=True) for power in range(7)]
        stats = [residual.amin(-1, keepdim=True), residual.amax(-1, keepdim=True), residual.std(-1, keepdim=True),
                 raw.mean(-1, keepdim=True), raw.std(-1, keepdim=True), raw.amin(-1, keepdim=True), raw.amax(-1, keepdim=True)]
        return torch.cat(signed + absolute + stats, dim=1)

    def forward(self, x):
        encoded = self.encoder(torch.cat([layer(x) for layer in self.stem], dim=1))
        pooled = torch.cat([encoded.mean(-1), encoded.amax(-1), encoded.std(-1)], dim=1)
        physics = (self.physics_features(x) - self.physics_mean) / self.physics_std
        return self.physics_head(physics) + self.correction(pooled)

    def calibrate(self, inputs, target):
        with torch.no_grad():
            features = self.physics_features(torch.from_numpy(inputs)).numpy().astype(np.float64)
        mean, std = features.mean(0), features.std(0)
        std[std < 1e-8] = 1.0
        design = np.column_stack([(features - mean) / std, np.ones(len(features))])
        penalty = np.eye(design.shape[1]) * 1e-3
        penalty[-1, -1] = 0
        weights = np.linalg.solve(design.T @ design + penalty, design.T @ target)
        with torch.no_grad():
            self.physics_mean.copy_(torch.from_numpy(mean[None]))
            self.physics_std.copy_(torch.from_numpy(std[None]))
            self.physics_head.weight.copy_(torch.from_numpy(weights[:-1].T))
            self.physics_head.bias.copy_(torch.from_numpy(weights[-1]))
        self.physics_head.requires_grad_(False)


def build_model(architecture):
    if architecture == "compact":
        return SmallPolarizationCNN()
    if architecture == "physics_multiscale":
        return PhysicsMultiscaleCNN()
    raise ValueError(f"Unknown architecture: {architecture}")

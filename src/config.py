"""Configuration helpers for reproducible OFDM receiver experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass
class OFDMConfig:
    """Physical-layer, model, and reproducibility settings.

    The receiver uses a block-fading channel inside a frame.  Consequently all
    estimators, including the neural estimator, work from the same frame-level
    aggregate of pilot observations rather than granting the network extra
    time-domain observations.
    """

    num_subcarriers: int = 64
    cp_length: int = 16
    num_ofdm_symbols: int = 14
    modulation: str = "qpsk"
    pilot_spacing: int = 4
    pilot_offset: int = 0
    pilot_pattern: str = "comb"  # "comb" or "staggered"
    pilot_stagger_step: int = 1
    pilot_value_real: float = 0.7071067811865475
    pilot_value_imag: float = 0.7071067811865475
    guard_subcarriers_each_side: int = 4
    null_dc: bool = True
    num_channel_taps: int = 8
    channel_pdp: str = "exponential"  # "uniform" or "exponential"
    exponential_decay: float = 0.45
    dft_truncation_taps: int | None = None
    lmmse_assumed_num_taps: int | None = None
    lmmse_assumed_pdp: str | None = None
    lmmse_assumed_exponential_decay: float | None = None
    snr_db_values: list[int] = field(default_factory=lambda: [0, 5, 10, 15, 20, 25, 30])
    train_snr_db_range: list[float] = field(default_factory=lambda: [0.0, 25.0])
    random_seed: int = 1234
    evaluation_seeds: list[int] = field(default_factory=lambda: [2101, 2201, 2301])
    num_eval_frames: int = 100
    min_error_count: int = 200
    max_eval_frames: int = 100
    model_hidden_channels: int = 32
    model_num_blocks: int = 4
    delay_loss_weight: float = 0.05
    pilot_loss_weight: float = 0.02
    sample_covariance_sizes: list[int] = field(default_factory=lambda: [100, 1000, 10000])
    sample_covariance_seed: int = 9100
    sample_covariance_diagonal_loading: float = 0.02
    sample_covariance_min_eigenvalue: float = 1e-6
    complexity_benchmark_frames: int = 64
    complexity_benchmark_repeats: int = 20
    complexity_warmup_iterations: int = 5
    domain_randomized_training: bool = False
    domain_tap_range: list[int] = field(default_factory=lambda: [8, 12])
    domain_pdp_choices: list[str] = field(default_factory=lambda: ["uniform", "exponential"])
    domain_exponential_decay_range: list[float] = field(default_factory=lambda: [0.25, 0.65])
    multiseed_training_seeds: list[int] = field(default_factory=lambda: [1101, 1201, 1301])
    multiseed_test_seeds: list[int] = field(default_factory=lambda: [8101, 8201, 8301])
    multiseed_snr_db_values: list[int] = field(default_factory=lambda: [10, 20, 30])
    multiseed_num_eval_frames: int = 60
    training_epochs: int = 8
    training_batch_size: int = 64
    training_learning_rate: float = 1e-3
    training_samples_per_epoch: int = 3000
    validation_samples: int = 600
    early_stopping_patience: int = 4
    scheduler_patience: int = 2

    @property
    def pilot_symbol(self) -> complex:
        return complex(self.pilot_value_real, self.pilot_value_imag)

    @property
    def bits_per_symbol(self) -> int:
        mapping = {"qpsk": 2, "16qam": 4}
        try:
            return mapping[self.modulation.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported modulation: {self.modulation}") from exc

    @property
    def num_active_subcarriers(self) -> int:
        return self.num_subcarriers - 2 * self.guard_subcarriers_each_side - int(self.null_dc)

    @property
    def num_data_symbols_per_frame(self) -> int:
        active = self.num_active_subcarriers
        pilot_total = 0
        for symbol_index in range(self.num_ofdm_symbols):
            if self.pilot_pattern == "staggered":
                offset = (self.pilot_offset + symbol_index * self.pilot_stagger_step) % self.pilot_spacing
            else:
                offset = self.pilot_offset
            pilot_total += len(range(offset, active, self.pilot_spacing))
        return self.num_ofdm_symbols * active - pilot_total

    @property
    def num_bits_per_frame(self) -> int:
        return self.num_data_symbols_per_frame * self.bits_per_symbol

    @property
    def effective_dft_taps(self) -> int:
        return self.num_channel_taps if self.dft_truncation_taps is None else self.dft_truncation_taps

    def validate(self) -> None:
        if self.num_subcarriers <= 0:
            raise ValueError("num_subcarriers must be positive.")
        if self.cp_length <= 0:
            raise ValueError("cp_length must be positive.")
        if self.num_channel_taps <= 0:
            raise ValueError("num_channel_taps must be positive.")
        if self.num_channel_taps > self.cp_length:
            raise ValueError("num_channel_taps must be less than or equal to cp_length.")
        if not 0 <= self.effective_dft_taps <= self.cp_length:
            raise ValueError("dft_truncation_taps must lie in [1, cp_length].")
        if self.pilot_spacing <= 0:
            raise ValueError("pilot_spacing must be positive.")
        if not 0 <= self.pilot_offset < self.pilot_spacing:
            raise ValueError("pilot_offset must satisfy 0 <= offset < pilot_spacing.")
        if self.pilot_pattern not in {"comb", "staggered"}:
            raise ValueError("pilot_pattern must be 'comb' or 'staggered'.")
        if self.modulation.lower() not in {"qpsk", "16qam"}:
            raise ValueError("modulation must be 'qpsk' or '16qam'.")
        if self.channel_pdp not in {"uniform", "exponential"}:
            raise ValueError("channel_pdp must be 'uniform' or 'exponential'.")
        if self.guard_subcarriers_each_side < 0:
            raise ValueError("guard_subcarriers_each_side must be non-negative.")
        if self.num_active_subcarriers < 2:
            raise ValueError("Guard/DC allocation leaves fewer than two active subcarriers.")
        if len(self.train_snr_db_range) != 2 or self.train_snr_db_range[0] > self.train_snr_db_range[1]:
            raise ValueError("train_snr_db_range must be [minimum, maximum].")
        if not self.evaluation_seeds:
            raise ValueError("evaluation_seeds must not be empty.")
        if not self.sample_covariance_sizes or any(size <= 1 for size in self.sample_covariance_sizes):
            raise ValueError("sample_covariance_sizes must contain integers greater than one.")
        if not 0.0 <= self.sample_covariance_diagonal_loading <= 1.0:
            raise ValueError("sample_covariance_diagonal_loading must be in [0, 1].")
        if self.sample_covariance_min_eigenvalue <= 0.0:
            raise ValueError("sample_covariance_min_eigenvalue must be positive.")
        if self.complexity_benchmark_frames <= 0 or self.complexity_benchmark_repeats <= 0:
            raise ValueError("Complexity benchmark frame/repeat counts must be positive.")
        if len(self.domain_tap_range) != 2 or not 1 <= self.domain_tap_range[0] <= self.domain_tap_range[1] <= self.cp_length:
            raise ValueError("domain_tap_range must be [min_taps, max_taps] within the CP length.")
        if any(pdp not in {"uniform", "exponential"} for pdp in self.domain_pdp_choices):
            raise ValueError("domain_pdp_choices must contain only 'uniform' and/or 'exponential'.")
        if len(self.domain_exponential_decay_range) != 2 or self.domain_exponential_decay_range[0] <= 0.0 or self.domain_exponential_decay_range[0] > self.domain_exponential_decay_range[1]:
            raise ValueError("domain_exponential_decay_range must be an increasing positive range.")
        if len(self.multiseed_training_seeds) < 3 or len(self.multiseed_test_seeds) < 3:
            raise ValueError("multiseed experiments require at least three training and test seeds.")
        if not self.multiseed_snr_db_values or self.multiseed_num_eval_frames <= 0:
            raise ValueError("multiseed SNR values and frame count must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        """Stable fingerprint used to prevent incompatible checkpoint reuse."""

        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_config(path: str | Path = "configs/default_config.json") -> OFDMConfig:
    """Load a JSON configuration and safely ignore execution-only metadata keys."""

    cfg_path = Path(path)
    base = OFDMConfig().to_dict()
    if cfg_path.exists():
        supplied = json.loads(cfg_path.read_text(encoding="utf-8"))
        supported = {item.name for item in fields(OFDMConfig)}
        base.update({key: value for key, value in supplied.items() if key in supported})
    cfg = OFDMConfig(**base)
    cfg.validate()
    return cfg

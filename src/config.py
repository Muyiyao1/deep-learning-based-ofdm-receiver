"""Configuration helpers for the OFDM receiver project."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class OFDMConfig:
    """Container for all physical-layer simulation parameters."""

    num_subcarriers: int = 64
    cp_length: int = 16
    num_ofdm_symbols: int = 14
    modulation: str = "qpsk"
    pilot_spacing: int = 4
    pilot_offset: int = 0
    pilot_value_real: float = 0.7071067811865475
    pilot_value_imag: float = 0.7071067811865475
    num_channel_taps: int = 8
    snr_db_values: list[int] = field(default_factory=lambda: [0, 5, 10, 15, 20, 25, 30])
    random_seed: int = 1234
    num_eval_frames: int = 500

    @property
    def pilot_symbol(self) -> complex:
        return complex(self.pilot_value_real, self.pilot_value_imag)

    @property
    def bits_per_symbol(self) -> int:
        mod = self.modulation.lower()
        if mod == "qpsk":
            return 2
        if mod == "16qam":
            return 4
        raise ValueError(f"Unsupported modulation: {self.modulation}")

    @property
    def num_pilots_per_symbol(self) -> int:
        return len(range(self.pilot_offset, self.num_subcarriers, self.pilot_spacing))

    @property
    def num_data_subcarriers(self) -> int:
        return self.num_subcarriers - self.num_pilots_per_symbol

    @property
    def num_data_symbols_per_frame(self) -> int:
        return self.num_ofdm_symbols * self.num_data_subcarriers

    @property
    def num_bits_per_frame(self) -> int:
        return self.num_data_symbols_per_frame * self.bits_per_symbol

    def validate(self) -> None:
        if self.num_subcarriers <= 0:
            raise ValueError("num_subcarriers must be positive.")
        if self.cp_length <= 0:
            raise ValueError("cp_length must be positive.")
        if self.num_channel_taps <= 0:
            raise ValueError("num_channel_taps must be positive.")
        if self.num_channel_taps > self.cp_length:
            raise ValueError("num_channel_taps must be less than or equal to cp_length.")
        if self.pilot_spacing <= 0:
            raise ValueError("pilot_spacing must be positive.")
        if not 0 <= self.pilot_offset < self.pilot_spacing:
            raise ValueError("pilot_offset must satisfy 0 <= offset < pilot_spacing.")
        if self.modulation.lower() not in {"qpsk", "16qam"}:
            raise ValueError("modulation must be 'qpsk' or '16qam'.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path = "configs/default_config.json") -> OFDMConfig:
    """Load a JSON configuration file and merge it with dataclass defaults."""

    cfg_path = Path(path)
    base = OFDMConfig().to_dict()
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as handle:
            base.update(json.load(handle))
    cfg = OFDMConfig(**base)
    cfg.validate()
    return cfg

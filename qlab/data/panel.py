# Excerpt from Hao Zheng's A-share quant research codebase (2026). Shared for interview review only;
# the data, signal, alpha-model and portfolio-optimizer layers are not included.
"""Panel: the standard in-memory data container.

Every field is a wide DataFrame indexed by trade_date (DatetimeIndex),
columns = symbol. Prices are adjusted (raw * adj_factor); `amount` is real
turnover value (CNY) from the source, falling back to the raw close * volume
proxy only for legacy sources without an amount column.

The lock_* fields are exact exchange limit-lock masks (bool), computable only
by builders that see RAW prices + adj_factor (engine.lock_masks; the t9 build
sets them). Sources without raw prices leave them None and trade_masks falls
back to its adjusted-return inference.
"""

from dataclasses import dataclass, fields
from pathlib import Path

import pandas as pd


@dataclass
class Panel:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    ret: pd.DataFrame  # close-to-close simple return
    amount: pd.DataFrame
    st: pd.DataFrame  # 1.0 = ST flag on
    suspended: pd.DataFrame  # 1.0 = suspended
    days_listed: pd.DataFrame  # calendar days since list_date
    vwap: pd.DataFrame | None = None  # adjusted day VWAP (amount / volume * adj); fill price
    lock_up: pd.DataFrame | None = None  # closed within one tick of the up limit
    lock_dn: pd.DataFrame | None = None
    lock_up_open: pd.DataFrame | None = None  # one-字板: opened at limit, never unsealed
    lock_dn_open: pd.DataFrame | None = None

    @property
    def field_names(self) -> list[str]:
        """Populated fields only — None lock_* fields stay invisible, so generic
        per-field iteration (save, truncation slicing) works on any Panel."""
        return [f.name for f in fields(self) if getattr(self, f.name) is not None]

    def save(self, cache_dir: Path) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        for name in self.field_names:
            getattr(self, name).to_parquet(cache_dir / f"{name}.parquet")

    @classmethod
    def load(cls, cache_dir: Path) -> "Panel":
        return cls(**{
            f.name: pd.read_parquet(p)
            for f in fields(cls) if (p := cache_dir / f"{f.name}.parquet").exists()
        })

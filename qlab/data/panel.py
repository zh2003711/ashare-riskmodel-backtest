# Excerpt from Hao Zheng's A-share quant research codebase, shared for interview review.
"""Panel: wide date x symbol DataFrames of adjusted prices, turnover and trading-status flags."""

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
    lock_up_open: pd.DataFrame | None = None  # opened at the limit and never traded off it
    lock_dn_open: pd.DataFrame | None = None

    @property
    def field_names(self) -> list[str]:
        """Names of the populated fields."""
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

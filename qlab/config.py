# Excerpt from Hao Zheng's A-share quant research codebase, shared for interview review.
"""Trading cost configuration for the backtest engine."""

from pydantic import BaseModel, ConfigDict, model_validator


class CostCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    buy_bps: float = 15.0  # commission + slippage
    sell_bps: float = 25.0  # + stamp duty
    # with a measured cost surface, buy/sell_bps hold fees only; spread and impact come from the surface
    transfer_bps: float = 0.0       # transfer fee per side (0.2 = 0.02‰); halved from 2022-04-29
    surface: str | None = None      # "v2" | "v1.3" | "v1.2"
    aum: float | None = None        # capital in CNY for the executable ledger
    participation: float = 0.10     # max daily participation per name

    @model_validator(mode="after")
    def _surface_needs_fee_only_bps(self):
        if self.surface and (self.buy_bps > 5.0 or self.sell_bps > 15.0):
            raise ValueError(
                f"costs.surface={self.surface} 已含滑点与冲击, buy/sell_bps 必须是纯费用表"
                f"(如 2.5/12.5), 现为 {self.buy_bps}/{self.sell_bps} —— 会重复计费")
        return self

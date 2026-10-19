# Excerpt from Hao Zheng's A-share quant research codebase (2026). Shared for interview review only;
# the data, signal, alpha-model and portfolio-optimizer layers are not included.
"""Cost configuration consumed by the backtest engine (the only config class the excerpt needs)."""

from pydantic import BaseModel, ConfigDict, model_validator


class CostCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    buy_bps: float = 15.0  # commission + slippage
    sell_bps: float = 25.0  # + stamp duty
    # 实测成本面(cost-calib): 设了 surface 就走 aum 账本, 执行项 = 曲面的 半价差 + k·σ·√(q/ADV),
    # 此时 buy/sell_bps 只能是声明费用表(佣金 2.5 / 佣金 2.5 + 印花税 10, 2023-08-28 后引擎自动减 5)
    # —— 会计恒等式 全口径 = 实测执行项 + 声明费用表, 两边不许重复计滑点。
    transfer_bps: float = 0.0       # 过户费 per side (0.2 = 0.02‰); engine halves it from 2022-04-29
    surface: str | None = None      # "v2" | "v1.3" | "v1.2"
    aum: float | None = None        # 设了就走 aum 可执行账本(元); surface 生效时默认 1e8
    participation: float = 0.10     # 单名日参与率上限(aum 账本用)

    @model_validator(mode="after")
    def _surface_needs_fee_only_bps(self):
        if self.surface and (self.buy_bps > 5.0 or self.sell_bps > 15.0):
            raise ValueError(
                f"costs.surface={self.surface} 已含滑点与冲击, buy/sell_bps 必须是纯费用表"
                f"(如 2.5/12.5), 现为 {self.buy_bps}/{self.sell_bps} —— 会重复计费")
        return self

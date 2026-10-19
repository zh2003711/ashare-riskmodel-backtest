# Excerpt from Hao Zheng's A-share quant research codebase, shared for interview review.
"""Execution prices, cost split, transfer fee and dividend tax."""

import numpy as np
import pandas as pd
import pytest

from qlab.backtest.engine import DIV_TAX, exec_leg, run, trade_masks
from qlab.config import CostCfg
from qlab.data.panel import Panel

DATES = pd.date_range("2024-06-03", periods=6, freq="B")
SYMS = ["A", "B"]
ZERO = CostCfg(buy_bps=0, sell_bps=0)


def _panel(ret_a, open_a=None, vwap_a=None):
    ret = pd.DataFrame({"A": ret_a, "B": [0.0] * 6}, index=DATES)
    close = (1 + ret).cumprod()
    zeros = pd.DataFrame(0.0, index=DATES, columns=SYMS)
    op, vw = close.copy(), close.copy()
    if open_a is not None:
        op["A"] = open_a
    if vwap_a is not None:
        vw["A"] = vwap_a
    return Panel(open=op, high=close * 1.01, low=close * 0.99, close=close, ret=ret,
                 amount=close * 1e6, st=zeros, suspended=zeros, days_listed=zeros + 999, vwap=vw)


def _target(day, weights):
    t = pd.DataFrame(np.nan, index=DATES, columns=SYMS)
    for s, w in weights.items():
        t.loc[DATES[day], s] = w
    return t


def test_close_mode_is_the_classic_book():
    panel = _panel([0, 0.10, 0.05, 0, 0, 0])
    a = run(_target(0, {"A": 1.0}), panel, CostCfg(buy_bps=10, sell_bps=20))
    b = run(_target(0, {"A": 1.0}), panel, CostCfg(buy_bps=10, sell_bps=20), execution="close")
    np.testing.assert_allclose(a.net_ret, b.net_ret, atol=0)
    assert (exec_leg(panel, "close") == 0).all()


def test_vwap_fill_earns_only_the_fill_to_close_leg():
    # d1: A closes at 1.10; fill at VWAP 1.05 -> the buy earns 1.10/1.05 - 1 on d1 itself
    panel = _panel([0, 0.10, 0.05, 0, 0, 0])
    panel.vwap.loc[DATES[1], "A"] = 1.05
    res = run(_target(0, {"A": 1.0}), panel, ZERO, execution="vwap")
    assert res.gross_ret.iloc[1] == pytest.approx(1.10 / 1.05 - 1)
    assert res.gross_ret.iloc[2] == pytest.approx(0.05)


def test_open_fill_sell_leg_uses_open_to_prev_close():
    # hold A from d1 (open fill = close, no leg-B gain); d3 open at 1.20 vs prev close 1.155:
    # selling at the open books 1.20/1.155 - 1 for the day, not the close-to-close return
    panel = _panel([0, 0.10, 0.05, -0.02, 0, 0])
    tgt = _target(0, {"A": 1.0})
    tgt.loc[DATES[2]] = [0.0, 0.0]
    panel.open.loc[DATES[3], "A"] = 1.20
    res = run(tgt, panel, ZERO, execution="open")
    prev_close = panel.close.loc[DATES[2], "A"]
    assert res.gross_ret.iloc[3] == pytest.approx(1.20 / prev_close - 1)
    assert res.n_holdings.iloc[3] == 0


def test_open_mode_uses_open_locked_masks():
    panel = _panel([0.0, 0.10, 0.10, 0.0, 0.0, 0.0])
    panel.open.loc[DATES[1:3], "A"] = panel.close.loc[DATES[1:3], "A"]
    panel.low.loc[DATES[1], "A"] = panel.open.loc[DATES[1], "A"]
    panel.high.loc[DATES[1], "A"] = panel.open.loc[DATES[1], "A"]
    b_open, _ = trade_masks(panel, "open")
    b_vwap, _ = trade_masks(panel, "vwap")
    assert not b_open.loc[DATES[1], "A"] and b_open.loc[DATES[2], "A"]
    assert (b_vwap == trade_masks(panel, "close")[0]).all().all()


def test_four_way_split_sums_to_cost_and_transfer_halves():
    panel = _panel([0.0] * 6)
    tgt = _target(0, {"A": 1.0})
    tgt.loc[DATES[2]] = [0.0, 0.0]
    res = run(tgt, panel, CostCfg(buy_bps=2.5, sell_bps=12.5, transfer_bps=0.2))
    total = res.cost_fee + res.cost_tax + res.cost_spread + res.cost_impact
    np.testing.assert_allclose(total, res.cost, atol=1e-15)
    b = 1.0 / (1.0 + (2.5 + 0.1) / 1e4)     # fee-aware full build; transfer 0.1bp after 2022-04
    assert res.cost_fee.iloc[1] == pytest.approx(b * 2.5 / 1e4)
    assert res.cost_tax.iloc[1] == pytest.approx(b * 0.1 / 1e4)
    # sell d3 (2024 -> stamp 5bp + transfer 0.1bp): tax = 5.1bp, fee = 2.5bp on the sold notional
    assert res.cost_tax.iloc[3] == pytest.approx(res.turnover.iloc[3] * 5.1 / 1e4)
    assert res.cost_fee.iloc[3] == pytest.approx(res.turnover.iloc[3] * 2.5 / 1e4)


def test_transfer_fee_before_halving_date_is_double():
    dates = pd.date_range("2022-04-25", periods=4, freq="B")     # 04-25 .. 04-28: pre-halving
    ret = pd.DataFrame(0.0, index=dates, columns=SYMS)
    close = (1 + ret).cumprod()
    zeros = pd.DataFrame(0.0, index=dates, columns=SYMS)
    panel = Panel(open=close, high=close, low=close, close=close, ret=ret, amount=close * 1e6,
                  st=zeros, suspended=zeros, days_listed=zeros + 999)
    tgt = pd.DataFrame(np.nan, index=dates, columns=SYMS)
    tgt.loc[dates[0], "A"] = 1.0
    res = run(tgt, panel, CostCfg(buy_bps=0, sell_bps=0, transfer_bps=0.2))
    assert res.cost_tax.iloc[1] == pytest.approx(res.turnover.iloc[1] * 0.2 / 1e4)


def test_dividend_tax_by_holding_age():
    dates = pd.date_range("2024-01-01", periods=300, freq="B")
    ret = pd.DataFrame(0.0, index=dates, columns=SYMS)
    close = (1 + ret).cumprod()
    zeros = pd.DataFrame(0.0, index=dates, columns=SYMS)
    panel = Panel(open=close, high=close, low=close, close=close, ret=ret, amount=close * 1e6,
                  st=zeros, suspended=zeros, days_listed=zeros + 999)
    tgt = pd.DataFrame(np.nan, index=dates, columns=SYMS)
    tgt.loc[dates[0], "A"] = 1.0                       # held from d1 on
    dy = pd.DataFrame(np.nan, index=dates, columns=SYMS)
    dy.loc[dates[10], "A"] = 0.02                      # age 10 bars  -> 20%
    dy.loc[dates[100], "A"] = 0.02                     # age 100 bars -> 10%
    dy.loc[dates[290], "A"] = 0.02                     # age 290 bars -> 0
    res = run(tgt, panel, ZERO, div_yield=dy)
    assert res.cost_tax.iloc[10] == pytest.approx(0.02 * DIV_TAX[0][1])
    assert res.cost_tax.iloc[100] == pytest.approx(0.02 * DIV_TAX[1][1])
    assert res.cost_tax.iloc[290] == 0.0
    assert res.cost.sum() == pytest.approx(0.02 * 0.30)

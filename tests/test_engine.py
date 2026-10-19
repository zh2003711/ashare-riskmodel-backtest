# Excerpt from Hao Zheng's A-share quant research codebase (2026). Shared for interview review only;
# the data, signal, alpha-model and portfolio-optimizer layers are not included.
"""Engine correctness against hand-computed cases."""

import numpy as np
import pandas as pd
import pytest

from qlab.backtest.engine import lock_masks, run, trade_masks
from qlab.config import CostCfg
from qlab.data.panel import Panel

DATES = pd.date_range("2020-01-01", periods=6, freq="B")
SYMS = ["A", "B"]
COSTS = CostCfg(buy_bps=10, sell_bps=20)


def _panel(ret_a, suspended_a=None):
    ret = pd.DataFrame({"A": ret_a, "B": [0.0] * 6}, index=DATES)
    close = (1 + ret).cumprod()
    zeros = pd.DataFrame(0.0, index=DATES, columns=SYMS)
    susp = zeros.copy()
    if suspended_a:
        susp.loc[DATES[suspended_a], "A"] = 1.0
    return Panel(
        open=close, high=close * 1.01, low=close * 0.99, close=close,
        ret=ret, amount=close, st=zeros, suspended=susp,
        days_listed=zeros + 999,
    )


def _target(day: int, weights: dict) -> pd.DataFrame:
    t = pd.DataFrame(np.nan, index=DATES, columns=SYMS)
    for s, w in weights.items():
        t.loc[DATES[day], s] = w
    return t


def test_no_phantom_leverage_when_sell_blocked():
    """Hold 100% A, then rotate to 100% B — but A is suspended on B's execution day (can't
    sell). The buy of B must be scaled to available cash (0), never levered on phantom cash."""
    both = _target(0, {"A": 1.0})                         # signal d0: buy A (fills d1)
    both.loc[DATES[3]] = _target(3, {"B": 1.0}).loc[DATES[3]]   # signal d3: rotate to B (fills d4)
    panel = _panel([0, 0, 0, 0, 0, 0], suspended_a=4)     # A suspended on d4 -> sell of A blocked
    res = run(both, panel, CostCfg(buy_bps=0, sell_bps=0))
    assert res.n_holdings.iloc[4] <= 1.01                 # A stuck ~1.0, B-buy scaled to ~0
    assert res.turnover.iloc[4] <= 1e-6                   # nothing tradable -> no lever


def test_costs_and_timing():
    # signal d0: 100% A; executes at close d1; A returns 10% on d2
    panel = _panel([0, 0, 0.10, 0, 0, 0])
    res = run(_target(0, {"A": 1.0}), panel, COSTS)
    # fee-aware budget: a full-invest build buys 1/(1+buy_bps) so cash covers the fee too
    b = 1.0 / (1.0 + 10 / 1e4)
    assert res.net_ret.iloc[1] == pytest.approx(-(b * 10 / 1e4))  # buy cost only
    assert res.gross_ret.iloc[2] == pytest.approx(0.10)  # return accrues after entry
    assert res.gross_ret.iloc[1] == 0.0
    assert res.turnover.iloc[1] == pytest.approx(b)


def test_sell_cost_on_exit():
    panel = _panel([0.0] * 6)
    target = _target(0, {"A": 1.0})
    target.loc[DATES[2], "A"] = 0.0
    target.loc[DATES[2], "B"] = 0.0
    res = run(target, panel, COSTS)
    assert res.cost.iloc[3] == pytest.approx(1.0 * 20 / 1e4)  # full exit at sell_bps
    assert res.n_holdings.iloc[3] == 0


def test_suspension_blocks_buy():
    # A suspended on execution day d1 -> buy blocked, missed 10% on d2
    panel = _panel([0, 0, 0.10, 0, 0, 0], suspended_a=1)
    res = run(_target(0, {"A": 1.0}), panel, COSTS)
    assert res.gross_ret.iloc[2] == 0.0
    assert res.cost.iloc[1] == 0.0
    assert res.n_holdings.iloc[2] == 0


def test_limit_up_lock_blocks_buy():
    # A up 10% on execution day and closed at the day's high -> locked, no buy
    panel = _panel([0, 0.10, 0.10, 0, 0, 0])
    panel.high.loc[DATES[1], "A"] = panel.close.loc[DATES[1], "A"]
    res = run(_target(0, {"A": 1.0}), panel, COSTS)
    assert res.cost.iloc[1] == 0.0
    assert res.gross_ret.iloc[2] == 0.0  # missed the move


def test_delisted_holding_written_off():
    dates = pd.date_range("2020-01-01", periods=30, freq="B")
    ret = pd.DataFrame({"A": [0.0] * 30, "B": [0.0] * 30}, index=dates)
    close = (1 + ret).cumprod()
    close.loc[dates[5]:, "A"] = np.nan  # A stops trading forever
    zeros = pd.DataFrame(0.0, index=dates, columns=SYMS)
    panel = Panel(open=close, high=close, low=close, close=close, ret=ret,
                  amount=close, st=zeros, suspended=zeros, days_listed=zeros + 999)
    target = pd.DataFrame(np.nan, index=dates, columns=SYMS)
    target.loc[dates[0], "A"] = 1.0
    res = run(target, panel, COSTS)
    assert res.n_holdings.iloc[10] == 1  # frozen but still held
    assert res.n_holdings.iloc[-1] == 0  # written off after DELIST_DAYS


def test_long_suspension_stays_frozen_and_books_resume_gap():
    # A suspends for 25 trading days (> DELIST_DAYS) then resumes 40% lower: the holding
    # must stay frozen (not written off) and the resume repricing must hit P&L via ret.
    dates = pd.date_range("2020-01-01", periods=35, freq="B")
    close = pd.DataFrame({"A": 1.0, "B": 1.0}, index=dates)
    close.loc[dates[5]:dates[29], "A"] = np.nan
    close.loc[dates[30]:, "A"] = 0.6
    ret = close.ffill().pct_change(fill_method=None).fillna(0.0)
    zeros = pd.DataFrame(0.0, index=dates, columns=SYMS)
    panel = Panel(open=close, high=close, low=close, close=close, ret=ret,
                  amount=close, st=zeros, suspended=zeros, days_listed=zeros + 999)
    target = pd.DataFrame(np.nan, index=dates, columns=SYMS)
    target.loc[dates[0], "A"] = 1.0
    res = run(target, panel, CostCfg(buy_bps=0, sell_bps=0))
    assert res.n_holdings.iloc[28] == 1  # past DELIST_DAYS, still frozen
    assert res.gross_ret.iloc[30] == pytest.approx(-0.4)
    assert (1 + res.gross_ret).prod() == pytest.approx(0.6)


def test_off_panel_target_raises():
    panel = _panel([0.0] * 6)
    with pytest.raises(ValueError):
        run(_target(0, {"A": 1.0}).rename(columns={"B": "B.SZ"}), panel, COSTS)
    off_calendar = pd.DataFrame(
        {"A": [1.0], "B": [np.nan]}, index=[pd.Timestamp("2020-01-04")]  # a Saturday
    )
    with pytest.raises(ValueError):
        run(off_calendar, panel, COSTS)


def test_open_execution_mask_blocks_one_word_board():
    # d1: A opens pinned at +10% and never trades off it -> open-locked. d2: A opens at
    # +10% but low prints below the limit (unseals) -> buyable at open even though the
    # close-based mask would call it locked.
    panel = _panel([0.0, 0.10, 0.10, 0.0, 0.0, 0.0])
    panel.open.loc[DATES[1:3], "A"] = panel.close.loc[DATES[1:3], "A"]
    panel.low.loc[DATES[1], "A"] = panel.open.loc[DATES[1], "A"]
    panel.high.loc[DATES[1], "A"] = panel.open.loc[DATES[1], "A"]
    buyable_open, _ = trade_masks(panel, "open")
    assert not buyable_open.loc[DATES[1], "A"]
    assert buyable_open.loc[DATES[2], "A"]  # low 0.99*close < open: traded off the limit


def test_weight_drift():
    # 50/50; A gains 10% on d2 -> gross = 5%; A weight drifts to 55/105
    panel = _panel([0, 0, 0.10, 0.10, 0, 0])
    res = run(_target(0, {"A": 0.5, "B": 0.5}), panel, COSTS)
    assert res.gross_ret.iloc[2] == pytest.approx(0.05)
    drifted_a = 0.55 / 1.05
    assert res.gross_ret.iloc[3] == pytest.approx(drifted_a * 0.10)


# ---------- exact limit-lock masks (raw prices) ----------

def test_lock_masks_exact_prices():
    dates = pd.date_range("2024-01-01", periods=2, freq="B")
    cols = ["600100", "600200", "000005", "920001", "600300"]
    close = pd.DataFrame([[100.0, 100.0, 10.0, 10.0, 20.0],
                          [110.0, 109.6, 10.5, 12.0, 11.0]], index=dates, columns=cols)
    ones = pd.DataFrame(1.0, index=dates, columns=cols)
    adj = ones.copy()
    adj.loc[dates[1], "600300"] = 2.0  # 1:1 split on day 1: 除权参考价 = 20*1/2 = 10
    st = ones * 0.0
    st["000005"] = 1.0
    m = lock_masks(close, close, close, close, adj, st, ones * 999)
    assert m["lock_up"].loc[dates[0]].sum() == 0  # no reference price on the first bar
    assert m["lock_up"].loc[dates[1]].to_dict() == {
        "600100": True,   # closed at the exact 110.00 limit
        "600200": False,  # +9.6% at day high, but below 110.00 minus one tick
        "000005": True,   # main-board ST 5% -> 10.50
        "920001": False,  # 北交所 30%: +20% is an ordinary day
        "600300": True,   # ex-div reference 10 -> limit 11.00 despite raw close halving
    }


def test_lock_masks_down_limit():
    dates = pd.date_range("2024-01-01", periods=2, freq="B")
    close = pd.DataFrame({"600100": [10.0, 9.0], "600200": [10.0, 9.02]}, index=dates)
    ones = pd.DataFrame(1.0, index=dates, columns=close.columns)
    m = lock_masks(close, close, close, close, ones, ones * 0.0, ones * 999)
    assert m["lock_dn"].loc[dates[1]].to_dict() == {"600100": True, "600200": False}


def test_lock_masks_half_up_rounding():
    """The exchange rounds half-up at both steps; np.round's half-even misses sealed
    boards. Real cases from review: 002190 2016-07-12 (ex-div ref 35.444901 -> 参考价
    35.44 -> limit 38.98, sealed) and the 1.15 -> down limit half-up(1.035) = 1.04."""
    dates = pd.date_range("2024-01-01", periods=2, freq="B")
    cols = ["002190", "600100"]
    close = pd.DataFrame([[35.54, 1.15], [38.98, 1.04]], index=dates, columns=cols)
    adj = pd.DataFrame(1.0, index=dates, columns=cols)
    adj.loc[dates[1], "002190"] = 35.54 / 35.444901  # ex-div: unrounded ref 35.444901
    zeros = close * 0.0
    m = lock_masks(close, close, close, close, adj, zeros, zeros + 999)
    assert m["lock_up"].loc[dates[1], "002190"]
    assert m["lock_dn"].loc[dates[1], "600100"]


def test_lock_masks_new_listing_window():
    # STAR listing: bars 1-5 limit-free (+30% daily), bar 6 locks at the exact 20% limit
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    px = [10.0, 13.0, 16.9, 21.97, 28.56, round(28.56 * 1.2, 2)]
    close = pd.DataFrame({"688001": px, "600100": [10.0] * 6}, index=dates)
    ones = pd.DataFrame(1.0, index=dates, columns=close.columns)
    days = ones.copy()
    days["688001"] = np.arange(1, 7)[:, None]
    m = lock_masks(close, close, close, close, ones, ones * 0.0, days * 1.0)
    assert not m["lock_up"]["688001"].iloc[:5].any()
    assert m["lock_up"].loc[dates[5], "688001"]


def test_lock_masks_open_one_word_board():
    # A opens pinned at the 11.00 limit and never trades off it; B opens there but unseals
    dates = pd.date_range("2024-01-01", periods=2, freq="B")
    cols = ["600100", "600200"]
    close = pd.DataFrame([[10.0, 10.0], [11.0, 11.0]], index=dates, columns=cols)
    op = close.copy()
    low = close.copy()
    low.loc[dates[1], "600200"] = 10.5
    ones = pd.DataFrame(1.0, index=dates, columns=cols)
    m = lock_masks(close, op, close, low, ones, ones * 0.0, ones * 999)
    assert m["lock_up_open"].loc[dates[1]].to_dict() == {"600100": True, "600200": False}
    assert m["lock_up"].loc[dates[1]].to_dict() == {"600100": True, "600200": True}


def test_trade_masks_prefers_exact_masks():
    # inference would call day 1 locked (+10% closing at the high); the exact masks say no
    panel = _panel([0, 0.10, 0, 0, 0, 0])
    panel.high.loc[DATES[1], "A"] = panel.close.loc[DATES[1], "A"]
    unlocked = panel.close * 0 > 1  # all-False bool frame
    panel.lock_up = panel.lock_dn = panel.lock_up_open = panel.lock_dn_open = unlocked
    buyable, _ = trade_masks(panel)
    assert buyable.loc[DATES[1], "A"]


def test_zero_amount_blocks_trade():
    panel = _panel([0.0] * 6)
    panel.amount.loc[DATES[1], "A"] = 0.0
    buyable, sellable = trade_masks(panel)
    assert not buyable.loc[DATES[1], "A"]
    assert not sellable.loc[DATES[1], "A"]


def test_pinned_regression():
    """Full-path drift tripwire (rqalpha's result-pinning pattern): a scenario exercising
    fee-aware entry, fee repricing, drift, a suspension-blocked rotation and a
    budget-scaled buy, pinned to hand-verified values. A diff here means the engine's
    semantics changed — intentional changes must re-pin AND re-verify by hand."""
    dates = pd.date_range("2020-01-01", periods=8, freq="B")
    ret = pd.DataFrame({"A": [0, 0, 0.10, 0, 0, 0, 0, 0.05],
                        "B": [0.0] * 5 + [-0.10, 0, 0]}, index=dates)
    close = (1 + ret).cumprod()
    zeros = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
    susp = zeros.copy()
    susp.loc[dates[4], "B"] = 1.0  # rotation day: sell of B blocked, no phantom leverage
    panel = Panel(open=close, high=close * 1.01, low=close * 0.99, close=close, ret=ret,
                  amount=close, st=zeros, suspended=susp, days_listed=zeros + 999)
    tgt = pd.DataFrame(np.nan, index=dates, columns=["A", "B"])
    tgt.loc[dates[0]] = [0.5, 0.5]
    tgt.loc[dates[3]] = [1.0, 0.0]
    tgt.loc[dates[5]] = [1.0, 0.0]
    res = run(tgt, panel, CostCfg(buy_bps=10, sell_bps=20))
    pinned = {
        "gross_ret": [0.0, 0.0, 0.05, 0.0, 0.0, -0.047619047619, 0.0, 0.05],
        "net_ret": [0.0, -0.000999000999, 0.05, 0.0, 0.0, -0.047619047619,
                    -0.001348651349, 0.05],
        "cost": [0.0, 0.000999000999, 0.0, 0.0, 0.0, 0.0, 0.001348651349, 0.0],
        "turnover": [0.0, 0.999000999001, 0.0, 0.0, 0.0, 0.0, 0.45, 0.0],
        "n_holdings": [0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0],
    }
    for name, expected in pinned.items():
        np.testing.assert_allclose(getattr(res, name).to_numpy(), expected, atol=1e-12,
                                   err_msg=name)

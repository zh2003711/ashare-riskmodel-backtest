# Excerpt from Hao Zheng's A-share quant research codebase, shared for interview review.
"""A-share daily backtest engine.

Weights set at the close of day s fill on s+1 at the close, open or VWAP. Suspended and
limit-locked names cannot trade; costs split into commission, taxes, spread and impact.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qlab.config import CostCfg
from qlab.data.panel import Panel

LIMIT = 0.095   # main-board locked-limit threshold
LIMIT_20 = 0.195   # STAR (688) since listing; ChiNext (300) from 2020-08-24
LIMIT_ST = 0.045   # main-board ST names
LIMIT_BJ = 0.295   # Beijing Stock Exchange (920)
CHINEXT_20 = pd.Timestamp("2020-08-24")
MAINBOARD_REG = pd.Timestamp("2023-04-10")  # registration-based main board: new listings limit-free for 5 bars
STAMP_HALVED = pd.Timestamp("2023-08-28")  # stamp duty 0.1% -> 0.05% (sell_bps includes 10bp)
TRANSFER_HALVED = pd.Timestamp("2022-04-29")  # transfer fee 0.02‰ -> 0.01‰, both sides
DIV_TAX = ((21, 0.20), (244, 0.10))  # holding bars -> dividend tax rate; longer = 0
TICK = 0.01   # A-share stock tick; limit prices round to it


def _limit_panel(panel: Panel) -> pd.DataFrame:
    """Limit-lock thresholds by board and date, for panels without raw prices."""
    cols = panel.close.columns
    base = np.where([c.startswith(("688", "689")) for c in cols], LIMIT_20,
                    np.where([c.startswith("920") for c in cols], LIMIT_BJ, LIMIT))
    lim = pd.DataFrame(np.tile(base, (len(panel.close), 1)),
                       index=panel.close.index, columns=cols)
    chinext = np.array([c.startswith(("300", "301", "302")) for c in cols])
    if chinext.any():
        lim.loc[lim.index >= CHINEXT_20, chinext] = LIMIT_20
    st_main = (panel.st == 1) & (lim == LIMIT)
    return lim.where(~st_main, LIMIT_ST)
DELIST_DAYS = 20  # no-price grace before a name that never trades again is written off


def lock_masks(raw_close: pd.DataFrame, raw_open: pd.DataFrame, raw_high: pd.DataFrame,
               raw_low: pd.DataFrame, adj_factor: pd.DataFrame, st: pd.DataFrame,
               days_listed: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Exact limit-lock masks from raw prices: exchange reference price, limits rounded half-up to the tick."""
    idx, cols = raw_close.index, raw_close.columns
    star = np.array([c.startswith(("688", "689")) for c in cols])
    chinext = np.array([c.startswith(("300", "301", "302")) for c in cols])
    bse = np.array([c.startswith("920") for c in cols])
    lim = pd.DataFrame(np.tile(np.where(star, 0.20, np.where(bse, 0.30, 0.10)),
                               (len(idx), 1)), index=idx, columns=cols)
    if chinext.any():
        lim.loc[idx >= CHINEXT_20, chinext] = 0.20
    lim = lim.mask((st == 1) & (lim == 0.10), 0.05)
    free5 = (star | (chinext & (idx >= CHINEXT_20)[:, None])
             | (~star & ~chinext & ~bse & (idx >= MAINBOARD_REG)[:, None]))
    new5 = (raw_close.notna().cumsum() <= 5) & (days_listed <= 12)
    lim = lim.mask(new5 & free5)

    # half-up to the tick; +1e-7 absorbs float representation (1.15*0.9 = 1.03499...)
    # without ever reaching the next genuine .005 boundary
    r01 = lambda x: np.floor(x * 100 + 0.5 + 1e-7) / 100  # noqa: E731
    ref = r01((raw_close * adj_factor).ffill().shift(1) / adj_factor)
    band_up = r01(ref * (1 + lim)) - TICK + 1e-6
    band_dn = r01(ref * (1 - lim)) + TICK - 1e-6
    return {
        "lock_up": raw_close >= band_up,
        "lock_dn": raw_close <= band_dn,
        "lock_up_open": (raw_open >= band_up) & (raw_low >= band_up),
        "lock_dn_open": (raw_open <= band_dn) & (raw_high <= band_dn),
    }


@dataclass
class BacktestResult:
    gross_ret: pd.Series
    net_ret: pd.Series
    cost: pd.Series  # fraction of NAV paid on each execution day
    turnover: pd.Series  # one-way, fraction of NAV, on execution days
    n_holdings: pd.Series
    # executable-ledger diagnostics (None in classic mode)
    fill: pd.Series | None = None        # |executed| / |desired| on execution days
    aim_gap: pd.Series | None = None     # |aim - h| after trading (unworked residual)
    cash: pd.Series | None = None        # 1 - sum(h) (cash drag from capped fills)
    impact_cost: pd.Series | None = None  # spread + sqrt-law impact (declared = cost - this)
    part_max: pd.Series | None = None    # max single-name participation that day
    # cost split: cost == fee + tax + spread + impact
    cost_fee: pd.Series | None = None     # commission, both sides (= buy_bps)
    cost_tax: pd.Series | None = None     # stamp (sells) + transfer (both) + dividend tax
    cost_spread: pd.Series | None = None  # half-spread on executed notional (surface grid)
    cost_impact: pd.Series | None = None  # sqrt-law impact on executed notional


def trade_masks(panel: Panel, execution: str = "close") -> tuple[pd.DataFrame, pd.DataFrame]:
    # no turnover = no fill opportunity
    ok = panel.close.notna() & (panel.suspended == 0) & (panel.amount > 0)
    if panel.lock_up is not None:  # exact masks from raw prices
        if execution == "open":
            return ok & ~panel.lock_up_open, ok & ~panel.lock_dn_open
        return ok & ~panel.lock_up, ok & ~panel.lock_dn   # close and vwap: sealed at the close
    lim = _limit_panel(panel)
    if execution == "open":
        # open-locked: opened at the limit and never traded off it (OHLC approximation)
        prev = panel.close.ffill().shift(1)
        up_locked = (panel.open >= prev * (1 + lim)) & (panel.low >= panel.open * 0.9999)
        dn_locked = (panel.open <= prev * (1 - lim)) & (panel.high <= panel.open * 1.0001)
    else:
        up_locked = (panel.ret > lim) & (panel.close >= panel.high * 0.9999)
        dn_locked = (panel.ret < -lim) & (panel.close <= panel.low * 1.0001)
    return ok & ~up_locked, ok & ~dn_locked  # buyable, sellable


def execute_day(h, aim, buyable_row, sellable_row, buy_bps: float, sell_bps: float,
                cap=None):
    """One day's fills toward `aim`; buys are limited to cash plus sale proceeds, net of fees.
    Returns (delta, bought, sold, cost)."""
    delta = aim - h
    delta[(delta > 0) & ~buyable_row] = 0.0
    delta[(delta < 0) & ~sellable_row] = 0.0
    if cap is not None:
        delta[np.abs(delta) < 1e-5] = 0.0
        delta = np.clip(delta, -cap, cap)
    sold = -delta[delta < 0].sum()
    buy_factor = 1.0 + buy_bps / 1e4
    buy_budget = (1.0 - h.sum()) + sold - sold * sell_bps / 1e4
    desired = delta[delta > 0].sum()
    if desired * buy_factor > buy_budget + 1e-12:
        delta[delta > 0] *= max(buy_budget, 0.0) / (desired * buy_factor)
    bought = delta[delta > 0].sum()
    cost = (bought * buy_bps + sold * sell_bps) / 1e4
    return delta, bought, sold, cost


def exec_leg(panel: Panel, execution: str) -> np.ndarray:
    """Return from the fill price to the close; zero for close execution."""
    if execution == "close":
        return np.zeros(panel.ret.shape)
    px = panel.open if execution == "open" else panel.vwap
    return (panel.close / px - 1.0).fillna(0.0).to_numpy()


def run(target: pd.DataFrame, panel: Panel, costs: CostCfg,
        aum: float | None = None, participation: float = 0.10,
        impact_c: float | np.ndarray = 1.0,
        spread_half: np.ndarray | None = None, execution: str = "close",
        div_yield: pd.DataFrame | None = None) -> BacktestResult:
    """Backtest `target` (weights on signal dates only) on `panel`.

    execution: fill price on s+1 ("close" | "open" | "vwap"). With `aum` set, each name's daily
    trade is capped at `participation` x 20-day ADV and pays spread plus square-root impact;
    `div_yield` adds dividend tax by holding period."""
    dates = panel.ret.index
    bad_syms = target.columns.difference(panel.ret.columns)
    bad_dates = target.index[target.notna().any(axis=1)].difference(dates)
    if len(bad_syms) or len(bad_dates):
        raise ValueError(
            f"target does not line up with the panel: "
            f"{len(bad_syms)} unknown symbols {list(bad_syms[:3])}, {len(bad_dates)} "
            f"signal dates off the trading calendar {list(bad_dates[:3])}"
        )
    ret = panel.ret.fillna(0.0).to_numpy()
    q = exec_leg(panel, execution)
    ret_a = (1.0 + ret) / (1.0 + q) - 1.0 if execution != "close" else ret   # close_{t-1} -> fill
    buyable_df, sellable_df = trade_masks(panel, execution)
    buyable, sellable = buyable_df.to_numpy(), sellable_df.to_numpy()
    dy = (div_yield.reindex(index=dates, columns=panel.ret.columns).fillna(0.0).to_numpy()
          if div_yield is not None else None)
    tgt = target.reindex(index=dates, columns=panel.ret.columns).to_numpy()
    is_signal = ~np.isnan(tgt).all(axis=1)
    # delisted = no price for DELIST_DAYS and none afterwards; long suspensions stay frozen
    dead = (
        (panel.close.isna().rolling(DELIST_DAYS, min_periods=DELIST_DAYS).sum()
         == DELIST_DAYS)
        & panel.close.bfill().isna()
    ).to_numpy()

    n_dates, n_sym = ret.shape
    stamp_cut = np.asarray(dates >= STAMP_HALVED)
    transfer = np.where(dates >= TRANSFER_HALVED, costs.transfer_bps / 2.0, costs.transfer_bps)
    h = np.zeros(n_sym)
    age = np.zeros(n_sym)
    gross = np.zeros(n_dates)
    cost = np.zeros(n_dates)
    turnover = np.zeros(n_dates)
    n_hold = np.zeros(n_dates)
    c_fee, c_tax, c_spr, c_imp = (np.zeros(n_dates) for _ in range(4))
    if aum is not None:
        # ADV and volatility from data up to yesterday
        adv_np = panel.amount.rolling(20, min_periods=1).mean().shift(1).to_numpy()
        sig_np = panel.ret.rolling(20, min_periods=5).std().shift(1).to_numpy()
        fill = np.full(n_dates, np.nan)
        aim_gap = np.full(n_dates, np.nan)
        cash = np.zeros(n_dates)
        impact_cost = np.zeros(n_dates)
        part_max = np.full(n_dates, np.nan)
    aim = None

    for t in range(n_dates):
        if dy is not None:                        # dividend tax on ex-dates, by holding age
            rate = np.select([age <= DIV_TAX[0][0], age <= DIV_TAX[1][0]],
                             [DIV_TAX[0][1], DIV_TAX[1][1]], 0.0)
            tax = np.where(h > 1e-8, h * dy[t] * rate, 0.0)
            h = h - tax                           # the taxed slice of the dividend never reinvests
            dtax = float(tax.sum())
            cost[t] += dtax                       # ...and leaves the NAV (repriced below)
            c_tax[t] += dtax
        g_a = h @ ret_a[t]
        h = h * (1 + ret_a[t]) / (1 + g_a)        # leg A: close_{t-1} -> fill price
        h[dead[t] & (h > 0)] = 0.0  # write off delisted holdings to cash

        if t > 0 and is_signal[t - 1]:
            aim = np.nan_to_num(tgt[t - 1])
        # classic mode trades only on the day after a signal; the aum ledger keeps working
        # the standing aim every day until the next signal replaces it
        if aim is not None and (aum is not None or (t > 0 and is_signal[t - 1])):
            cap = None
            if aum is not None:
                cap = participation * adv_np[t] / aum        # per-name daily fill capacity
                cap = np.where(np.isfinite(cap), cap, 0.0)   # no ADV -> untradable today
            # stamp cut can never make selling a rebate (zero-cost configs stay zero)
            sell_bps = max(costs.sell_bps - 5.0, 0.0) if stamp_cut[t] else costs.sell_bps
            desired = float(np.abs(aim - h).sum())
            delta, bought, sold, declared = execute_day(
                h, aim, buyable[t], sellable[t], costs.buy_bps + transfer[t],
                sell_bps + transfer[t], cap=cap)
            cost[t] += declared
            c_fee[t] = (bought + sold) * costs.buy_bps / 1e4
            c_tax[t] += declared - c_fee[t]
            if aum is not None:
                ex = np.abs(delta)
                part = np.where(adv_np[t] > 0, ex * aum / adv_np[t], 0.0)
                kc = impact_c[t] if isinstance(impact_c, np.ndarray) else impact_c
                kc = np.where(np.isfinite(kc), kc, 1.0)   # uncalibrated cells fall back to the scalar
                impact = (kc * np.nan_to_num(sig_np[t])
                          * np.sqrt(np.minimum(part, 1.0)) * np.maximum(part, 1.0))
                c_imp[t] = float((ex * impact).sum())
                if spread_half is not None:
                    c_spr[t] = float((ex * np.nan_to_num(spread_half[t])).sum())
                impact_cost[t] = c_imp[t] + c_spr[t]
                cost[t] += impact_cost[t]
                if desired > 1e-9:
                    fill[t] = float(ex.sum()) / desired
                part_max[t] = float(part.max())
            turnover[t] = max(bought, sold)
            h = h + delta
        if cost[t] > 0:            # fees and taxes leave the NAV; positions reprice against it
            h = h / (1.0 - cost[t])
        g_b = h @ q[t]
        h = h * (1 + q[t]) / (1 + g_b)            # leg B: fill price -> close_t (empty for close)
        gross[t] = g_a + g_b * (1 + g_a)
        if aum is not None:
            if aim is not None:
                aim_gap[t] = float(np.abs(aim - h).sum())
            cash[t] = 1.0 - float(h.sum())
        held = h > 1e-8
        age = np.where(held, age + 1, 0)
        n_hold[t] = held.sum()

    as_series = lambda a: pd.Series(a, index=dates)  # noqa: E731
    extra = ({"fill": as_series(fill), "aim_gap": as_series(aim_gap),
              "cash": as_series(cash), "impact_cost": as_series(impact_cost),
              "part_max": as_series(part_max)} if aum is not None else {})
    return BacktestResult(
        gross_ret=as_series(gross),
        net_ret=as_series((1 + gross) * (1 - cost) - 1),   # NAV-compound: fees shrink the NAV
        cost=as_series(cost),
        turnover=as_series(turnover),
        n_holdings=as_series(n_hold),
        cost_fee=as_series(c_fee), cost_tax=as_series(c_tax),
        cost_spread=as_series(c_spr), cost_impact=as_series(c_imp),
        **extra,
    )


def ew_benchmark(panel: Panel, mask: pd.DataFrame) -> pd.Series:
    """Equal-weight investable universe, membership lagged one day, gross."""
    return panel.ret.where(mask.shift(1)).mean(axis=1)

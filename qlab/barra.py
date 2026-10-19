# Excerpt from Hao Zheng's A-share quant research codebase (2026). Shared for interview review only;
# the data, signal, alpha-model and portfolio-optimizer layers are not included.
"""CNE6-lite style block, for answering "is this alpha or is it a style tilt?"

Eleven of CNE6's fifteen styles are buildable from what this lab holds:
Size, MidCap, Beta, Momentum, LTReversal, ResVol, Liquidity, BP, EarningsYield,
DividendYield, Profitability, EarningsQuality. GROWTH, INVESTMENT_QUALITY,
LEVERAGE and EARNINGS_VARIABILITY need income-statement history and debt detail
we never pulled -- their absence is disclosed on every report rather than
papered over, and it matters most for fundamental candidates.

`build()` returns raw daily panels; `exposures()` does the CNE6 cross-section
work at one date -- winsorize, cap-weighted-mean-zero standardisation, and the
orthogonalisations that make MidCap/ResVol/Liquidity mean what they are meant to.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

PANELS = Path("/Volumes/T9/qlab/alpha101/panels_2009_2021")
EXT = Path("/Volumes/T9/qlab/alpha101/panels_ext_2009_2026")
EXT_LEGACY = Path("/Volumes/T9/qlab/alpha101/panels_ext_2009_2021")
FUND = Path("/Volumes/T9/qlab/alpha101/fund_frontier")
STYLES = ["Size", "MidCap", "Beta", "Momentum", "LTReversal", "ResVol",
          "Liquidity", "BP", "EarnYield", "DivYield", "Profit", "EarnQuality"]
MISSING = ["Growth", "InvestmentQuality", "Leverage", "EarningsVariability"]


def _ew(df, hl, minp):
    return df.ewm(halflife=hl, min_periods=minp).mean()


def build(idx, cols, ret, cap, valid):
    """Raw (unstandardised) daily style panels aligned to (idx, cols)."""
    # pb/dv_ttm have no 2026 refresh yet: those fall back to the 2021 panels, so
    # BP/DivYield truncate at 2021-12 on extended grids (disclosed on reports);
    # Liquidity/EarnYield read the 2026 panels and cover the full grid.
    # QLAB_STYLE_PATCH (09-05 轴升级第一步): 目录里有同名面板就优先读 (pb/dv_ttm/ROE/ACC 重建到
    # 2026, scripts/build_style_inputs_2026.py); 不设 = 行为不变 (BP/DivYield/Profit/EarnQuality
    # 2022 起为空).
    patch = os.environ.get("QLAB_STYLE_PATCH")

    def ext(n):
        f = EXT / f"{n}.parquet"
        if patch and (Path(patch) / f"{n}.parquet").exists():
            f = Path(patch) / f"{n}.parquet"
        elif not f.exists():
            f = EXT_LEGACY / f"{n}.parquet"
        return pd.read_parquet(f).astype("float64").reindex(index=idx, columns=cols)

    def fnd(n):
        f = FUND / f"{n}.parquet"
        if patch and (Path(patch) / f"{n}.parquet").exists():
            f = Path(patch) / f"{n}.parquet"
        return pd.read_parquet(f).astype("float64").reindex(index=idx, columns=cols)

    R = pd.DataFrame(np.where(valid, ret, np.nan), index=idx, columns=cols)
    w = np.where(np.isfinite(cap) & (cap > 0), cap, 0.0)
    mkt = pd.Series(np.nansum(np.where(valid, ret, 0.0) * w, axis=1)
                    / np.maximum(w.sum(axis=1), 1e-9), index=idx)     # cap-weighted
    m = pd.DataFrame(np.repeat(mkt.values[:, None], len(cols), 1), index=idx, columns=cols)

    # 10-03: 每张原料算完即转 float32 并释放中间帧 (原版把 ~15 张全长 float64 留到函数末尾, 峰值 ≈5.6 GB,
    # 8 GB Mac 上任何产品臂/检验都先把机器压进交换区); 运算顺序逐项不变, 输出逐位相同.
    f32 = lambda v: np.asarray(v, dtype="float32")
    er, em = _ew(R, 63, 120), _ew(m, 63, 120)
    cov = _ew(R * m, 63, 120) - er * em
    varm = _ew(m * m, 63, 120) - em * em
    del m
    beta = cov / varm.where(varm > 0)
    del varm
    varr = _ew(R * R, 63, 120) - er * er
    del er, em
    hsigma = np.sqrt((varr - beta * cov).clip(lower=0))
    del varr, cov
    out = {"Beta": f32(beta.values), "_HSIGMA": f32(hsigma.values)}
    del beta, hsigma

    lr = np.log1p(R.clip(lower=-0.99))
    out["Momentum"] = f32((lr.ewm(halflife=126, min_periods=250).mean().shift(21) * 504).values)
    out["LTReversal"] = f32((-(lr.ewm(halflife=260, min_periods=500).mean().shift(273) * 1040)).values)
    out["_DASTD"] = f32(R.ewm(halflife=42, min_periods=120).std().values)
    del R
    # CMRA by running max/min over the 12 trailing monthly cumulations -- the
    # stack-and-concat form of this is a 1.4GB frame and will not fit here.
    csum = lr.rolling(21, min_periods=12).sum().astype("float32")
    del lr
    run = np.zeros(csum.shape, dtype="float32")
    mx = np.full(csum.shape, -np.inf, dtype="float32")
    mn = np.full(csum.shape, np.inf, dtype="float32")
    for k in range(12):
        run = run + csum.shift(21 * k).values
        mx, mn = np.fmax(mx, run), np.fmin(mn, run)
    with np.errstate(all="ignore"):
        out["_CMRA"] = f32(np.log1p(np.clip(mx, -0.99, None)) - np.log1p(np.clip(mn, -0.99, None)))
    del run, mx, mn, csum

    turn = ext("turnover_rate_f") / 100.0
    out["_STOM"] = f32(np.log(turn.rolling(21, min_periods=12).sum().clip(lower=1e-6)).values)
    out["_STOQ"] = f32(np.log((turn.rolling(63, min_periods=40).sum() / 3).clip(lower=1e-6)).values)
    out["_STOA"] = f32(np.log((turn.rolling(252, min_periods=150).sum() / 12).clip(lower=1e-6)).values)
    del turn
    pb = ext("pb"); out["BP"] = f32((1.0 / pb.where(pb > 0)).values); del pb
    pe = ext("pe_ttm"); out["EarnYield"] = f32((1.0 / pe.where(pe != 0)).values); del pe
    out["DivYield"] = f32(ext("dv_ttm").values)
    out["Profit"] = f32(fnd("ROE").values)
    out["EarnQuality"] = f32((-fnd("ACC")).values)          # low accruals = high quality
    size = np.log(np.where(cap > 0, cap, np.nan))
    out["Size"], out["MidCap"] = f32(size), f32(size ** 3)
    return {k: out[k] for k in ("Size", "MidCap", "Beta", "Momentum", "LTReversal", "_DASTD", "_CMRA", "_HSIGMA",
                                "_STOM", "_STOQ", "_STOA", "BP", "EarnYield", "DivYield", "Profit", "EarnQuality")}


def _z(x, wt):
    """CNE6 standardisation: cap-weighted mean zero, equal-weighted unit sd."""
    ok = np.isfinite(x)
    if ok.sum() < 30:
        return np.full(len(x), np.nan)
    v = np.where(ok, x, np.nan).astype("float64")
    # winsorize BEFORE computing the scale, or one outlier compresses everyone
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med))
    if mad > 0:
        v = np.clip(v, med - 5 * mad, med + 5 * mad)
    mu = np.nansum(np.where(ok, v * wt, 0.0)) / max(np.nansum(np.where(ok, wt, 0.0)), 1e-12)
    sd = np.nanstd(v)
    return np.clip((v - mu) / (sd if sd > 0 else 1.0), -3.0, 3.0)


def _orth(y, xs):
    """Residual of y on [1, *xs]; NaNs stay NaN."""
    m = np.isfinite(y) & np.all([np.isfinite(x) for x in xs], axis=0)
    if m.sum() < 30:
        return y
    A = np.column_stack([np.ones(m.sum())] + [x[m] for x in xs])
    r = np.full(len(y), np.nan)
    r[m] = y[m] - A @ np.linalg.lstsq(A, y[m], rcond=None)[0]
    return r


def exposures(raw, p, sel, cap):
    """Standardised, orthogonalised CNE6-lite exposures for names in `sel` at row p."""
    c = cap[p][sel]
    wt = np.where(np.isfinite(c) & (c > 0), c, 0.0)
    g = lambda k: raw[k][p][sel].astype("float64")
    e = {k: _z(g(k), wt) for k in
         ("Size", "Beta", "Momentum", "LTReversal", "BP", "EarnYield",
          "DivYield", "Profit", "EarnQuality")}
    e["MidCap"] = _z(_orth(_z(g("MidCap"), wt), [e["Size"]]), wt)
    rv = 0.74 * _z(g("_DASTD"), wt) + 0.16 * _z(g("_CMRA"), wt) + 0.10 * _z(g("_HSIGMA"), wt)
    e["ResVol"] = _z(_orth(rv, [e["Beta"], e["Size"]]), wt)
    lq = 0.35 * _z(g("_STOM"), wt) + 0.35 * _z(g("_STOQ"), wt) + 0.30 * _z(g("_STOA"), wt)
    e["Liquidity"] = _z(_orth(lq, [e["Size"]]), wt)
    return {k: e[k] for k in STYLES}


def style_panel(raw, name, ps, pool, cap, shape):
    """Materialise one standardised style as a (T, S) panel on the rebalance rows.

    Derived styles (MidCap / ResVol / Liquidity) exist only after the
    cross-sectional orthogonalisation, so they cannot be read out of `raw` --
    this is the only correct way to use one as a double-sort control.
    """
    out = np.full(shape, np.nan, dtype="float32")
    for i, p in enumerate(ps):
        sel = pool[i]
        if sel.sum() < 100:
            continue
        out[p][sel] = exposures(raw, p, sel, cap)[name]
    return out

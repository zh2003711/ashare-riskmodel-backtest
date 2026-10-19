# Excerpt from Hao Zheng's A-share quant research codebase, shared for interview review.
"""Daily multi-factor risk model for index enhancement: active risk = a'(XFX' + diag(d))a."""

import warnings

import numpy as np
import pandas as pd

FRET = "/Volumes/T9/qlab/alpha101/xsrc_final/env_fret.parquet"   # market-wide daily style factor returns
ALPHA_VR = 1.4   # persistence multiplier for the alpha factor


def load_fret() -> pd.DataFrame:
    return pd.read_parquet(FRET).astype("float64")


def style_vr(fret: pd.DataFrame, s, k: int = 20, win: int = 750, clip=(0.8, 2.5)) -> dict:
    """Per-style variance ratio var(k-day sum) / (k * var(daily)) over the last `win` days, clipped."""
    f = fret.loc[:s].iloc[-win:]
    return {c: float(np.clip(f[c].rolling(k).sum().var() / (k * f[c].var()), *clip)) for c in fret.columns}


def fac_vr_vector(n_lead: int, names, vr: dict) -> np.ndarray:
    """Variance multipliers in exposure-column order: industries 1, styles vr, extras 1, alpha ALPHA_VR."""
    return np.array([1.0] * n_lead + [vr.get(nm, ALPHA_VR if nm == "alpha" else 1.0) for nm in names])


def build_exposures(symbols: list[str], sectors: pd.Series, styles_z: pd.DataFrame) -> pd.DataFrame:
    """n x k exposures: industry dummies plus standardized styles."""
    sec = sectors.reindex(symbols).fillna("UNK")
    dummies = pd.get_dummies(sec).astype(float)
    dummies.index = symbols
    return pd.concat([dummies, styles_z.reindex(symbols).fillna(0.0)], axis=1)


def factor_risk_model(ret_window: pd.DataFrame, B: pd.DataFrame, halflife: int = 60,
                      shrink: float = 0.0, resid_pca: int = 0):
    """(X, F, d) from a return window with fixed exposures B (snapshot variant)."""
    X = B.to_numpy(dtype=float)
    Rraw = ret_window.reindex(columns=B.index).to_numpy(dtype=float)
    obs = ~np.isnan(Rraw)                          # suspended/pre-list days are NOT observations
    R = np.nan_to_num(Rraw, nan=0.0)
    t = np.arange(len(R))
    wts = 0.5 ** ((len(R) - 1 - t) / halflife)
    wts /= wts.sum()
    pinvX = np.linalg.pinv(X)                    # k x n
    fac_ret = R @ pinvX.T                         # T x k : f_t = pinv(X) @ ret_t
    resid = R - fac_ret @ X.T                     # T x n
    if resid_pca > 0:                             # statistical factors from residual PCA
        rw = np.sqrt(wts)[:, None] * (resid - wts @ resid)
        _, _, Vt = np.linalg.svd(rw, full_matrices=False)
        L = Vt[:resid_pca].T                      # n x k unit loadings
        f_extra = resid @ L                       # T x k statistical factor returns
        resid = resid - f_extra @ L.T
        X = np.hstack([X, L])
        fac_ret = np.hstack([fac_ret, f_extra])
    # specific variance over observed days only (suspended days would understate it)
    Wobs = wts[:, None] * obs
    part = Wobs.sum(axis=0)                        # EWMA participation in [0,1] (wts sum to 1)
    d = (Wobs * resid ** 2).sum(axis=0) / np.maximum(part, 1e-12)
    typical = np.median(d[part > 0.5]) if (part > 0.5).any() else np.median(d)
    d = np.where(part < 0.2, typical, d)          # too few obs to estimate -> a typical name's d
    d = np.maximum(d, typical * 1e-3 + 1e-8)       # tiny floor to keep Sigma PD
    fc = fac_ret - wts @ fac_ret                  # demean by EWMA mean
    F = (fc * wts[:, None]).T @ fc                 # EWMA factor covariance
    if shrink > 0:                                # a-priori shrink toward the diagonal
        F = shrink * np.diag(np.diag(F)) + (1 - shrink) * F
    F = 0.5 * (F + F.T)
    w, V = np.linalg.eigh(F)                       # PSD-project (clip tiny negatives)
    F = (V * np.maximum(w, 0.0)) @ V.T
    return X, F, d


def _cs_vra(sq, obs, hl_sig, hl_b):
    """Volatility regime multiplier: EWMA of the cross-sectional bias mean(r^2 / sigma^2_{t-1})."""
    x = np.where(obs, sq, np.nan)
    sig2 = pd.DataFrame(x).ewm(halflife=hl_sig, min_periods=hl_sig, ignore_na=True).mean().shift(1).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # warm-up rows are all NaN
        b2 = np.nanmean(x / sig2, axis=1)
    lam2 = pd.Series(b2).ewm(halflife=hl_b, min_periods=5, ignore_na=True).mean().iloc[-1]
    return float(np.clip(lam2, 0.25, 9.0)) if np.isfinite(lam2) else 1.0


def factor_risk_model_daily(ret_window: pd.DataFrame, dummies: pd.DataFrame,
                            styles_daily: dict, halflife: int = 60,
                            nw_lags: int = 0, shrink: float = 0.0, resid_pca: int = 0,
                            vra_hl: int = 0, fac_vr=None, corr_hl: int = 0):
    """Daily factor returns from one-day-lagged exposures, then `risk_from_daily`."""
    Rraw = ret_window.to_numpy(dtype=float)
    obs = ~np.isnan(Rraw)
    R = np.nan_to_num(Rraw, nan=0.0)
    Dm = dummies.to_numpy(dtype=float)
    S = np.stack([styles_daily[nm].to_numpy(dtype=float) for nm in styles_daily], axis=-1)
    T = R.shape[0]
    fac, resid = [], []
    for t in range(1, T):
        X_t = np.hstack([Dm, np.nan_to_num(S[t - 1])])
        f = np.linalg.pinv(X_t) @ R[t]
        fac.append(f)
        resid.append(R[t] - X_t @ f)
    X_last = np.hstack([Dm, np.nan_to_num(S[-1])])
    return risk_from_daily(np.array(fac), np.array(resid), obs[1:], X_last, halflife=halflife, nw_lags=nw_lags,
                           shrink=shrink, resid_pca=resid_pca, vra_hl=vra_hl, fac_vr=fac_vr, corr_hl=corr_hl)


def risk_from_daily(fac, resid, obs_r, X_last, halflife: int = 60, nw_lags: int = 0, shrink: float = 0.0,
                    resid_pca: int = 0, vra_hl: int = 0, fac_vr=None, corr_hl: int = 0):
    """(X, F, d) from a window of daily factor returns and residuals, with optional adjustments."""
    fac, resid = np.asarray(fac, float), np.asarray(resid, float)
    obs_r = np.asarray(obs_r, bool)
    t_idx = np.arange(len(fac))
    wts = 0.5 ** ((len(fac) - 1 - t_idx) / halflife)
    wts /= wts.sum()
    L = None
    if resid_pca > 0:                          # statistical factors from residual PCA
        rw = np.sqrt(wts)[:, None] * (resid - wts @ resid)
        _, _, Vt = np.linalg.svd(rw, full_matrices=False)
        L = Vt[:resid_pca].T
        f_extra = resid @ L
        resid = resid - f_extra @ L.T
        fac = np.hstack([fac, f_extra])
    Wobs = wts[:, None] * obs_r
    part = Wobs.sum(axis=0)
    d = (Wobs * resid ** 2).sum(axis=0) / np.maximum(part, 1e-12)
    typical = np.median(d[part > 0.5]) if (part > 0.5).any() else np.median(d)
    d = np.where(part < 0.2, typical, d)
    d = np.maximum(d, typical * 1e-3 + 1e-8)
    fc = fac - wts @ fac
    F = (fc * wts[:, None]).T @ fc
    if corr_hl > 0:                            # correlations from a longer half-life, variances unchanged
        w2 = 0.5 ** ((len(fac) - 1 - t_idx) / corr_hl); w2 /= w2.sum()
        f2 = fac - w2 @ fac
        F2 = (f2 * w2[:, None]).T @ f2
        s1, s2 = np.sqrt(np.maximum(np.diag(F), 1e-18)), np.sqrt(np.maximum(np.diag(F2), 1e-18))
        F = F2 / np.outer(s2, s2) * np.outer(s1, s1)
    if fac_vr is not None:                     # scale variances to the holding period (PCA columns unscaled)
        mlt = np.ones(F.shape[0]); mlt[:len(fac_vr)] = np.asarray(fac_vr, float); sq_ = np.sqrt(mlt)
        F = F * np.outer(sq_, sq_)
    lam_f = lam_d = 1.0
    if vra_hl > 0:                             # volatility regime adjustment
        lam_f = _cs_vra(fc ** 2, np.ones_like(fc, dtype=bool), halflife, vra_hl)
        lam_d = _cs_vra(resid ** 2, obs_r, halflife, vra_hl)
        F = F * lam_f
        d = d * lam_d
    factor_risk_model_daily.last_vra = (lam_f, lam_d)
    for k in range(1, nw_lags + 1):            # Newey-West autocorrelation terms
        g = (fc[k:] * wts[k:, None]).T @ fc[:-k]
        F += (1 - k / (nw_lags + 1)) * (g + g.T) * (lam_f if vra_hl > 0 else 1.0)
    if shrink > 0:                             # a-priori shrink toward the diagonal
        F = shrink * np.diag(np.diag(F)) + (1 - shrink) * F
    F = 0.5 * (F + F.T)
    w_eig, V = np.linalg.eigh(F)
    F = (V * np.maximum(w_eig, 0.0)) @ V.T
    X = np.asarray(X_last, float)
    if L is not None:
        X = np.hstack([X, L])
    return X, F, d


class DailyRegressions:
    """Daily cross-sectional regressions on that day's index members, computed once and reused."""

    def __init__(self, ret: pd.DataFrame, style_names, style_at, sector_at, members_at, ind_names, alpha_at=None):
        self.ret, self.cols = ret, ret.columns
        self.style_names, self.style_at, self.sector_at, self.members_at, self.alpha_at = list(style_names), style_at, sector_at, members_at, alpha_at
        self.ind_names = list(ind_names)
        self.names = self.style_names + (["alpha"] if alpha_at is not None else [])
        self.k = len(self.ind_names) + len(self.names)
        T, N = ret.shape
        self.fac = np.full((T, self.k), np.nan)
        self.resid = np.zeros((T, N))
        self.obs = np.zeros((T, N), bool)
        self.built = np.zeros(T, bool)

    def exposures(self, day):
        """Exposures as of `day`: industry dummies, styles and alpha residual standardized within members."""
        cols, N = self.cols, len(self.cols)
        m = cols.isin(list(self.members_at(day)))
        sec = self.sector_at(day).reindex(cols).fillna("UNK")
        j = pd.Index(self.ind_names).get_indexer(sec.to_numpy())
        j = np.where(j < 0, self.ind_names.index("UNK"), j)
        D = np.zeros((N, len(self.ind_names))); D[np.arange(N), j] = 1.0
        Z = np.zeros((N, len(self.style_names)))
        for q, nm in enumerate(self.style_names):
            v = self.style_at(nm, day)
            vm = v[m]
            mu, sd = np.nanmean(vm), np.nanstd(vm, ddof=1)
            Z[:, q] = np.nan_to_num((v - mu) / (sd + 1e-9))
        E = np.hstack([D, Z])
        if self.alpha_at is not None:
            a = self.alpha_at(day)
            if a is None or not np.isfinite(a[m]).any():
                E = np.hstack([E, np.zeros((N, 1))])
            else:
                a = np.nan_to_num(a - np.nanmean(a[m]))
                r = a - E @ np.linalg.lstsq(E[m], a[m], rcond=None)[0]
                E = np.hstack([E, (r / (r[m].std() + 1e-12))[:, None]])
        return E, m

    def build(self, i0: int, i1: int):
        """Regress days i0..i1 (panel positions); day t uses exposures as of t-1."""
        idx, R = self.ret.index, self.ret.to_numpy(dtype=float)
        E_prev, _ = self.exposures(idx[i0 - 1])
        for t in range(i0, i1 + 1):
            if not self.built[t]:
                obs = np.isfinite(R[t])
                m = self.cols.isin(list(self.members_at(idx[t])))
                rows = obs & m
                f = np.linalg.lstsq(E_prev[rows], R[t][rows], rcond=None)[0]
                self.fac[t] = f
                self.resid[t] = np.where(obs, np.nan_to_num(R[t]) - E_prev @ f, 0.0)
                self.obs[t] = obs
                self.built[t] = True
            E_prev, _ = self.exposures(idx[t])
        self.E_last_day, self.E_last = idx[i1], E_prev

    def window(self, i: int, risk_window: int, u_idx):
        """Factor returns, residuals and observation mask for the window ending at day i, plus X."""
        t0 = max(i - risk_window + 2, 0)
        assert self.built[t0:i + 1].all(), "DailyRegressions.build() must cover the window"
        E, _ = self.exposures(self.ret.index[i])
        return self.fac[t0:i + 1], self.resid[t0:i + 1][:, u_idx], self.obs[t0:i + 1][:, u_idx], E[u_idx]

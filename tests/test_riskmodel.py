# Excerpt from Hao Zheng's A-share quant research codebase, shared for interview review.
"""Risk model tests on synthetic data with known answers."""

import numpy as np
import pandas as pd

from qlab.portfolio.riskmodel import (DailyRegressions, build_exposures, factor_risk_model, factor_risk_model_daily,
                                      risk_from_daily)

T, N = 600, 120
RNG = np.random.default_rng(3)
SECT = pd.Series(["A", "B", "C"] * (N // 3), index=[f"s{i}" for i in range(N)])
STYLE = pd.DataFrame({"size": RNG.normal(size=N)}, index=SECT.index)
B = build_exposures(list(SECT.index), SECT, STYLE)
FVAR = np.array([4e-4, 4e-4, 4e-4, 1e-4])      # A, B, C dummies then size
DVAR = 2.5e-4


def _returns():
    f = RNG.normal(size=(T, 4)) * np.sqrt(FVAR)
    e = RNG.normal(size=(T, N)) * np.sqrt(DVAR)
    return pd.DataFrame(f @ B.to_numpy().T + e, columns=SECT.index)


def test_recovers_factor_and_specific_variance_with_flat_weights():
    R = _returns()
    X, F, d = factor_risk_model(R, B, halflife=1e9)      # flat EWMA = sample covariance
    assert X.shape == (N, 4) and F.shape == (4, 4)
    # industry dummies are collinear with the intercept-free size column only through
    # pinv; diag recovery within sampling error (T = 600 -> ~6% se on a variance)
    np.testing.assert_allclose(np.diag(F), FVAR, rtol=0.35)
    assert abs(np.median(d) / DVAR - 1) < 0.25
    assert np.linalg.eigvalsh(F).min() >= -1e-12


def test_suspended_name_keeps_a_typical_specific_variance():
    R = _returns()
    R.iloc[::2, 0] = np.nan                                # name 0 unobserved half the window
    R.iloc[:, 1] = np.nan                                  # name 1 never observed
    R.iloc[T // 2:, 1] = 0.0
    _, _, d = factor_risk_model(R, B, halflife=60)
    typical = np.median(d[2:])
    assert d[0] > 0.5 * typical                            # not collapsed by zero-filled halts
    assert d[1] == d[2:].max() or d[1] >= 0.5 * typical   # too few obs -> typical, never ~0


def test_daily_variant_matches_snapshot_variant_on_static_styles():
    R = _returns()
    dummies = pd.get_dummies(SECT).astype(float)
    styles_daily = {"size": pd.DataFrame(np.tile(STYLE["size"].to_numpy(), (T, 1)),
                                         index=R.index, columns=SECT.index)}
    X1, F1, d1 = factor_risk_model(R, B, halflife=1e9)
    X2, F2, d2 = factor_risk_model_daily(R, dummies, styles_daily, halflife=1e9)
    np.testing.assert_allclose(np.diag(F2), np.diag(F1), rtol=0.05)
    np.testing.assert_allclose(np.median(d2), np.median(d1), rtol=0.05)
    assert np.linalg.eigvalsh(F2).min() >= -1e-12


def test_newey_west_changes_only_with_autocorrelation():
    R = _returns()
    dummies = pd.get_dummies(SECT).astype(float)
    sd = {"size": pd.DataFrame(np.tile(STYLE["size"].to_numpy(), (T, 1)),
                               index=R.index, columns=SECT.index)}
    _, F0, _ = factor_risk_model_daily(R, dummies, sd, halflife=1e9, nw_lags=0)
    _, F5, _ = factor_risk_model_daily(R, dummies, sd, halflife=1e9, nw_lags=5)
    # iid factor returns: the NW correction is sampling noise, not a level shift
    assert abs(np.trace(F5) / np.trace(F0) - 1) < 0.15


def _daily_inputs(R):
    dummies = pd.get_dummies(SECT).astype(float)
    sd = {"size": pd.DataFrame(np.tile(STYLE["size"].to_numpy(), (T, 1)), index=R.index, columns=SECT.index)}
    return dummies, sd


def test_daily_cache_matches_window_estimator():
    # the per-day regression cache reproduces the window estimator
    R = _returns()
    ST = pd.DataFrame(RNG.normal(size=(T, N)).cumsum(axis=0) / 10, columns=SECT.index)
    A = pd.DataFrame(RNG.normal(size=(T, N)), columns=SECT.index)
    W, i = 120, T - 1
    win = R.index[i - W + 1:i + 1]
    dummies = pd.get_dummies(SECT).astype(float)
    sz = ST.loc[win]; sz = sz.sub(sz.mean(axis=1), axis=0).div(sz.std(axis=1) + 1e-9, axis=0)
    Aw = A.loc[win].to_numpy(); Aw = Aw - Aw.mean(axis=1, keepdims=True)
    Hh = np.zeros_like(Aw)
    for t_ in range(W):
        X_t = np.hstack([dummies.to_numpy(), sz.to_numpy()[t_][:, None]])
        r_t = Aw[t_] - X_t @ np.linalg.lstsq(X_t, Aw[t_], rcond=None)[0]
        Hh[t_] = r_t / (r_t.std() + 1e-12)
    sd = {"size": sz, "alpha": pd.DataFrame(Hh, index=win, columns=SECT.index)}
    X0, F0, d0 = factor_risk_model_daily(R.loc[win], dummies, sd, halflife=60, resid_pca=3, fac_vr=[1, 1, 1, 1.3, 1.4])
    dr = DailyRegressions(R, ["size"], lambda nm, d: ST.loc[d].to_numpy(), lambda d: SECT, lambda d: SECT.index,
                          ["A", "B", "C", "UNK"], lambda d: A.loc[d].to_numpy())
    dr.build(i - W + 2, i)
    X1, F1, d1 = risk_from_daily(*dr.window(i, W, np.arange(N)), halflife=60, resid_pca=3, fac_vr=[1, 1, 1, 1, 1.3, 1.4])
    np.testing.assert_allclose(d1, d0, rtol=1e-8)
    a = RNG.normal(size=(5, N)); a -= a.mean(axis=1, keepdims=True)
    v0 = [ai @ X0 @ F0 @ X0.T @ ai + (d0 * ai ** 2).sum() for ai in a]
    v1 = [ai @ X1 @ F1 @ X1.T @ ai + (d1 * ai ** 2).sum() for ai in a]
    np.testing.assert_allclose(v1, v0, rtol=1e-7)


def test_member_mask_keeps_non_members_out_of_the_cross_section():
    # non-members' contaminated returns stay out of the factor covariance
    R = _returns()
    bad = np.zeros((T, N), bool); bad[: T // 2, :30] = True
    R.values[bad] += RNG.normal(size=int(bad.sum())) * 0.5
    dummies, sd = _daily_inputs(R)
    _, F_all, _ = factor_risk_model_daily(R, dummies, sd, halflife=1e9)
    members = lambda d: SECT.index[30:] if d < T // 2 else SECT.index
    dr = DailyRegressions(R, ["size"], lambda nm, d: STYLE["size"].to_numpy(), lambda d: SECT, members, ["A", "B", "C", "UNK"])
    dr.build(1, T - 1)
    _, F_mem, _ = risk_from_daily(*dr.window(T - 1, T, np.arange(N)), halflife=1e9)
    assert np.trace(F_mem) < 0.5 * np.trace(F_all)
    np.testing.assert_allclose(np.diag(F_mem)[[0, 1, 2, 4]], FVAR, rtol=0.35)

"""Physics witnesses: what the feeder *should* look like given what was reported.

Two checks, both computable by a utility that knows its own feeder topology:
  1. Voltage twin   - rebuild each meter's voltage from the transformer voltage
                      and every reported current (Ohm's law along the feeder).
  2. Energy balance - transformer power must equal the sum of meter powers
                      plus estimated cable losses (Kirchhoff / Tellegen).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .config import DEFAULT, Config


def voltage_twin(tx_v: np.ndarray, meter_i: np.ndarray, cfg: Config = DEFAULT):
    """Expected meter voltages (T, N) from transformer voltage + reported currents."""
    fd = cfg.feeder
    i_down = np.cumsum(meter_i[:, ::-1], axis=1)[:, ::-1]
    v_node = tx_v[:, None] - np.cumsum(fd.r_segment * i_down, axis=1)
    return v_node - fd.r_service * meter_i


def local_residual(meter_v, v_hat, offset=None, k=2, untrusted=None):
    """The part of a meter's twin residual that belongs to that meter alone.

    A lying or wrong current at meter m shifts the twin of every meter
    downstream of m by the same amount: a *step* along the feeder. A drifting
    sensor or a bad connection at meter j makes j stand out from *both* sides.
    So we compare j with the median of its k upstream and k downstream
    neighbours separately and keep the smaller deviation when both agree in
    sign (zero otherwise). Steps cancel, single-meter problems survive.
    Meters at the feeder ends only have one side, so they use k+1 neighbours
    there; meters already marked untrusted never serve as a reference.
    """
    r = meter_v - v_hat
    if offset is not None:
        r = r - offset
    ref = r.copy()
    if untrusted is not None:
        ref[untrusted] = np.nan
    N = r.shape[1]
    out = np.empty_like(r)

    def dev(j, idx):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN rows
            med = np.nanmedian(ref[:, idx], axis=1)
        return r[:, j] - np.nan_to_num(med, nan=0.0)

    for j in range(N):
        up = [n for n in range(j - k, j) if n >= 0]
        dn = [n for n in range(j + 1, j + k + 1) if n < N]
        if not up or not dn:
            side = list(range(1, k + 2)) if not up else list(range(N - k - 2, N - 1))
            out[:, j] = dev(j, side)
            continue
        d_up, d_dn = dev(j, up), dev(j, dn)
        same = np.sign(d_up) == np.sign(d_dn)
        out[:, j] = np.where(same, np.sign(d_up) * np.minimum(abs(d_up), abs(d_dn)), 0.0)
    return out


def balance_gap(tx_p, meter_p, meter_i, cfg: Config = DEFAULT):
    """Unexplained power at the transformer, kW, per second."""
    fd = cfg.feeder
    i_down = np.cumsum(meter_i[:, ::-1], axis=1)[:, ::-1]
    losses_kw = ((fd.r_segment * i_down ** 2).sum(axis=1)
                 + (fd.r_service * meter_i ** 2).sum(axis=1)) / 1000
    return tx_p - meter_p.sum(axis=1) - losses_kw


def gap_onsets(gap: np.ndarray, meter_p: np.ndarray, step_kw: float, half: int = 8,
               untrusted=None):
    """Find the second a balance gap opens and the meter that opened it.

    When a real load switches off, the transformer and the meter both see it,
    so the gap does not move. When a meter starts under-reporting, its reading
    falls while the transformer reading does not: the gap jumps up by exactly
    the amount the reading fell, at the same second. We look for those jumps
    and match each one to the meter whose reading moved the opposite way.

    Returns a list of (t, meter, d_gap_kw, match_ratio).
    """
    T = len(gap)
    out = []
    t = half
    while t < T - half:
        before = np.median(gap[t - half:t])
        after = np.median(gap[t:t + half])
        d_gap = after - before
        if abs(d_gap) > step_kw:
            # pin the exact second inside the step
            seg = gap[t - half:t + half]
            t_on = t - half + int(np.argmax(np.abs(np.diff(seg)))) + 1
            d_gap = np.median(gap[t_on:t_on + half]) - np.median(gap[max(0, t_on - half):t_on])
            dP = (np.median(meter_p[t_on:t_on + half], axis=0)
                  - np.median(meter_p[max(0, t_on - half):t_on], axis=0))
            ratio = -dP / d_gap                     # 1.0 = perfect opposite move
            bad = (untrusted[t_on] if untrusted is not None
                   else np.zeros(len(ratio), dtype=bool))
            # a meter that already lost its pulse can move the gap on its own;
            # if one of them jumped at this second, the jump is not evidence
            explained_by_liar = bad.any() and np.max(np.abs(ratio[bad])) > 0.3
            ratio = np.where(bad, np.inf, ratio)
            j = int(np.argmin(np.abs(ratio - 1)))
            if 0.6 < ratio[j] < 1.4 and not explained_by_liar:
                out.append((t_on, j, float(d_gap), float(ratio[j])))
            t = t_on + half
        else:
            t += 1
    return out


def bias_suspects(gap, meter_p, step_kw, level_kw, untrusted=None):
    """(T, N) boolean map: meter j is the attributed cause of an open gap at t.

    A suspicion opens at an upward gap jump matched to meter j and closes when
    a matched downward jump on the same meter happens or the gap closes.
    """
    T, N = meter_p.shape
    sus = np.zeros((T, N), dtype=bool)
    onsets = gap_onsets(gap, meter_p, step_kw, untrusted=untrusted)
    smooth = pd.Series(gap).rolling(15, center=True, min_periods=1).median().to_numpy()
    for k, (t, j, d, ratio) in enumerate(onsets):
        # only a jump that leaves energy unaccounted for opens a suspicion; a
        # jump back to zero (a lying meter going honest again) does not
        if d <= 0 or np.median(gap[t:t + 8]) < level_kw:
            continue
        end = T
        for t2, j2, d2, _ in onsets[k + 1:]:
            if j2 == j and d2 < 0:
                end = t2
                break
        closed = np.nonzero(smooth[t + 8:end] < 0.5 * level_kw)[0]
        if len(closed):
            end = t + 8 + int(closed[0])
        sus[t:end, j] = True
    return sus, onsets


def impute_untrusted(meter_i, tx_i, untrusted):
    """Kirchhoff imputation: replace currents we no longer trust.

    The transformer current is the sum of all meter currents, so whatever the
    trusted meters do not explain belongs to the untrusted ones (shared
    equally when several are untrusted at the same second).
    """
    out = meter_i.copy()
    n_bad = untrusted.sum(axis=1)
    rows = n_bad > 0
    if rows.any():
        trusted_sum = np.where(untrusted, 0.0, meter_i).sum(axis=1)
        share = (tx_i - trusted_sum)[rows] / n_bad[rows]
        out[rows] = np.where(untrusted[rows], share[:, None], meter_i[rows])
    return out


def attribute_gap(gap: np.ndarray, meter_p: np.ndarray):
    """Which meter best explains a balance gap? Fit gap ~ alpha * P_j per meter.

    An under-reporting meter shows reported P = k * P_true, so the gap equals
    (1-k)/k * P_reported: a clean proportional fit with alpha > 0.
    We fit on second-to-second *changes*: every time the guilty meter's
    appliance switches, the gap jumps by a proportional amount at the same
    second, while other meters' switching leaves the gap untouched.
    Returns (alpha, r2, alpha_level) arrays over meters.
    """
    g = np.diff(gap)
    P = np.diff(meter_p, axis=0)
    g = g - g.mean()
    P = P - P.mean(axis=0)
    var = (P ** 2).sum(axis=0) + 1e-9
    alpha_dyn = (P * g[:, None]).sum(axis=0) / var
    r2 = alpha_dyn ** 2 * var / ((g ** 2).sum() + 1e-9)
    # level fit (through origin), reported for the explanation
    alpha_lvl = (meter_p * gap[:, None]).sum(axis=0) / ((meter_p ** 2).sum(axis=0) + 1e-9)
    return alpha_dyn, np.clip(r2, 0, 1), alpha_lvl

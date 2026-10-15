"""Pulse checks: does a meter share the grid's and the feeder's heartbeat?

We never compare raw values here, only second-to-second changes. A meter
drifting by 5 V still has the right pulse; a fabricated stream at a perfect
231 V does not.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _corr_cols(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Pearson correlation of every column of x (n, N) with ref (n,)."""
    xc = x - x.mean(axis=0)
    rc = ref - ref.mean()
    den = np.sqrt((xc ** 2).sum(axis=0) * (rc ** 2).sum())
    out = np.zeros(x.shape[1])
    ok = den > 1e-12
    out[ok] = (xc[:, ok] * rc[:, None]).sum(axis=0) / den[ok]
    return out


def grid_pulse(meter_f_win: np.ndarray, tx_f_win: np.ndarray) -> np.ndarray:
    """Correlation of frequency changes with the reference (one value per meter)."""
    return _corr_cols(np.diff(meter_f_win, axis=0), np.diff(tx_f_win))


def feeder_pulse(meter_v_win: np.ndarray, v_hat_win: np.ndarray) -> np.ndarray:
    """Correlation of each meter's voltage changes with its physics twin."""
    dm = np.diff(meter_v_win, axis=0)
    dh = np.diff(v_hat_win, axis=0)
    dmc, dhc = dm - dm.mean(0), dh - dh.mean(0)
    den = np.sqrt((dmc ** 2).sum(0) * (dhc ** 2).sum(0))
    return np.where(den > 1e-12, (dmc * dhc).sum(0) / np.maximum(den, 1e-12), 0.0)


def frozen_fraction(v_win: np.ndarray, i_win: np.ndarray) -> np.ndarray:
    """Share of seconds where neither voltage nor current changed at all."""
    dv = np.diff(v_win, axis=0) == 0
    di = np.diff(i_win, axis=0) == 0
    return (dv & di).mean(axis=0)


def find_replay_lag(meter_f_win: np.ndarray, tx_f: np.ndarray, start: int,
                    min_lag: int, max_lag: int):
    """Search the reference history for the moment this pulse was really recorded.

    Returns (best_lag_seconds, best_correlation). A high correlation at a
    non-zero lag means genuine data, re-sent later: a replay.
    """
    x = np.diff(meter_f_win)
    n = len(x)
    max_lag = min(max_lag, start)
    if max_lag < min_lag or np.std(x) < 1e-12:
        return 0, 0.0
    d = np.diff(tx_f)
    lags = np.arange(min_lag, max_lag + 1)
    cand = sliding_window_view(d, n)[start - lags]          # (n_lags, n)
    xc = x - x.mean()
    cc = cand - cand.mean(axis=1, keepdims=True)
    corr = (cc @ xc) / (np.sqrt((cc ** 2).sum(1) * (xc ** 2).sum()) + 1e-12)
    k = int(np.argmax(corr))
    return int(lags[k]), float(corr[k])

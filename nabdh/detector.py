"""NABDH detector: pulse + physics features, calibration, verdicts.

Pipeline per window (60 s, hop 30 s), for every meter at once:

  1. Feeder-level context: is the whole grid doing something right now?
     (frequency excursion, transformer voltage sag, energy-balance gap)
  2. Per-meter evidence: grid pulse, feeder pulse, frozen fraction,
     voltage-twin residual, residual-vs-current correlation.
  3. A fixed, readable rule ladder turns evidence into one verdict.
  4. An Isolation Forest trained on clean windows is the safety net for
     anything the rules do not describe (verdict UNEXPLAINED).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from . import physics, pulse
from .config import (DEFAULT, Config, NORMAL, GRID_EVENT, STUCK, DRIFT, BIAS,
                     FABRICATED, REPLAY, EQUIPMENT_FAULT, UNEXPLAINED)
from .simulator import Episode

# Per-meter evidence only. The balance gap is feeder-wide, so it would make
# every meter look odd whenever one meter lies; it is used by the BIAS rule instead.
IF_FEATURES = ["f_corr", "v_corr", "res_abs"]


@dataclass
class Thresholds:
    f_corr: float
    v_corr: float
    res_abs: float
    gap_abs: float
    if_score: float
    res_offset: list

    def to_json(self):
        return json.dumps(asdict(self), indent=2)


def _sig(x):
    return float(1 / (1 + np.exp(-x)))


def episode_features(ep: Episode, cfg: Config = DEFAULT, res_offset=None,
                     untrusted=None) -> pd.DataFrame:
    """One row per (window, meter) with every piece of evidence the rules use.

    `untrusted` (T, N) marks seconds where a meter already failed the pulse
    test; its current is then rebuilt from Kirchhoff before the voltage twin
    is computed, so one liar cannot distort its neighbours' physics checks.
    """
    dc = cfg.detector
    T, N = ep.meter_v.shape
    i_used = ep.meter_i if untrusted is None else physics.impute_untrusted(
        ep.meter_i, ep.tx_i, untrusted)
    v_hat = physics.voltage_twin(ep.tx_v, i_used, cfg)
    res = physics.local_residual(ep.meter_v, v_hat, res_offset, untrusted=untrusted)
    gap = physics.balance_gap(ep.tx_p, ep.meter_p, ep.meter_i, cfg)
    suspects, _ = physics.bias_suspects(gap, ep.meter_p, step_kw=0.25, level_kw=0.15,
                                        untrusted=untrusted)
    rows = []
    for s in range(0, T - dc.window_s + 1, dc.step_s):
        w = slice(s, s + dc.window_s)
        lb = slice(max(0, s + dc.window_s - dc.lookback_s), s + dc.window_s)
        f_corr = pulse.grid_pulse(ep.meter_f[w], ep.tx_f[w])
        v_corr = pulse.feeder_pulse(ep.meter_v[w], v_hat[w])
        frozen = pulse.frozen_fraction(ep.meter_v[w], ep.meter_i[w])
        r_w = res[w]
        r_lb, i_lb = res[lb], ep.meter_i[lb]
        ri_corr = np.array([np.corrcoef(r_lb[:, j], i_lb[:, j])[0, 1]
                            if i_lb[:, j].std() > 1e-6 and r_lb[:, j].std() > 1e-6 else 0.0
                            for j in range(N)])
        alpha_dyn, r2, alpha_lvl = physics.attribute_gap(gap[lb], ep.meter_p[lb])
        bias_sus = suspects[w].mean(axis=0)
        f_dev = np.abs(ep.tx_f[w] - cfg.feeder.f_nominal).max()
        v_ref = np.median(ep.tx_v[max(0, s - 300):s + 1])
        sag_pct = 100 * (1 - ep.tx_v[w].min() / v_ref)
        for j in range(N):
            rows.append(dict(
                episode=ep.spec.name, t0=s, meter=j,
                f_corr=f_corr[j], v_corr=v_corr[j], frozen=frozen[j],
                res_mean=r_w[:, j].mean(), res_abs=abs(r_w[:, j].mean()),
                res_std=r_w[:, j].std(), ri_corr=ri_corr[j],
                gap_mean=gap[w].mean(), gap_abs=abs(gap[w].mean()),
                gap_alpha=alpha_dyn[j], gap_r2=r2[j], gap_alpha_lvl=alpha_lvl[j],
                bias_sus=bias_sus[j],
                f_dev=f_dev, sag_pct=sag_pct,
            ))
    return pd.DataFrame(rows)


class NabdhDetector:
    def __init__(self, cfg: Config = DEFAULT):
        self.cfg = cfg
        self.th: Thresholds | None = None
        self.forest: IsolationForest | None = None
        self._gap_center = 0.0

    # ---------- calibration on clean data ----------
    def fit(self, clean_episodes: list[Episode]):
        dc = self.cfg.detector
        # per-meter residual offset (model mismatch, meter calibration at install)
        offs = []
        for ep in clean_episodes:
            v_hat = physics.voltage_twin(ep.tx_v, ep.meter_i, self.cfg)
            offs.append(np.median(ep.meter_v - v_hat, axis=0))
        res_offset = np.mean(offs, axis=0)
        feats = pd.concat([episode_features(ep, self.cfg, res_offset) for ep in clean_episodes])
        self._gap_center = float(feats.gap_mean.median())
        feats["gap_abs"] = (feats.gap_mean - self._gap_center).abs()
        q = dc.pulse_quantile
        self.forest = IsolationForest(n_estimators=200, random_state=0).fit(feats[IF_FEATURES])
        scores = -self.forest.score_samples(feats[IF_FEATURES])
        self.th = Thresholds(
            f_corr=float(min(feats.f_corr.quantile(q), 0.5)),
            v_corr=float(min(feats.v_corr.quantile(q), 0.5)),
            res_abs=float(max(feats.res_abs.quantile(dc.residual_quantile) * 1.5, 0.3)),
            gap_abs=float(max(feats.gap_abs.quantile(dc.residual_quantile) * 1.5, 0.15)),
            # safety net, kept conservative: clean data never scores above max + margin
            if_score=float(scores.max() + dc.if_margin),
            res_offset=res_offset.round(4).tolist(),
        )
        return self

    # ---------- inference ----------
    def detect(self, ep: Episode) -> pd.DataFrame:
        t_start = time.perf_counter()
        th, dc = self.th, self.cfg.detector
        offset = np.array(th.res_offset)
        # pass 1: who has lost the pulse or frozen? pass 2: physics without them
        first = episode_features(ep, self.cfg, offset)
        T, N = ep.meter_v.shape
        untrusted = np.zeros((T, N), dtype=bool)
        bad = first[(first.f_corr < th.f_corr) | (first.v_corr < th.v_corr) | (first.frozen > 0.5)]
        for t0, j in zip(bad.t0, bad.meter):
            untrusted[t0:t0 + dc.window_s, j] = True
        feats = episode_features(ep, self.cfg, offset, untrusted)
        feats["gap_abs"] = (feats.gap_mean - self._gap_center).abs()
        feats["if_score"] = -self.forest.score_samples(feats[IF_FEATURES])

        verdicts, confs, lags, lag_corrs = [], [], [], []
        for r in feats.itertuples(index=False):
            v, c, lag, lc = self._rule_ladder(r, ep)
            verdicts.append(v); confs.append(c); lags.append(lag); lag_corrs.append(lc)
        feats["verdict"] = verdicts
        feats["confidence"] = np.round(confs, 3)
        feats["replay_lag"] = lags
        feats["replay_corr"] = np.round(lag_corrs, 3)
        n_windows = feats.t0.nunique()
        feats.attrs["ms_per_window"] = 1000 * (time.perf_counter() - t_start) / max(n_windows, 1)
        return feats

    def _bias_suspect(self, r):
        th = self.th
        return r.gap_abs > th.gap_abs and r.bias_sus > 0.5

    def _rule_ladder(self, r, ep: Episode):
        """Ordered rules. The first one that fires gives the verdict.

        Order matters and is deliberate: a dead sensor has no pulse, so STUCK is
        tested before the pulse; a sensor without the right pulse cannot be
        trusted for physics checks, so the pulse comes before drift/bias/fault.
        """
        th, dc = self.th, self.cfg.detector
        grid_event = r.f_dev > dc.freq_event_hz or r.sag_pct > dc.sag_event_pct

        if r.frozen > 0.5:
            return STUCK, min(0.99, r.frozen), 0, 0.0

        if r.f_corr >= th.f_corr and r.v_corr < th.v_corr and r.res_mean < -th.res_abs:
            # grid pulse intact, local voltage disturbed and pulled down:
            # a real meter seeing a real local problem (arcing / loose joint)
            return EQUIPMENT_FAULT, min(0.99, _sig((r.res_abs - th.res_abs) / th.res_abs)), 0, 0.0

        if r.f_corr < th.f_corr or r.v_corr < th.v_corr:
            w = slice(r.t0, r.t0 + dc.window_s)
            lag, lc = pulse.find_replay_lag(ep.meter_f[w, r.meter], ep.tx_f, r.t0,
                                            dc.replay_min_lag_s, dc.max_replay_lag_s)
            if lc > dc.replay_corr:
                return REPLAY, round(min(0.99, lc), 3), lag, lc
            strength = max(r.f_corr, r.v_corr, 0.0)
            return FABRICATED, min(0.99, _sig((th.f_corr - strength) / 0.08)), 0, lc

        if r.res_abs > th.res_abs:
            if r.ri_corr < -0.5 and r.res_mean < 0:
                return EQUIPMENT_FAULT, min(0.99, _sig((r.res_abs - th.res_abs) / th.res_abs)
                                            * abs(r.ri_corr)), 0, 0.0
            return DRIFT, min(0.99, _sig((r.res_abs - th.res_abs) / (0.5 * th.res_abs))), 0, 0.0

        if self._bias_suspect(r):
            return BIAS, min(0.99, r.bias_sus * _sig((r.gap_abs - th.gap_abs) / th.gap_abs)), 0, 0.0

        if grid_event:
            return GRID_EVENT, min(0.99, 0.5 + 0.5 * max(r.f_corr, 0)), 0, 0.0

        if r.if_score > th.if_score:
            return UNEXPLAINED, min(0.95, _sig((r.if_score - th.if_score) / 0.02)), 0, 0.0

        return NORMAL, min(0.99, _sig((th.if_score - r.if_score) / 0.02)), 0, 0.0

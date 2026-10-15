"""Data-quality anomalies injected on top of clean measurements.

Each injector edits the *reported* channels of one meter between t0 and t1.
The physical grid is untouched, which is what separates these from the
genuine events and equipment faults created inside the simulator.

  STUCK       sensor freezes: all channels repeat the value at t0
  DRIFT       voltage sensor offset creeps up linearly (degraded sensor)
  BIAS        current/power scaled down: miscalibrated CT or meter tampering
  FABRICATED  plausible synthetic values, generated without access to the grid
  REPLAY      the meter's own genuine data from L seconds earlier is re-sent
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .config import STUCK, DRIFT, BIAS, FABRICATED, REPLAY
from .simulator import Episode, _ou


@dataclass
class Injection:
    episode: str
    meter: int
    kind: str
    t0: int
    t1: int
    param: float = 0.0     # drift V at end / bias factor / replay lag

    def as_dict(self):
        return asdict(self)


def _stuck(ep: Episode, j, t0, t1, rng):
    for ch in (ep.meter_v, ep.meter_i, ep.meter_p, ep.meter_f):
        ch[t0:t1, j] = ch[t0, j]
    return 0.0


def _drift(ep: Episode, j, t0, t1, rng):
    end_v = rng.uniform(3.0, 8.0)
    ramp = np.linspace(0, end_v, t1 - t0)
    old = ep.meter_v[t0:t1, j].copy()
    ep.meter_v[t0:t1, j] = old + ramp
    ep.meter_p[t0:t1, j] *= ep.meter_v[t0:t1, j] / old
    return end_v


def _bias(ep: Episode, j, t0, t1, rng):
    k = rng.uniform(0.4, 0.7)
    ep.meter_i[t0:t1, j] *= k
    ep.meter_p[t0:t1, j] *= k
    return k


def _fabricated(ep: Episode, j, t0, t1, rng):
    """What a tamperer can produce without being on the feeder: values with the
    right level, noise and smoothness, and even a believable 50 Hz wobble."""
    n = t1 - t0
    v_level = ep.meter_v[max(0, t0 - 300):t0, j].mean()
    p_level = ep.meter_p[max(0, t0 - 300):t0, j].mean()
    v = _ou(rng, n, v_level, 0.01, 0.15)
    p = np.clip(_ou(rng, n, p_level, 0.01, 0.03), -5, 8)
    ep.meter_v[t0:t1, j] = v
    ep.meter_p[t0:t1, j] = p
    ep.meter_i[t0:t1, j] = p * 1000 / v
    ep.meter_f[t0:t1, j] = _ou(rng, n, 50.0, 0.02, 0.004)
    return 0.0


def _replay(ep: Episode, j, t0, t1, rng):
    lag = int(rng.integers(600, min(1800, t0) + 1))
    for ch in (ep.meter_v, ep.meter_i, ep.meter_p, ep.meter_f):
        ch[t0:t1, j] = ch[t0 - lag:t1 - lag, j].copy()
    return float(lag)


INJECTORS = {STUCK: _stuck, DRIFT: _drift, BIAS: _bias,
             FABRICATED: _fabricated, REPLAY: _replay}


def inject(ep: Episode, j: int, kind: str, t0: int, t1: int, rng) -> Injection:
    param = INJECTORS[kind](ep, j, t0, t1, rng)
    return Injection(ep.spec.name, j, kind, t0, t1, float(param))


def random_schedule(ep: Episode, rng, per_kind=2, start_after=900,
                    dur_range=(240, 600), busy=()):
    """Place `per_kind` injections of each type on random meters without
    overlapping each other or meters already busy with an equipment fault."""
    T, N = ep.meter_v.shape
    taken = {j: [(a, b)] for j, a, b in busy}
    out = []
    kinds = [k for k in INJECTORS for _ in range(per_kind)]
    rng.shuffle(kinds)
    for kind in kinds:
        for _ in range(200):
            j = int(rng.integers(0, N))
            dur = int(rng.integers(*dur_range))
            t0 = int(rng.integers(start_after, T - dur))
            t1 = t0 + dur
            if all(t1 + 60 < a or t0 > b + 60 for a, b in taken.get(j, [])):
                taken.setdefault(j, []).append((t0, t1))
                out.append(inject(ep, j, kind, t0, t1, rng))
                break
    return out

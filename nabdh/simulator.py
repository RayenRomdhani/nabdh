"""1 Hz simulator for one LV feeder: grid frequency, MV voltage, household loads,
rooftop PV and a radial voltage-drop model.

The point of the simulator is to produce the two "pulses" NABDH relies on:
  * grid pulse  - second-by-second frequency wobble, identical for every device
                  on the synchronous grid;
  * feeder pulse - voltage wobble from the MV side plus shared cable drops,
                  identical (after physics) for every meter on this transformer.
Genuine grid events and genuine equipment faults are created here, in the
physics. Data-quality problems are injected later on the *measurements*
(see injectors.py), which is exactly the distinction the challenge asks for.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import DEFAULT, Config

PROFILES = {
    # base kW, AC/heater kW, mean on s, mean off s, PV irradiance start->end
    "summer_evening": dict(base=0.7, cyc_kw=(1.2, 2.0), on=420, off=360, pv=(0.30, 0.05)),
    "summer_heatwave": dict(base=0.9, cyc_kw=(1.5, 2.4), on=700, off=200, pv=(0.45, 0.20)),
    "ramadan_iftar": dict(base=0.5, cyc_kw=(0.8, 1.4), on=300, off=600, pv=(0.15, 0.00)),
    "winter_morning": dict(base=0.6, cyc_kw=(1.0, 2.0), on=500, off=400, pv=(0.25, 0.50)),
}


@dataclass
class EpisodeSpec:
    name: str
    kind: str
    duration_s: int = 3600
    seed: int = 0
    # (type, t_start, duration) with type in {"freq_dip", "mv_sag"}
    grid_events: list = field(default_factory=list)
    # (meter, t_start, duration, fault_ohm) - loose/high-resistance connection
    equipment_faults: list = field(default_factory=list)


@dataclass
class Episode:
    spec: EpisodeSpec
    t: np.ndarray            # (T,)
    meter_v: np.ndarray      # (T, N) measured
    meter_i: np.ndarray
    meter_p: np.ndarray
    meter_f: np.ndarray
    tx_v: np.ndarray         # (T,) transformer LV-side meter (trusted reference)
    tx_i: np.ndarray
    tx_p: np.ndarray
    tx_f: np.ndarray
    true_v: np.ndarray       # (T, N) physical truth, kept for analysis only
    true_p: np.ndarray


def _ou(rng, n, mu, theta, sigma, x0=None):
    x = np.empty(n)
    x[0] = mu if x0 is None else x0
    eps = rng.standard_normal(n)
    for k in range(1, n):
        x[k] = x[k - 1] + theta * (mu - x[k - 1]) + sigma * eps[k]
    return x


def _cycling(rng, n, kw, mean_on, mean_off):
    """On/off appliance (AC compressor, heater) with exponential dwell times."""
    out = np.zeros(n)
    t, on = 0, rng.random() < mean_on / (mean_on + mean_off)
    while t < n:
        dwell = int(rng.exponential(mean_on if on else mean_off)) + 30
        if on:
            out[t:t + dwell] = kw
        t += dwell
        on = not on
    return out


def _bursts(rng, n, rate_per_h, kw_range, dur_range):
    """Short appliances: kettle, oven, washing machine heater."""
    out = np.zeros(n)
    for _ in range(rng.poisson(rate_per_h * n / 3600)):
        t0 = rng.integers(0, n)
        out[t0:t0 + rng.integers(*dur_range)] += rng.uniform(*kw_range)
    return out


def household_loads(spec: EpisodeSpec, cfg: Config, rng) -> np.ndarray:
    """Net active power per meter in kW, shape (T, N). PV makes it negative."""
    T, N = spec.duration_s, cfg.feeder.n_meters
    prof = PROFILES[spec.kind]
    P = np.zeros((T, N))
    for j in range(N):
        scale = rng.uniform(0.6, 1.5)
        P[:, j] = prof["base"] * scale + _ou(rng, T, 0.0, 0.05, 0.01)
        if rng.random() < 0.8:
            P[:, j] += _cycling(rng, T, rng.uniform(*prof["cyc_kw"]), prof["on"], prof["off"])
        P[:, j] += _cycling(rng, T, 0.15, 900, 1200)                 # fridge
        P[:, j] += _bursts(rng, T, 0.6, (1.8, 2.4), (60, 300))       # kettle / oven

    if spec.kind == "ramadan_iftar":
        # Synchronised cooking surge before sunset (mid-episode), quiet meal,
        # then lights/TV. A naive detector flags this; NABDH must not.
        sunset = T // 2
        for j in range(N):
            start = sunset - rng.integers(300, 1500)
            P[start:sunset - 60, j] += rng.uniform(1.2, 2.2)
            P[sunset:sunset + 900, j] -= 0.25 * P[sunset:sunset + 900, j]
            P[sunset + 900:, j] += rng.uniform(0.2, 0.5)

    # Rooftop PV with a cloud factor shared by the neighbourhood.
    irr = np.linspace(*prof["pv"], T)
    cloud = np.clip(_ou(rng, T, 0.9, 0.01, 0.01), 0.4, 1.0)
    pv_houses = rng.choice(N, cfg.feeder.n_pv, replace=False)
    for j in pv_houses:
        P[:, j] -= cfg.feeder.pv_kw * irr * cloud * rng.uniform(0.85, 1.0)
    return P


def solve_feeder(v0: np.ndarray, p_kw: np.ndarray, r_service: np.ndarray, cfg: Config):
    """Radial feeder voltage drop. House j sits at node j+1.

    v_node[k] = v0 - sum_{m<=k} R_seg * I_down[m],  I_down[m] = sum_{j>=m} I_j
    v_meter[j] = v_node[j] - R_service[j] * I_j
    Two fixed-point iterations are enough at these drop levels.
    """
    fd = cfg.feeder
    v = np.repeat(v0[:, None], p_kw.shape[1], axis=1)
    for _ in range(3):
        i = p_kw * 1000.0 / v
        i_down = np.cumsum(i[:, ::-1], axis=1)[:, ::-1]
        v_node = v0[:, None] - np.cumsum(fd.r_segment * i_down, axis=1)
        v = v_node - r_service * i
    return v, i, i_down


def simulate(spec: EpisodeSpec, cfg: Config = DEFAULT) -> Episode:
    rng = np.random.default_rng(spec.seed)
    T, N = spec.duration_s, cfg.feeder.n_meters
    fd, gd, nz = cfg.feeder, cfg.grid, cfg.noise

    f = _ou(rng, T, fd.f_nominal, gd.f_theta, gd.f_sigma)
    v_src = _ou(rng, T, fd.v_source, gd.v_theta, gd.v_sigma)
    for kind, t0, dur in spec.grid_events:
        k = np.arange(T - t0)
        if kind == "freq_dip":      # generator trip somewhere on the synchronous grid
            dip = -0.18 * np.minimum(k / 4, 1) * np.exp(-np.maximum(k - 4, 0) / (dur / 3))
            f[t0:] += dip
        elif kind == "mv_sag":      # upstream MV fault cleared after `dur` seconds
            v_src[t0:t0 + dur] *= 0.90

    p_true = household_loads(spec, cfg, rng)

    r_serv = np.full((T, N), fd.r_service)
    for j, t0, dur, ohm in spec.equipment_faults:
        sl = slice(t0, t0 + dur)
        # loose connection: higher resistance plus intermittent arcing
        r_serv[sl, j] = ohm * (1 + 0.6 * (rng.random(dur) < 0.05) * rng.random(dur))

    # transformer impedance drop, then the feeder
    i_tot_guess = p_true.sum(axis=1) * 1000 / fd.v_nominal
    v0 = v_src - fd.r_transformer * i_tot_guess
    v_true, i_true, i_down = solve_feeder(v0, p_true, r_serv, cfg)

    losses_w = (fd.r_segment * i_down ** 2).sum(axis=1) + (r_serv * i_true ** 2).sum(axis=1)
    tx_i_true = i_true.sum(axis=1)
    tx_p_true = (v_true * i_true).sum(axis=1) / 1000 + losses_w / 1000

    meas = lambda x, sd: x + sd * rng.standard_normal(x.shape)
    meter_v = meas(v_true, nz.v_sd)
    meter_i = i_true * (1 + nz.i_rel_sd * rng.standard_normal(i_true.shape))
    meter_p = meter_v * meter_i / 1000
    meter_f = meas(np.repeat(f[:, None], N, axis=1), nz.f_sd)

    return Episode(
        spec=spec, t=np.arange(T),
        meter_v=meter_v, meter_i=meter_i, meter_p=meter_p, meter_f=meter_f,
        tx_v=meas(v0, nz.v_sd),
        tx_i=tx_i_true * (1 + nz.i_rel_sd * rng.standard_normal(T)),
        tx_p=tx_p_true * (1 + nz.i_rel_sd * rng.standard_normal(T)),
        tx_f=meas(f, nz.f_sd),
        true_v=v_true, true_p=p_true,
    )

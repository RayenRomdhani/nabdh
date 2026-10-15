"""Build, save and load the NABDH benchmark dataset.

Layout on disk (data/):
  <episode>_meters.csv.gz       t, meter, v, i, p, f        (1 Hz, 24 meters)
  <episode>_transformer.csv.gz  t, v, i, p, f               (1 Hz)
  ground_truth.csv              every injected anomaly / genuine event
  episodes.json                 episode specs (seeds, types, splits)
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT, Config, GRID_EVENT, EQUIPMENT_FAULT, NORMAL
from .simulator import EpisodeSpec, Episode, simulate
from .injectors import random_schedule

# mtime=0 keeps the .gz bytes identical on every rebuild (clean git diffs)
GZIP = {"method": "gzip", "mtime": 0}

TRAIN_SPECS = [
    EpisodeSpec("train_summer_evening", "summer_evening", seed=101),
    EpisodeSpec("train_winter_morning", "winter_morning", seed=102),
]
TEST_KINDS = [("test_summer_evening", "summer_evening", 201),
              ("test_summer_heatwave", "summer_heatwave", 202),
              ("test_ramadan_iftar", "ramadan_iftar", 203),
              ("test_winter_morning", "winter_morning", 204)]


def _test_spec(name, kind, seed, cfg: Config):
    rng = np.random.default_rng(seed + 1000)
    T = 3600
    t_dip = int(rng.integers(900, 3300))
    t_sag = int(rng.integers(900, 3300))
    while abs(t_sag - t_dip) < 200:
        t_sag = int(rng.integers(900, 3300))
    fault_meter = int(rng.integers(0, cfg.feeder.n_meters))
    t_fault = int(rng.integers(900, T - 500))
    return EpisodeSpec(
        name, kind, T, seed,
        grid_events=[("freq_dip", t_dip, 90), ("mv_sag", t_sag, 20)],
        equipment_faults=[(fault_meter, t_fault, 420, 0.5)],
    )


def build(cfg: Config = DEFAULT):
    """Returns (episodes, ground_truth DataFrame, split dict)."""
    episodes, rows = [], []
    for spec in TRAIN_SPECS:
        episodes.append(simulate(spec, cfg))
    for name, kind, seed in TEST_KINDS:
        spec = _test_spec(name, kind, seed, cfg)
        ep = simulate(spec, cfg)
        rng = np.random.default_rng(seed + 2000)
        busy = [(j, t0, t0 + d) for j, t0, d, _ in spec.equipment_faults]
        for inj in random_schedule(ep, rng, per_kind=2, busy=busy):
            rows.append(inj.as_dict())
        for kind_, t0, dur in spec.grid_events:
            span = dur if kind_ == "mv_sag" else 30   # dip is beyond 80 mHz for ~30 s
            rows.append(dict(episode=name, meter=-1, kind=GRID_EVENT, t0=t0, t1=t0 + span,
                             param=0.0, detail=kind_))
        for j, t0, dur, ohm in spec.equipment_faults:
            rows.append(dict(episode=name, meter=j, kind=EQUIPMENT_FAULT, t0=t0, t1=t0 + dur,
                             param=ohm, detail="loose_connection"))
        episodes.append(ep)
    gt = pd.DataFrame(rows)
    gt["detail"] = gt.get("detail", pd.Series(dtype=str)).fillna("injected")
    split = {"train": [s.name for s in TRAIN_SPECS], "test": [n for n, _, _ in TEST_KINDS]}
    return episodes, gt, split


def save(episodes, gt, split, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for ep in episodes:
        T, N = ep.meter_v.shape
        m = pd.DataFrame({
            "t": np.repeat(ep.t, N), "meter": np.tile(np.arange(N), T),
            "v": ep.meter_v.ravel().round(3), "i": ep.meter_i.ravel().round(3),
            "p": ep.meter_p.ravel().round(4), "f": ep.meter_f.ravel().round(4),
        })
        m.to_csv(out / f"{ep.spec.name}_meters.csv.gz", index=False, compression=GZIP)
        tx = pd.DataFrame({"t": ep.t, "v": ep.tx_v.round(3), "i": ep.tx_i.round(3),
                           "p": ep.tx_p.round(4), "f": ep.tx_f.round(4)})
        tx.to_csv(out / f"{ep.spec.name}_transformer.csv.gz", index=False, compression=GZIP)
    gt.to_csv(out / "ground_truth.csv", index=False)
    meta = {"split": split, "episodes": [asdict(ep.spec) for ep in episodes]}
    (out / "episodes.json").write_text(json.dumps(meta, indent=2))


def load(data_dir: Path):
    """Load episodes back as Episode objects (truth arrays left empty)."""
    meta = json.loads((data_dir / "episodes.json").read_text())
    episodes = []
    for s in meta["episodes"]:
        spec = EpisodeSpec(**s)
        m = pd.read_csv(data_dir / f"{spec.name}_meters.csv.gz")
        tx = pd.read_csv(data_dir / f"{spec.name}_transformer.csv.gz")
        T, N = len(tx), m["meter"].nunique()
        grid = lambda c: m[c].to_numpy().reshape(T, N)
        episodes.append(Episode(
            spec, tx["t"].to_numpy(), grid("v"), grid("i"), grid("p"), grid("f"),
            tx["v"].to_numpy(), tx["i"].to_numpy(), tx["p"].to_numpy(), tx["f"].to_numpy(),
            np.empty(0), np.empty(0)))
    gt = pd.read_csv(data_dir / "ground_truth.csv")
    return episodes, gt, meta["split"]


def window_labels(ep_name: str, T: int, N: int, gt: pd.DataFrame, window: int, step: int):
    """Ground-truth label per (window, meter).

    A window takes a label when the anomaly covers at least half of it.
    Windows touched by an anomaly for less than that are marked None and left
    out of scoring (they are transitions, not clean examples of either side).
    """
    starts = np.arange(0, T - window + 1, step)
    labels = np.full((len(starts), N), NORMAL, dtype=object)
    rows = gt[gt.episode == ep_name]
    # grid events first: they apply to every meter, data problems override them
    for _, r in rows[rows.meter == -1].iterrows():
        ov = np.clip(np.minimum(starts + window, r.t1) - np.maximum(starts, r.t0), 0, None)
        labels[ov > 0, :] = r.kind
    for _, r in rows[rows.meter >= 0].iterrows():
        ov = np.clip(np.minimum(starts + window, r.t1) - np.maximum(starts, r.t0), 0, None)
        labels[ov >= window // 2, r.meter] = r.kind
        labels[(ov > 0) & (ov < window // 2), r.meter] = None
    return starts, labels

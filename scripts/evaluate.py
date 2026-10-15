"""Calibrate on clean episodes, run NABDH on the test episodes, score it,
compare with a value-only Isolation Forest baseline, and write results/.

    python scripts/evaluate.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nabdh import dataset, governance  # noqa: E402
from nabdh.audit import AuditLog, verify  # noqa: E402
from nabdh.config import (DEFAULT, ALL_LABELS, DATA_PROBLEMS, NORMAL, GRID_EVENT,  # noqa: E402
                          EQUIPMENT_FAULT, UNEXPLAINED)
from nabdh.detector import NabdhDetector  # noqa: E402
from nabdh.explain import explain  # noqa: E402

RES = ROOT / "results"
PROBLEMS = set(DATA_PROBLEMS) | {EQUIPMENT_FAULT}
FLAGGED = PROBLEMS | {UNEXPLAINED}


def value_features(ep, window, step):
    """What a classic detector sees: levels and spreads of the raw values."""
    rows = []
    T, N = ep.meter_v.shape
    for s in range(0, T - window + 1, step):
        w = slice(s, s + window)
        v, p, f = ep.meter_v[w], ep.meter_p[w], ep.meter_f[w]
        feats = np.column_stack([v.mean(0), v.std(0), p.mean(0), p.std(0),
                                 np.diff(v, axis=0).std(0), f.mean(0), f.std(0)])
        for j in range(N):
            rows.append([ep.spec.name, s, j, *feats[j]])
    return pd.DataFrame(rows, columns=["episode", "t0", "meter", "v_mean", "v_std", "p_mean",
                                       "p_std", "dv_std", "f_mean", "f_std"])


def binary_scores(label, flagged):
    pos = label.isin(PROBLEMS)
    tp = int((pos & flagged).sum()); fp = int((~pos & flagged).sum())
    fn = int((pos & ~flagged).sum()); tn = int((~pos & ~flagged).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
            "false_positive_rate": round(fp / (fp + tn), 5) if fp + tn else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def latency(det: pd.DataFrame, gt: pd.DataFrame, window: int):
    """Seconds from anomaly start to the end of the first window that flags it."""
    out = []
    for _, r in gt[gt.meter >= 0].iterrows():
        d = det[(det.episode == r.episode) & (det.meter == r.meter) &
                (det.t0 + window > r.t0) & (det.t0 < r.t1) & det.verdict.isin(FLAGGED)]
        out.append({"kind": r.kind, "latency_s": float(d.t0.min() + window - r.t0) if len(d) else None})
    return pd.DataFrame(out)


def main():
    cfg = DEFAULT
    dc = cfg.detector
    episodes, gt, split = dataset.load(ROOT / "data")
    train = [e for e in episodes if e.spec.name in split["train"]]
    test = [e for e in episodes if e.spec.name in split["test"]]

    det = NabdhDetector(cfg).fit(train)
    RES.mkdir(exist_ok=True)
    (RES / "thresholds.json").write_text(det.th.to_json())

    frames, ms = [], []
    for ep in test:
        out = det.detect(ep)
        ms.append(out.attrs["ms_per_window"])
        T, N = ep.meter_v.shape
        starts, lab = dataset.window_labels(ep.spec.name, T, N, gt, dc.window_s, dc.step_s)
        idx = {s: k for k, s in enumerate(starts)}
        out["label"] = [lab[idx[a], b] for a, b in zip(out.t0, out.meter)]
        frames.append(out)
    allw = pd.concat(frames, ignore_index=True)
    scored = allw[allw.label.notna()].copy()

    # --- NABDH scores
    nabdh_bin = binary_scores(scored.label, scored.verdict.isin(FLAGGED))
    per_class = {}
    for lab in ALL_LABELS:
        sub = scored[scored.label == lab]
        per_class[lab] = {"windows": int(len(sub)),
                          "exact_class_recall": round(float((sub.verdict == lab).mean()), 4),
                          "flagged_rate": round(float(sub.verdict.isin(FLAGGED).mean()), 4)}
    ge = scored[scored.label == GRID_EVENT]
    discrimination = {
        "grid_event_windows": int(len(ge)),
        "recognised_as_grid_event": round(float((ge.verdict == GRID_EVENT).mean()), 4),
        "mistaken_for_data_problem": round(float(ge.verdict.isin(FLAGGED).mean()), 4),
    }
    lat = latency(allw, gt, dc.window_s)
    lat_summary = lat.groupby("kind").latency_s.agg(["median", "max"]).round(1)
    lat_summary["injected"] = lat.groupby("kind").size()
    lat_summary["detected"] = lat.groupby("kind").latency_s.apply(lambda s: int(s.notna().sum()))

    # --- Baseline: Isolation Forest on raw value features only
    vtrain = pd.concat([value_features(e, dc.window_s, dc.step_s) for e in train])
    vtest = pd.concat([value_features(e, dc.window_s, dc.step_s) for e in test])
    cols = ["v_mean", "v_std", "p_mean", "p_std", "dv_std", "f_mean", "f_std"]
    forest = IsolationForest(n_estimators=200, random_state=0).fit(vtrain[cols])
    thr = np.quantile(-forest.score_samples(vtrain[cols]), 0.995)
    vtest["flag"] = -forest.score_samples(vtest[cols]) > thr
    base = scored[["episode", "t0", "meter", "label"]].merge(vtest[["episode", "t0", "meter", "flag"]])
    base_bin = binary_scores(base.label, base.flag)
    base_per_class = {lab: round(float(base[base.label == lab].flag.mean()), 4) for lab in ALL_LABELS}

    cm = pd.crosstab(scored.label, scored.verdict).reindex(index=list(ALL_LABELS), fill_value=0)
    cm.to_csv(RES / "confusion_matrix.csv")

    metrics = {
        "dataset": {"test_episodes": split["test"], "scored_windows": int(len(scored)),
                    "window_s": dc.window_s, "step_s": dc.step_s},
        "nabdh": {**nabdh_bin, "per_class": per_class, "event_vs_data": discrimination,
                  "processing_ms_per_window_all_24_meters": round(float(np.mean(ms)), 2)},
        "baseline_isolation_forest_values_only": {**base_bin, "flagged_rate_per_class": base_per_class},
        "detection_latency_s": lat_summary.reset_index().to_dict(orient="records"),
    }
    (RES / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # --- Governance + explanations + tamper-evident log, per incident
    log = AuditLog(RES / "decision_log.jsonl")
    inc = governance.incidents(allw[allw.verdict != NORMAL], dc.step_s)
    inc["route"] = [governance.route(v, c)[0] for v, c in zip(inc.verdict, inc.confidence)]
    queue = []
    for r in inc.sort_values(["episode", "start"]).itertuples(index=False):
        rt, action = governance.route(r.verdict, r.confidence)
        rec = log.append(episode=r.episode, start_s=int(r.start), end_s=int(r.end),
                         meter=int(r.meter), verdict=r.verdict,
                         confidence=round(float(r.confidence), 3), route=rt, action=action,
                         evidence={k: round(float(getattr(r, k)), 4) for k in
                                   ("f_corr", "v_corr", "frozen", "res_mean", "gap_mean",
                                    "replay_lag", "replay_corr")},
                         explanation=explain(r))
        if rt == governance.HUMAN:
            queue.append(rec)
    ok, bad = verify(RES / "decision_log.jsonl")
    canaries = governance.add_canaries(queue, rate=0.05, rng=np.random.default_rng(0))
    gov = {"incidents": int(len(inc)),
           "route_counts": inc.groupby(["verdict", "route"]).size().unstack(fill_value=0)
           .to_dict(orient="index"),
           "human_queue": len(queue), "canaries_planted": len(canaries),
           "audit_chain_valid": ok}
    (RES / "governance_summary.json").write_text(json.dumps(gov, indent=2, default=int))

    # one example explanation per verdict, for the README / proposal
    ex = ["# Example verdicts produced by NABDH\n"]
    for v in list(ALL_LABELS) + [UNEXPLAINED]:
        sub = allw[(allw.verdict == v)]
        if len(sub):
            r = sub.sort_values("confidence", ascending=False).iloc[len(sub) // 3]
            rt, action = governance.route(r.verdict, r.confidence)
            ex.append(f"## {v}  (confidence {r.confidence:.2f}, route {rt})\n\n"
                      f"{explain(r)}\n\n*Action:* {action}\n")
    (RES / "example_explanations.md").write_text("\n".join(ex), encoding="utf-8")

    allw.to_csv(RES / "window_verdicts.csv.gz", index=False, float_format="%.4f",
                compression={"method": "gzip", "mtime": 0})
    print(json.dumps({"nabdh": nabdh_bin, "baseline": base_bin,
                      "event_vs_data": discrimination}, indent=2))
    print(cm.to_string())
    print(lat_summary.to_string())
    print("incidents:", len(inc), "| audit chain valid:", ok, "| human queue:", len(queue),
          "| canaries:", len(canaries))
    print(inc.groupby(["verdict", "route"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()

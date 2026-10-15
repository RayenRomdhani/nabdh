"""Who decides? Escalation rules between the AI and the human operator.

Guiding principle:
    The AI may always *stop trusting* data on its own, because that is
    reversible and harms no one. Anything that acts on a person (billing,
    accusation, disconnection) or on equipment (crew dispatch, switching)
    needs a human.

Routes:
    AUTO   - the AI acts and logs it; the operator sees it in the daily digest
    ASSIST - the AI prepares the action, the operator confirms with one click
    HUMAN  - the operator must decide; the AI only quarantines the data
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (NORMAL, GRID_EVENT, STUCK, DRIFT, BIAS, FABRICATED, REPLAY,
                     EQUIPMENT_FAULT, UNEXPLAINED)

AUTO, ASSIST, HUMAN, NONE = "AUTO", "ASSIST", "HUMAN", "NONE"


@dataclass(frozen=True)
class Policy:
    action: str          # what happens to the data / the asset
    auto_above: float    # confidence needed to act alone (1.01 = never)
    assist_above: float  # confidence needed for one-click approval
    severity: str        # how the operator queue sorts it


POLICIES = {
    NORMAL: Policy("accept data", 0.0, 0.0, "info"),
    GRID_EVENT: Policy("accept data, alert grid operations", 0.80, 0.0, "high"),
    STUCK: Policy("quarantine readings, open maintenance ticket", 0.90, 0.6, "medium"),
    DRIFT: Policy("apply offset correction, schedule recalibration", 0.95, 0.7, "medium"),
    # 1.01 = never: these act on a person or an asset, a human always decides
    BIAS: Policy("quarantine billing data, request field inspection", 1.01, 1.01, "high"),
    FABRICATED: Policy("quarantine readings, open security incident", 1.01, 1.01, "critical"),
    REPLAY: Policy("quarantine readings, open security incident", 1.01, 1.01, "critical"),
    EQUIPMENT_FAULT: Policy("dispatch crew to customer connection", 1.01, 1.01, "critical"),
    UNEXPLAINED: Policy("quarantine readings, human review", 1.01, 1.01, "medium"),
}


def route(verdict: str, confidence: float) -> tuple[str, str]:
    """Return (route, action) for one verdict."""
    p = POLICIES[verdict]
    if verdict == NORMAL:
        return NONE, p.action
    if confidence >= p.auto_above:
        return AUTO, p.action
    if confidence >= p.assist_above:
        return ASSIST, p.action
    return HUMAN, p.action


def incidents(windows, step: int):
    """Merge consecutive flagged windows of one meter and verdict into incidents.

    The operator reviews incidents, not 60-second windows: one stuck sensor
    for ten minutes is one ticket, not twenty.
    A grid event is one feeder-wide incident however many meters saw it.
    A one-window gap inside an incident does not split it.
    """
    rows = []
    w = windows.copy()
    w["key"] = np.where(w.verdict == GRID_EVENT, -1, w.meter)
    w = w.sort_values(["episode", "key", "verdict", "t0"])
    for (ep, key, v), g in w.groupby(["episode", "key", "verdict"], sort=False):
        t = np.sort(g.t0.unique())
        breaks = np.nonzero(np.diff(t) > 2 * step)[0] + 1
        for seg in np.split(t, breaks):
            sub = g[g.t0.isin(seg)]
            best = sub.loc[sub.confidence.idxmax()]
            rows.append({**best.to_dict(), "start": int(seg.min()), "end": int(seg.max()) + 60,
                         "n_windows": len(seg), "meters_involved": int(sub.meter.nunique()),
                         "confidence": float(sub.confidence.max())})
    return pd.DataFrame(rows)


def add_canaries(queue, rate: float, rng: np.random.Generator):
    """Mix known-answer test cases into the human queue.

    Against over-trust: if the operator approves a planted fake as genuine,
    the dashboard shows it. Against under-trust: canaries that are genuine
    events measure how often the operator overrides a correct AI verdict.
    Canaries are tagged in the audit log so they never reach billing or crews.
    """
    n = int(np.ceil(rate * len(queue)))
    picks = rng.choice(len(queue), size=min(n, len(queue)), replace=False) if len(queue) else []
    canaries = []
    for k in picks:
        c = dict(queue[k])
        c["canary"] = True
        c["expected_operator_answer"] = "reject" if c["verdict"] != GRID_EVENT else "accept"
        canaries.append(c)
    return canaries


def trust_metrics(decisions):
    """decisions: list of dicts with ai_verdict, operator_answer, canary, expected.

    over_trust  = share of planted fakes the operator approved
    under_trust = share of correct AI verdicts the operator overrode
    """
    can = [d for d in decisions if d.get("canary")]
    fakes = [d for d in can if d["expected"] == "reject"]
    over = np.mean([d["operator_answer"] == "accept" for d in fakes]) if fakes else 0.0
    real = [d for d in decisions if not d.get("canary") and d.get("ai_correct")]
    under = np.mean([d["operator_answer"] == "override" for d in real]) if real else 0.0
    return {"over_trust": float(over), "under_trust": float(under)}

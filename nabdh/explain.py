"""Plain-language explanations for every verdict.

Each explanation names the evidence, gives the number, and says what a
person can check for themselves. No model internals, no jargon beyond
"pulse" which the operator interface defines once.
"""
from __future__ import annotations

from .config import (NORMAL, GRID_EVENT, STUCK, DRIFT, BIAS, FABRICATED, REPLAY,
                     EQUIPMENT_FAULT, UNEXPLAINED)


def _clock(t: int) -> str:
    return f"{t // 60:02d}:{t % 60:02d}"


def explain(r) -> str:
    m = f"Meter M{int(r.meter):02d}"
    when = f"window {_clock(int(r.t0))}-{_clock(int(r.t0) + 60)}"
    v = r.verdict
    if v == NORMAL:
        return (f"{m}, {when}: healthy. Its readings carry the grid pulse "
                f"(match {r.f_corr:.2f}) and the feeder pulse (match {r.v_corr:.2f}), "
                f"and its voltage agrees with the physics model within {r.res_abs:.2f} V.")
    if v == GRID_EVENT:
        what = (f"grid frequency moved {r.f_dev * 1000:.0f} mHz away from 50 Hz"
                if r.f_dev > 0.08 else f"the transformer voltage sagged {r.sag_pct:.1f}%")
        return (f"{m}, {when}: genuine grid event. The {what}, and this meter saw it "
                f"in step with the reference (pulse match {r.f_corr:.2f}). "
                f"The data is real: act on the event, not on the meter.")
    if v == STUCK:
        return (f"{m}, {when}: sensor frozen. Voltage and current did not change at all "
                f"for {r.frozen * 100:.0f}% of the window. Real mains voltage always "
                f"wobbles, so these values are a repeated last reading, not a measurement.")
    if v == DRIFT:
        return (f"{m}, {when}: calibration drift. The meter still carries the pulse "
                f"(match {r.f_corr:.2f}), so it is real and in place, but it reads "
                f"{r.res_mean:+.2f} V away from what Ohm's law and its neighbours predict, "
                f"and that offset does not follow its own current. The sensor moved, not the grid.")
    if v == BIAS:
        return (f"{m}, {when}: under-reporting. The transformer delivers "
                f"{r.gap_mean:.2f} kW that no meter accounts for. That gap opened at the "
                f"same second this meter's reading dropped by the same amount. A real load "
                f"switching off would have lowered the transformer reading too. Possible CT "
                f"miscalibration or tampering: needs a human decision.")
    if v == FABRICATED:
        return (f"{m}, {when}: readings not measured here. Values look plausible, but "
                f"their wobble does not match the grid pulse (match {r.f_corr:.2f}) or the "
                f"feeder pulse (match {r.v_corr:.2f}), and no earlier moment matches either "
                f"(best {r.replay_corr:.2f}). Someone or something is generating these numbers.")
    if v == REPLAY:
        return (f"{m}, {when}: replayed data. The wobble is genuine but belongs to "
                f"{int(r.replay_lag)} s ago (match {r.replay_corr:.2f} at that delay, "
                f"{r.f_corr:.2f} now). Old real readings are being re-sent as new ones.")
    if v == EQUIPMENT_FAULT:
        if r.ri_corr < -0.5:
            why = (f"and sinks further when the house draws more current "
                   f"(correlation {r.ri_corr:.2f})")
        else:
            why = (f"and its voltage flickers out of step with the feeder (feeder pulse "
                   f"match {r.v_corr:.2f}), as arcing does")
        return (f"{m}, {when}: likely bad connection at the customer side. Grid pulse is "
                f"intact (match {r.f_corr:.2f}), so the meter is real, but the voltage sits "
                f"{abs(r.res_mean):.2f} V below the physics model {why}. That is the signature "
                f"of a loose or burnt joint: a real safety issue, not a data issue.")
    if v == UNEXPLAINED:
        return (f"{m}, {when}: unusual pattern that matches no known signature "
                f"(isolation score {r.if_score:.2f}). Passed to a human with all evidence.")
    return f"{m}, {when}: {v}"

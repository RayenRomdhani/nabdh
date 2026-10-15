"""Figures for the README and the proposal (results/figures/*.png).

    python scripts/make_figures.py      (run evaluate.py first)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nabdh import dataset  # noqa: E402
from nabdh.config import ALL_LABELS  # noqa: E402

FIG = ROOT / "results" / "figures"
BLUE, ORANGE, VIOLET, AQUA = "#2a78d6", "#eb6834", "#4a3aa7", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlecolor": INK,
})


def save(fig, name):
    fig.savefig(FIG / name, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def pulse_overlay(episodes, gt):
    """Frequency changes of a meter against the reference, three situations."""
    E = {e.spec.name: e for e in episodes}
    cases = [("Genuine meter", None), ("Fabricated reading", "FABRICATED"), ("Replayed data", "REPLAY")]
    fig, axes = plt.subplots(3, 1, figsize=(8, 6.2), sharex=True)
    for ax, (title, kind) in zip(axes, cases):
        if kind is None:
            ep, j, t0 = E["test_summer_evening"], 10, 600
        else:
            r = gt[(gt.kind == kind)].iloc[0]
            ep, j, t0 = E[r.episode], int(r.meter), int(r.t0) + 60
        t = np.arange(60)
        ref = np.diff(ep.tx_f[t0:t0 + 61]) * 1000
        m = np.diff(ep.meter_f[t0:t0 + 61, j]) * 1000
        c = np.corrcoef(ref, m)[0, 1]
        ax.plot(t, ref, color=BLUE, lw=2, label="Grid pulse (reference)")
        ax.plot(t, m, color=ORANGE, lw=2, ls="--", label=f"Meter M{j:02d}")
        ax.axhline(0, color=GRID, lw=1, zorder=0)
        ax.set_title(f"{title}: pulse match {c:.2f}", loc="left")
        ax.set_ylabel("Δf per second (mHz)")
    axes[0].legend(frameon=False, loc="upper right", ncol=2, labelcolor=INK2)
    axes[-1].set_xlabel("seconds in the window")
    fig.tight_layout()
    save(fig, "fig1_pulse_overlay.png")


def cover_pulse(episodes, gt):
    """The whole idea in one picture, sized for the proposal cover."""
    E = {e.spec.name: e for e in episodes}
    r = gt[gt.kind == "FABRICATED"].iloc[0]
    cases = [(E["test_summer_evening"], 10, 600, "Genuine meter: follows the grid pulse"),
             (E[r.episode], int(r.meter), int(r.t0) + 60, "Fabricated data: no pulse")]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 2.7), sharey=True)
    for ax, (ep, j, t0, title) in zip(axes, cases):
        t = np.arange(40)
        ref = np.diff(ep.tx_f[t0:t0 + 41]) * 1000
        m = np.diff(ep.meter_f[t0:t0 + 41, j]) * 1000
        c = np.corrcoef(ref, m)[0, 1]
        ax.plot(t, ref, color=BLUE, lw=2, label="grid pulse")
        ax.plot(t, m, color=ORANGE, lw=2, ls="--", label="meter")
        ax.axhline(0, color=GRID, lw=1, zorder=0)
        ax.set_title(f"{title} (match {c:.2f})", loc="left", fontsize=9.5)
        ax.set_xlabel("seconds")
    axes[0].set_ylabel("frequency change\nper second (mHz)")
    axes[1].legend(frameon=False, loc="upper right", ncol=2, labelcolor=INK2, fontsize=8)
    fig.tight_layout()
    save(fig, "fig_cover_pulse.png")


def confusion(cm: pd.DataFrame):
    labels = list(ALL_LABELS)
    cols = [c for c in labels + ["UNEXPLAINED"] if c in cm.columns]
    cm = cm.reindex(index=labels, columns=cols, fill_value=0)
    norm = cm.div(cm.sum(axis=1), axis=0).fillna(0)
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#f1f6fd", "#9ec3ef", BLUE, "#123f78"])
    ax.imshow(norm.values, cmap=cmap, vmin=0, vmax=1)
    for i in range(norm.shape[0]):
        for k in range(norm.shape[1]):
            v = norm.values[i, k]
            if cm.values[i, k]:
                ax.text(k, i, f"{v:.2f}\n({cm.values[i, k]})", ha="center", va="center",
                        fontsize=7.5, color="white" if v > 0.55 else INK)
    short = [s.replace("EQUIPMENT_FAULT", "EQUIP_FAULT") for s in cols]
    ax.set_xticks(range(len(cols)), short, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), [s.replace("EQUIPMENT_FAULT", "EQUIP_FAULT") for s in labels], fontsize=8)
    ax.set_xlabel("NABDH verdict"); ax.set_ylabel("ground truth")
    ax.set_title("Verdicts per ground-truth class (row share, window count)", loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "fig2_confusion_matrix.png")


def baseline_bars(metrics):
    labels = [l for l in ALL_LABELS if l not in ("NORMAL",)]
    nab = [metrics["nabdh"]["per_class"][l]["flagged_rate"] for l in labels]
    base = [metrics["baseline_isolation_forest_values_only"]["flagged_rate_per_class"][l] for l in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(x - 0.2, base, 0.38, color=ORANGE, label="Isolation Forest on values only")
    ax.bar(x + 0.2, nab, 0.38, color=BLUE, label="NABDH (pulse + physics)")
    for xi, v in zip(x, nab):
        ax.text(xi + 0.2, v + 0.02, f"{v:.0%}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, [l.replace("EQUIPMENT_FAULT", "EQUIP_FAULT") for l in labels], fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("windows flagged as a problem")
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title("Flag rate by class. GRID_EVENT should stay low: it is real data", loc="left")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.13), ncol=2, labelcolor=INK2)
    save(fig, "fig3_baseline_comparison.png")


def timeline(verdicts: pd.DataFrame, gt: pd.DataFrame, episode="test_summer_evening"):
    groups = {"GRID_EVENT": 1, "STUCK": 2, "DRIFT": 2, "BIAS": 2, "FABRICATED": 2,
              "REPLAY": 2, "UNEXPLAINED": 2, "EQUIPMENT_FAULT": 3}
    v = verdicts[verdicts.episode == episode]
    t0s = np.sort(v.t0.unique())
    N = v.meter.max() + 1
    grid = np.zeros((N, len(t0s)))
    idx = {t: k for k, t in enumerate(t0s)}
    for r in v.itertuples():
        grid[r.meter, idx[r.t0]] = groups.get(r.verdict, 0)
    cmap = matplotlib.colors.ListedColormap(["#f1f0ec", BLUE, ORANGE, VIOLET])
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=3, interpolation="nearest",
              extent=[0, t0s[-1] / 60 + 1, N - 0.5, -0.5])
    for r in gt[(gt.episode == episode) & (gt.meter >= 0)].itertuples():
        ax.add_patch(Rectangle((r.t0 / 60, r.meter - 0.5), (r.t1 - r.t0) / 60, 1,
                               fill=False, ec=INK, lw=1.1))
        ax.text(r.t1 / 60 + 0.3, r.meter, r.kind.lower(), va="center", fontsize=6.5, color=INK2)
    ax.set_xlabel("minutes into the episode"); ax.set_ylabel("meter")
    ax.set_title("One hour on the feeder: NABDH verdicts (fill) vs injected truth (outline)", loc="left")
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE, VIOLET)] + \
              [Rectangle((0, 0), 1, 1, fill=False, ec=INK)]
    ax.legend(handles, ["genuine grid event", "data problem", "equipment fault", "ground truth"],
              frameon=False, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=4, labelcolor=INK2)
    save(fig, "fig4_timeline.png")


def rule_ladder():
    """The verdict ladder as yes/no questions, top to bottom."""
    steps = [
        ("1. Are voltage and current frozen?", "yes", "STUCK", ORANGE),
        ("2. Does the meter carry the grid & feeder pulse?", "no", "REPLAY if the pulse matches an\nearlier moment, else FABRICATED", ORANGE),
        ("3. Does its voltage match Ohm's-law prediction?", "no", "EQUIPMENT_FAULT if the error grows\nwith its own current, else DRIFT", ORANGE),
        ("4. Does the energy add up at the transformer?", "no", "BIAS (gap opened when this\nmeter's reading dropped)", ORANGE),
        ("5. Is the whole feeder moving together?", "yes", "GRID_EVENT (real data,\nalert operations)", BLUE),
        ("6. Does the Isolation Forest find it unusual?", "yes", "UNEXPLAINED (human review)", VIOLET),
    ]
    fig, ax = plt.subplots(figsize=(9, 7.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    top, gap, h = 92, 13.5, 8.5
    for k, (q, ans, verdict, col) in enumerate(steps):
        y = top - k * gap
        ax.add_patch(FancyBboxPatch((2, y - h / 2), 50, h, boxstyle="round,pad=0.3,rounding_size=1.5",
                                    fc="#f1f0ec", ec=INK2, lw=1))
        ax.text(27, y, q, ha="center", va="center", fontsize=9.2, color=INK)
        ax.add_patch(FancyBboxPatch((63, y - h / 2), 35, h, boxstyle="round,pad=0.3,rounding_size=1.5",
                                    fc=col + "22", ec=col, lw=1.2))
        ax.text(80.5, y, verdict, ha="center", va="center", fontsize=8.6, color=INK, linespacing=1.3)
        ax.annotate("", (62.4, y), (52.8, y), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4))
        ax.text(57.6, y + 1.4, ans, ha="center", fontsize=8.5, color=col, fontweight="bold")
        nxt = "yes" if ans == "no" else "no"
        ax.annotate("", (27, y - h / 2 - 4.6), (27, y - h / 2 - 0.4),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.1))
        ax.text(28.4, y - h / 2 - 2.6, nxt, fontsize=8.5, color=INK2)
    y = top - len(steps) * gap
    ax.add_patch(FancyBboxPatch((14, y - h / 2 + 1), 26, h - 2, boxstyle="round,pad=0.3,rounding_size=1.5",
                                fc=AQUA + "22", ec=AQUA, lw=1.2))
    ax.text(27, y + 0, "NORMAL: trust the data", ha="center", va="center", fontsize=9.2, color=INK)
    ax.text(2, 99, "Every minute, for every meter: the first answer that points to a problem decides",
            fontsize=10, color=INK, fontweight="bold")
    save(fig, "fig5_rule_ladder.png")


def operator_mockup(episodes):
    """Operator review screen, filled with a real incident from the decision log."""
    recs = [json.loads(l) for l in (ROOT / "results" / "decision_log.jsonl").read_text().splitlines()]
    ep_name = "test_summer_evening"
    queue = sorted([r for r in recs if r["episode"] == ep_name and r["route"] == "HUMAN"
                    and r["confidence"] > 0.5], key=lambda r: r["start_s"])[:6]
    sel = next(r for r in queue if r["verdict"] == "REPLAY")
    queue = [r for r in queue if r is sel or r["meter"] != sel["meter"]]
    ep = {e.spec.name: e for e in episodes}[ep_name]
    j, lag = sel["meter"], int(sel["evidence"]["replay_lag"])
    t0 = sel["start_s"] + 60

    fig = plt.figure(figsize=(11, 6.6))
    fig.patch.set_facecolor("#f4f3ef")
    bar = fig.add_axes([0, 0.93, 1, 0.07]); bar.axis("off")
    bar.add_patch(Rectangle((0, 0), 1, 1, color="#1d2b3a", transform=bar.transAxes))
    bar.text(0.015, 0.5, "NABDH operator console   |   Feeder F-014   |   human review queue",
             color="white", va="center", fontsize=10.5, fontweight="bold")
    bar.text(0.985, 0.5, "operator: on shift", color="#c3c2b7", va="center", ha="right", fontsize=9)

    q = fig.add_axes([0.012, 0.03, 0.27, 0.88]); q.axis("off")
    q.text(0, 0.985, "Queue (newest first)", fontsize=10, fontweight="bold", color=INK, va="top")
    sev = {"REPLAY": "#e34948", "FABRICATED": "#e34948", "EQUIPMENT_FAULT": "#e34948",
           "BIAS": "#eda100", "DRIFT": "#eda100"}
    for k, r in enumerate(reversed(queue)):
        y = 0.9 - k * 0.135
        on = r is sel
        q.add_patch(FancyBboxPatch((0, y - 0.105), 0.98, 0.11, boxstyle="round,pad=0.005,rounding_size=0.02",
                                   fc="white" if not on else "#e8f0fb", ec=BLUE if on else GRID,
                                   lw=1.4 if on else 0.8, transform=q.transAxes))
        q.add_patch(Rectangle((0.02, y - 0.09), 0.018, 0.08, color=sev.get(r["verdict"], INK2),
                              transform=q.transAxes))
        q.text(0.07, y - 0.025, f"{r['verdict'].replace('_', ' ')}  M{r['meter']:02d}", fontsize=9,
               fontweight="bold", color=INK, transform=q.transAxes)
        q.text(0.07, y - 0.07, f"confidence {r['confidence']:.2f}  |  {r['start_s'] // 60:02d}:{r['start_s'] % 60:02d}",
               fontsize=8, color=INK2, transform=q.transAxes)

    m = fig.add_axes([0.3, 0.03, 0.685, 0.88]); m.axis("off")
    m.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.0,rounding_size=0.015", fc="white",
                               ec=GRID, transform=m.transAxes))
    m.text(0.025, 0.955, f"REPLAY on meter M{j:02d}", fontsize=14, fontweight="bold", color="#a32d2d")
    m.text(0.025, 0.905, f"confidence {sel['confidence']:.2f}   |   route: HUMAN decision required   |   "
           f"data already quarantined", fontsize=9, color=INK2)
    expl = sel["explanation"].split(": ", 1)[1]
    expl = expl[0].upper() + expl[1:]
    import textwrap
    m.text(0.025, 0.86, "\n".join(textwrap.wrap(expl, 112)), fontsize=8.8, color=INK, va="top",
           linespacing=1.45)

    for k, (ref_t, title, col) in enumerate([(t0, "Pulse now vs reference now", ORANGE),
                                             (t0 - lag, f"Pulse now vs reference {lag} s ago", BLUE)]):
        ax = fig.add_axes([0.33 + k * 0.335, 0.33, 0.3, 0.33])
        ref = np.diff(ep.tx_f[ref_t:ref_t + 41]) * 1000
        mm = np.diff(ep.meter_f[t0:t0 + 41, j]) * 1000
        ax.plot(ref, color=col, lw=1.8, label="reference")
        ax.plot(mm, color=INK2, lw=1.5, ls="--", label=f"meter M{j:02d}")
        ax.set_title(f"{title}: match {np.corrcoef(ref, mm)[0, 1]:.2f}", fontsize=9, loc="left")
        ax.tick_params(labelsize=7); ax.set_xlabel("seconds", fontsize=7.5)
        ax.legend(frameon=False, fontsize=7, loc="upper right", ncol=2)
        ax.set_facecolor("white")

    ev = sel["evidence"]
    m.text(0.025, 0.23, f"Evidence:  grid pulse now {ev['f_corr']:.2f}   |   feeder pulse now {ev['v_corr']:.2f}"
           f"   |   best earlier match {ev['replay_corr']:.2f} at {lag} s", fontsize=8.6, color=INK)
    m.text(0.025, 0.19, f"Suggested action: {sel['action']}", fontsize=8.6, color=INK)
    buttons = [("Confirm: open security incident", "#a32d2d", "white"),
               ("Reject: data is genuine", "white", INK),
               ("Escalate", "white", INK), ("AI was wrong because...", "white", INK2)]
    x = 0.025
    for label, fc, tc in buttons:
        w = 0.02 + 0.0088 * len(label)
        m.add_patch(FancyBboxPatch((x, 0.05), w, 0.08, boxstyle="round,pad=0.004,rounding_size=0.015",
                                   fc=fc, ec="#a32d2d" if fc != "white" else INK2, lw=1, transform=m.transAxes))
        m.text(x + w / 2, 0.09, label, ha="center", va="center", fontsize=8.6, color=tc,
               fontweight="bold" if fc != "white" else "normal", transform=m.transAxes)
        x += w + 0.015
    save(fig, "fig6_operator_console.png")


def architecture():
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis("off")

    def box(x, y, w, h, title, sub, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=color + "1f", ec=color, lw=1.2))
        ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=9.5,
                fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 7.2, sub, ha="center", va="top", fontsize=7.6, color=INK2,
                linespacing=1.35)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.1))

    box(1, 14, 15, 22, "Raw 1 Hz data", "24 smart meters\nV, I, P, f\n+ transformer\nmeter", INK2)
    box(21, 25, 18, 15, "Pulse checks", "grid pulse (frequency)\nfeeder pulse (voltage)\nfrozen test, replay search", BLUE)
    box(21, 4, 18, 17, "Physics witnesses", "voltage twin (Ohm)\nenergy balance (Kirchhoff)\nKirchhoff imputation", AQUA)
    box(44, 14, 17, 22, "Verdict ladder", "8 verdicts\nconfidence score\nplain-language why\n+ Isolation Forest\nsafety net", VIOLET)
    box(66, 25, 15, 15, "Governance", "AUTO / ASSIST /\nHUMAN routing\ncanary checks", ORANGE)
    box(66, 4, 15, 17, "Audit log", "hash-chained\nJSONL record of\nevery decision", INK2)
    box(86, 14, 13, 22, "Noosphere", "share pulse +\nfingerprints\nkeep household\ndata local", BLUE)
    arrow(16.6, 27, 20.4, 32); arrow(16.6, 22, 20.4, 13)
    arrow(39.6, 32, 43.4, 28); arrow(39.6, 13, 43.4, 21)
    arrow(61.6, 28, 65.4, 32); arrow(61.6, 21, 65.4, 13)
    arrow(81.6, 32, 85.4, 28)
    ax.text(50, 43, "NABDH trust layer: between raw grid data and automated action",
            ha="center", fontsize=11, fontweight="bold", color=INK)
    save(fig, "fig0_architecture.png")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    episodes, gt, _ = dataset.load(ROOT / "data")
    metrics = json.loads((ROOT / "results" / "metrics.json").read_text())
    cm = pd.read_csv(ROOT / "results" / "confusion_matrix.csv", index_col=0)
    verdicts = pd.read_csv(ROOT / "results" / "window_verdicts.csv.gz")
    architecture()
    rule_ladder()
    operator_mockup(episodes)
    cover_pulse(episodes, gt)
    pulse_overlay(episodes, gt)
    confusion(cm)
    baseline_bars(metrics)
    timeline(verdicts, gt)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Best worst-case detection frontier over time / Elo.

Frontier:   F(t) = max_d  min_{g available by t}  Perf(d, g)
For each detector the attacker picks its worst available generator (the inner min);
the frontier tracks the single best-worst-case detector (the outer max) as the pool
of available generators grows. The curve is non-increasing by construction.

Three compact, single-column paper figures:
  frontier_time_unrestricted.{pdf,png}  A) x=release date, every detector available throughout
  frontier_time_valid.{pdf,png}         B) x=release date, only detectors released by that date
  frontier_elo.{pdf,png}                C) x=generator Elo, every detector (no detector chronology)

Data:  acc_matrix.csv (detector,generator,acc) + generator_meta.csv (dates/elo, EDITABLE).
Detector release years: authoritative DYEAR map below.
"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from datetime import datetime

B = "/shared/home/sarim.hashmi/usenix/benchmark"

# ---- palette (validated data-viz default, light surface) -------------------
INK, MUT, GRID = "#0b0b0b", "#52514e", "#e6e6e2"
CLOUD  = "#b8b8b3"          # individual detectors (context)
MAJOR  = "#e34948"          # major-release marker (e.g. GPT-family)
# per-detector colours for the frontier (validated data-viz slots 1-3: all-pairs safe).
# assigned in first-appearance order; <=3 detectors ever define a single frontier.
DETPAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

# Authoritative detector release years (matches aggregate_attacks.py; used for era-gating).
DYEAR = {"UFD": 2023, "FreqNet": 2024, "NPR": 2024, "FatFormer": 2024, "AEROBLADE": 2024,
         "C2P-CLIP": 2025, "D3": 2025, "FIRE": 2025, "DDA": 2025, "FerretNet": 2025,
         "WaRPAD": 2025, "AllPatchesMatter": 2026, "OmniAID": 2026, "PGC": 2026, "PROBE": 2026,
         "DEAR": 2026, "DGS-Net": 2026, "SICA": 2026, "IAPL": 2026, "ForensicConcept": 2026}
# Threshold-degenerate detectors (constant ~0.5): shown in the cloud as context but NEVER
# allowed to define the frontier -- "best worst-case accuracy" must be a real classifier.
DEGEN = {"FIRE", "WaRPAD", "AEROBLADE"}

plt.rcParams.update({
    "font.size": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "axes.edgecolor": MUT, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": MUT, "ytick.color": MUT,
    "font.family": "DejaVu Sans", "figure.dpi": 200,
})


def load():
    acc = {}
    dets, gens = set(), set()
    for r in csv.DictReader(open(f"{B}/acc_matrix.csv")):
        acc[(r["detector"], r["generator"])] = float(r["acc"])
        dets.add(r["detector"]); gens.add(r["generator"])
    meta = {}
    for r in csv.DictReader(l for l in open(f"{B}/generator_meta.csv") if not l.startswith("#")):
        meta[r["generator"]] = dict(date=datetime.strptime(r["release_date"], "%Y-%m-%d"),
                                    elo=float(r["elo"]), major=r["is_major"] == "1",
                                    disp=r["display"])
    return acc, sorted(dets), meta, DYEAR


def worstcase(acc, det, avail):
    """min over the available generators the detector actually has a score for."""
    vals = [acc[(det, g)] for g in avail if (det, g) in acc]
    return min(vals) if vals else None


def frontier_stages(acc, dets, order, det_ok):
    """order: generators sorted along the x-axis. det_ok(stage_idx)-> set of usable detectors.
    Returns per stage: cloud [(det, wc)], and (best_det, best_wc)."""
    out = []
    for k in range(len(order)):
        avail = order[: k + 1]
        usable = det_ok(k)
        cloud = [(d, worstcase(acc, d, avail)) for d in dets if d in usable]
        cloud = [(d, v) for d, v in cloud if v is not None]
        # frontier max ignores threshold-degenerate detectors (they still show in the cloud)
        front = [(d, v) for d, v in cloud if d not in DEGEN]
        if not front:                                    # no real detector existed yet
            out.append(dict(cloud=cloud, best_det=None, best_wc=None))
            continue
        best_det, best_wc = max(front, key=lambda t: t[1])
        out.append(dict(cloud=cloud, best_det=best_det, best_wc=best_wc))
    return out


def draw(order, xs, labels, stages, meta, xlabel, title, fname):
    fig, ax = plt.subplots(figsize=(3.4, 2.75))
    ax.set_axisbelow(True)
    ax.set_ylim(0.40, 1.0)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # chance line
    ax.axhline(0.5, ls=(0, (4, 3)), lw=0.8, color=MUT, zorder=1)
    ax.text(0.005, 0.5, "chance", transform=ax.get_yaxis_transform(),
            va="bottom", ha="left", fontsize=5.4, color=MUT)

    # cloud: every detector's worst-case at each stage (context)
    for x, st in zip(xs, stages):
        ax.scatter([x] * len(st["cloud"]), [v for _, v in st["cloud"]],
                   s=7, c=CLOUD, alpha=0.55, edgecolors="none", zorder=2)

    # frontier: step coloured by the detector that OWNS each segment; identity via legend
    # (no on-plot text -> no overlap). horizontal tread = owner's colour, vertical riser = neutral.
    fpts = [(x, st["best_wc"], st["best_det"]) for x, st in zip(xs, stages)
            if st["best_wc"] is not None]
    fx = [p[0] for p in fpts]; fy = [p[1] for p in fpts]; fd = [p[2] for p in fpts]
    cmap = {}
    for d in fd:                                          # colour per detector, first-appearance order
        if d not in cmap:
            cmap[d] = DETPAL[len(cmap) % len(DETPAL)]
    for i in range(len(fx) - 1):
        ax.plot([fx[i], fx[i + 1]], [fy[i], fy[i]], color=cmap[fd[i]], lw=2.2,
                solid_capstyle="round", zorder=3)
        ax.plot([fx[i + 1]] * 2, [fy[i], fy[i + 1]], color=MUT, lw=1.0, zorder=3)
    ax.scatter(fx, fy, s=24, c=[cmap[d] for d in fd], edgecolors="white",
               linewidths=0.7, zorder=4)

    # major releases (e.g. GPT-family): red marker at the axis floor
    for x, g in zip(xs, order):
        if meta[g]["major"]:
            ax.scatter([x], [0.40], marker="^", s=18, c=MAJOR,
                       clip_on=False, zorder=5)

    ax.set_ylabel("accuracy", fontsize=7)
    ax.set_title(title, fontsize=8, color=INK, pad=6)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=5.4)
    ax.margins(x=0.04)
    for lab, g in zip(ax.get_xticklabels(), order):
        lab.set_color(MAJOR if meta[g]["major"] else MUT)
        if meta[g]["major"]:
            lab.set_fontweight("bold")

    leg = [Line2D([0], [0], color=cmap[d], lw=2.2, marker="o", ms=4, mec="white",
                  label=f"{d} (frontier)") for d in cmap]
    leg.append(Line2D([0], [0], color=CLOUD, lw=0, marker="o", ms=4,
                      label="other detectors"))
    if any(meta[g]["major"] for g in order):
        leg.append(Line2D([0], [0], color=MAJOR, lw=0, marker="^", ms=5, label="major release"))
    ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.50),
              ncol=3, fontsize=5.6, frameon=False, handletextpad=0.4,
              columnspacing=1.0, borderaxespad=0.0)

    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(f"{B}/plots/{fname}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote plots/{fname}.pdf  frontier: "
          + " -> ".join(f"{st['best_det']}:{st['best_wc']:.2f}"
                        for st in stages if st["best_wc"] is not None))


def main():
    acc, dets, meta, dyear = load()
    ALL = lambda k: set(dets)                     # unrestricted

    # ---- A) time, unrestricted detectors (ordinal x by release order) ----
    order = sorted(meta, key=lambda g: meta[g]["date"])
    xs = list(range(len(order)))
    tlab = [f"{meta[g]['disp']}\n{meta[g]['date']:%b}'{meta[g]['date']:%y}" for g in order]
    stA = frontier_stages(acc, dets, order, ALL)
    draw(order, xs, tlab, stA, meta, "generator (release order →)",
         "Detection accuracy over time", "frontier_time_unrestricted")

    # ---- B) time, historically valid detectors ----
    dates = [meta[g]["date"] for g in order]
    def valid(k):
        return {d for d in dets if dyear.get(d, 9999) <= dates[k].year}
    stB = frontier_stages(acc, dets, order, valid)
    draw(order, xs, tlab, stB, meta, "generator (release order →)",
         "Detection accuracy over time (era-valid detectors)", "frontier_time_valid")

    # ---- C) Elo, all detectors (ordinal x by Elo) ----
    order_e = sorted(meta, key=lambda g: meta[g]["elo"])
    xs_e = list(range(len(order_e)))
    elab = [f"{meta[g]['disp']}\n{int(meta[g]['elo'])}" for g in order_e]
    stC = frontier_stages(acc, dets, order_e, ALL)
    draw(order_e, xs_e, elab, stC, meta, "generator (Elo →)",
         "Detection accuracy vs. generator Elo", "frontier_elo")


if __name__ == "__main__":
    main()

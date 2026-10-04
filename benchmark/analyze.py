#!/usr/bin/env python3
"""Arms-race analysis: era-matched detector-vs-generator performance.

Reads scores/<detector>.csv (generator,gen_year,label,path,score; higher=fake),
writes results.csv / results.md and two figures:
  plots/arms_race.{png,pdf}  — era-matched diagonal, detector points + cohort line
  plots/heatmap.{png,pdf}    — full detector x generator AUC matrix
"""
import csv, os
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

B = "benchmark"

DETECTOR_YEAR = {
    "UFD": 2023,
    "AEROBLADE": 2024, "FreqNet": 2024, "NPR": 2024, "FatFormer": 2024,
    "C2P-CLIP": 2025, "D3": 2025, "FIRE": 2025, "DDA": 2025,
    "FerretNet": 2025, "WaRPAD": 2025,
    "AllPatchesMatter": 2026, "OmniAID": 2026, "PGC": 2026, "PROBE": 2026,
    "DEAR": 2026, "DGS-Net": 2026, "SICA": 2026, "IAPL": 2026,
    "ForensicConcept": 2026,
}
GEN_ORDER = ["SD2.1", "SD3", "SD3.5", "FLUX-dev", "FLUX-LoRA", "Boreal-FLUX",
             "FLUX2-plain", "FLUX2-boreal", "HiDream-O1", "GPTimage2-photoreal"]
GEN_YEAR = {"SD2.1": 2022, "SD3": 2024, "SD3.5": 2024, "FLUX-dev": 2024, "FLUX-LoRA": 2024,
            "Boreal-FLUX": 2024, "FLUX2-plain": 2025, "FLUX2-boreal": 2025,
            "HiDream-O1": 2026, "GPTimage2-photoreal": 2026}
# arms race: at each year Y, defender = best detector released <= Y,
# attacker = best (hardest) generator released <= Y  ->  min_g max_d metric

# palette (validated, light mode)
COHORT_C = {2023: "#2a78d6", 2024: "#eb6834", 2025: "#1baf7a", 2026: "#eda100"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


FPR_LEVELS = [0.01, 0.05, 0.10, 0.20]


def fake_recall_at_fpr(y, s, fpr=0.01):
    real = np.sort(s[y == 0])
    thr = real[int(np.ceil((1 - fpr) * len(real))) - 1]
    return float((s[y == 1] > thr).mean())


def accuracy(y, s):
    """Accuracy at the detector's natural threshold: 0.5 for probability-style
    scores (range within [0,1]), otherwise 0 (logit-style)."""
    thr = 0.5 if (s.min() >= 0.0 and s.max() <= 1.0) else 0.0
    return float(((s > thr).astype(int) == y).mean())


def load_scores():
    rows = []
    for f in sorted(os.listdir(f"{B}/scores")):
        if not f.endswith(".csv"):
            continue
        det = f[:-4]
        if det not in DETECTOR_YEAR:
            continue
        df = pd.read_csv(f"{B}/scores/{f}")
        df = df.dropna(subset=["score"])
        df["detector"], df["det_year"] = det, DETECTOR_YEAR[det]
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def main():
    df = load_scores()
    gens = [g for g in GEN_ORDER if g in set(df.generator)]
    dets = sorted(set(df.detector), key=lambda d: (DETECTOR_YEAR[d], d))
    print(f"{len(dets)} detectors x {len(gens)} generators")

    res = []
    for det in dets:
        for gen in gens:
            sub = df[(df.detector == det) & (df.generator == gen)]
            if len(sub) < 50:
                continue
            y, s = sub.label.values, sub.score.values.astype(float)
            row = {
                "detector": det, "det_year": DETECTOR_YEAR[det],
                "generator": gen, "gen_year": GEN_YEAR[gen],
                "n": len(sub), "acc": accuracy(y, s), "auc": roc_auc_score(y, s),
            }
            for f in FPR_LEVELS:
                row[f"recall@{int(f*100)}%fpr"] = fake_recall_at_fpr(y, s, f)
            res.append(row)
    R = pd.DataFrame(res)
    R.to_csv(f"{B}/results.csv", index=False)
    R.round(3).to_markdown(f"{B}/results.md", index=False)

    os.makedirs(f"{B}/plots", exist_ok=True)
    plot_arms_race(R, gens)
    plot_heatmap(R, dets, gens)
    print("wrote results.csv, results.md, plots/")


def minimax(R, year, metric):
    """Best available detector vs hardest available generator at `year`:
    min over generators<=year of (max over detectors<=year of metric)."""
    sub = R[(R.det_year <= year) & (R.gen_year <= year)]
    if sub.empty:
        return None
    best_per_gen = sub.groupby("generator")[metric].max()          # defender responds
    g_star = best_per_gen.idxmin()                                 # attacker picks worst case
    d_star = sub[sub.generator == g_star].sort_values(metric).iloc[-1].detector
    return float(best_per_gen.min()), g_star, d_star


def plot_arms_race(R, gens_avail):
    """Paper-Fig-17-style: minimax recall over time at several FPR operating
    points, plus AUC. Each point = best detector <= year vs hardest generator <= year."""
    years = sorted(set(R.det_year))
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")

    annot = {}
    xs, ys = [], []
    for yr in years:
        m = minimax(R, yr, "acc")
        if m is None:
            continue
        v, g_star, d_star = m
        xs.append(yr); ys.append(v)
        annot[yr] = (g_star, d_star)
    ax.plot(xs, ys, color="#2a78d6", lw=2.2, marker="o", ms=6.5,
            markeredgecolor="white", markeredgewidth=1.2, zorder=3)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.2f}", (x, y), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8, color=INK2)
    ax.axhline(0.5, color=MUTED, lw=1, ls=":", zorder=1)
    ax.annotate("chance", (xs[0], 0.512), fontsize=7.5, color=MUTED)
    ax.set_ylim(0.28, 1.05)
    # annotate the minimax pair per year under the axis
    for yr in years:
        if yr in annot:
            g_star, d_star = annot[yr]
            ax.annotate(f"{d_star}\nvs {g_star}", (yr, 0.28),
                        xytext=(0, -34), textcoords="offset points",
                        ha="center", fontsize=6.5, color=MUTED,
                        annotation_clip=False)
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], fontsize=9)
    ax.set_ylabel("accuracy  ·  best detector vs hardest generator", fontsize=9, color=INK)
    ax.set_title("Arms race — best available detector vs. hardest available generator",
                 fontsize=11, color=INK, pad=12)
    ax.grid(axis="y", color=GRID, lw=0.7); ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(f"{B}/plots/arms_race.{ext}", bbox_inches="tight")
    plt.close(fig)

    # companion curve: minimax AUC
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=200)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    xs, ys, labels = [], [], []
    for yr in years:
        m = minimax(R, yr, "auc")
        if m:
            xs.append(yr); ys.append(m[0]); labels.append(f"{m[2]}\nvs {m[1]}")
    ax.plot(xs, ys, color="#2a78d6", lw=2, marker="o", ms=6,
            markeredgecolor="white", markeredgewidth=1.2, zorder=3)
    for x, y, t in zip(xs, ys, labels):
        ax.annotate(t, (x, y), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=6.5, color=INK2)
    ax.axhline(0.5, color=MUTED, lw=1, ls=":")
    ax.annotate("chance", (xs[0], 0.507), fontsize=7, color=MUTED)
    ax.set_xticks(xs); ax.set_xticklabels([str(y) for y in xs], fontsize=9)
    ax.set_ylabel("minimax AUC", fontsize=9, color=INK)
    ax.set_title("Arms race — minimax AUC over time", fontsize=11, color=INK, pad=10)
    ax.set_ylim(0.38, 1.05)
    ax.grid(axis="y", color=GRID, lw=0.7); ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(f"{B}/plots/arms_race_auc.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(R, dets, gens):
    M = np.full((len(dets), len(gens)), np.nan)
    for _, r in R.iterrows():
        M[dets.index(r.detector), gens.index(r.generator)] = r.auc
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
    cmap.set_bad("#f0efec")
    fig, ax = plt.subplots(figsize=(1.1 * len(gens) + 3.2, 0.42 * len(dets) + 2), dpi=200)
    fig.patch.set_facecolor("white")
    im = ax.imshow(M, cmap=cmap, vmin=0.4, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(gens)))
    ax.set_xticklabels([f"{g}\n({GEN_YEAR[g]})" for g in gens], fontsize=8)
    ax.set_yticks(range(len(dets)))
    ax.set_yticklabels([f"{d}  ·{DETECTOR_YEAR[d]}" for d in dets], fontsize=7.5)
    for i in range(len(dets)):
        for j in range(len(gens)):
            if not np.isnan(M[i, j]):
                dark = M[i, j] > 0.78
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=6.8, color="white" if dark else INK)
    # cohort separators
    prev = None
    for i, d in enumerate(dets):
        if prev is not None and DETECTOR_YEAR[d] != prev:
            ax.axhline(i - 0.5, color="white", lw=2.5)
        prev = DETECTOR_YEAR[d]
    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("AUC", fontsize=8, color=INK2)
    cb.outline.set_visible(False)
    ax.set_title("Detector x generator AUC (rows grouped by detector release year)",
                 fontsize=10.5, color=INK, pad=10)
    ax.tick_params(colors=MUTED, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(f"{B}/plots/heatmap.{ext}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

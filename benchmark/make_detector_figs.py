#!/usr/bin/env python
"""Per-detector figures matching the paper's two example styles, from REAL data only
(attacks_out/<Det>.csv: path,label,score_clean,pred_clean,score_adv,pred_adv).

  <Det>_distribution.pdf : histogram of clean scores over all 2000 images, coloured
                           TP/TN/FP/FN (counts in legend), threshold line. (like ufd_distribution)
  <Det>_pgd.pdf          : Before|After PGD, correct-before samples only, TP(red)/TN(blue). (like D3_pgd)

Classification uses the detector's own pred_* columns, so counts EXACTLY match tab:attack_results.
Prob-in-[0,1] detectors are plotted natively (threshold 0.5); non-[0,1] (threshold-degenerate:
AEROBLADE/FIRE/WaRPAD) are min-max normalised to [0,1] with the threshold line at their real cut.
"""
import csv, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = "/shared/home/sarim.hashmi/usenix/benchmark"
OUT = f"{B}/plots/detector_figs"

# match the example figures' scale (ufd_distribution 13.89x5.88in, D3_pgd 12x4.46in, large fonts)
plt.rcParams.update({"font.size": 15, "xtick.labelsize": 14, "ytick.labelsize": 14})

# 14 detectors that are NOT the 6 paper baselines (those already have figures with the OLD numbers).
DETS = ["AEROBLADE", "FIRE", "DDA", "FerretNet", "WaRPAD", "AllPatchesMatter", "OmniAID",
        "PGC", "PROBE", "DEAR", "DGS-Net", "SICA", "IAPL", "ForensicConcept"]

# colours matched to the example PDFs
D_TP, D_TN, D_FP, D_FN = "#E74C3C", "#5CB85C", "#BCE3BC", "#EBA83A"
P_TP, P_TN = "#F08080", "#4169E1"
# original vs PGD overlay: real=blue family, fake=red family
C_REAL_O, C_REAL_P = "#2471A3", "#7FB3D5"   # Real Original (dark blue), Real PGD (light blue)
C_FAKE_O, C_FAKE_P = "#C0392B", "#F1948A"   # Fake Original (dark red),  Fake PGD (light red)
BINS = np.linspace(0.0, 1.0, 51)


def load(det):
    rows = []
    for r in csv.DictReader(open(f"{B}/attacks_out/{det}.csv")):
        if r["score_clean"] in ("", None) or r["pred_clean"] in ("", None):
            continue
        rows.append((int(r["label"]), float(r["score_clean"]), int(r["pred_clean"]),
                     float(r["score_adv"]) if r["score_adv"] not in ("", None) else None,
                     int(r["pred_adv"]) if r["pred_adv"] not in ("", None) else None))
    return rows


def transform(rows):
    """Return (xf, thr, xlabel). Prob detectors -> identity/0.5; else min-max to [0,1]."""
    sc = [r[1] for r in rows] + [r[3] for r in rows if r[3] is not None]
    lo, hi = min(sc), max(sc)
    if lo >= -1e-6 and hi <= 1 + 1e-6:
        return (lambda v: v), 0.5, "Model Probability ( 0.5 = Fake, < 0.5 = Real)"
    # non-probability score: normalise, put threshold at the real decision boundary
    p0 = [r[1] for r in rows if r[2] == 0]
    p1 = [r[1] for r in rows if r[2] == 1]
    if p0 and p1:
        t = 0.5 * (max(p0) + min(p1))
    elif p1:
        t = min(p1) - 1e-6
    else:
        t = max(p0) + 1e-6
    span = (hi - lo) or 1.0
    xf = lambda v: (v - lo) / span
    return xf, xf(t), "Model Score (normalized; ≥ thr = Fake, < thr = Real)"


def distribution(det, rows, xf, thr, xlabel):
    lab = np.array([r[0] for r in rows]); pc = np.array([r[2] for r in rows])
    x = np.array([xf(r[1]) for r in rows])
    tp = x[(lab == 1) & (pc == 1)]; tn = x[(lab == 0) & (pc == 0)]
    fp = x[(lab == 0) & (pc == 1)]; fn = x[(lab == 1) & (pc == 0)]
    fig, ax = plt.subplots(figsize=(13.9, 5.9))
    for d, c, lb in [(tn, D_TN, f"TN ({len(tn)}): Real correctly detected as Real"),
                     (fn, D_FN, f"FN ({len(fn)}): Fake incorrectly detected as Real"),
                     (fp, D_FP, f"FP ({len(fp)}): Real incorrectly detected as Fake"),
                     (tp, D_TP, f"TP ({len(tp)}): Fake correctly detected as Fake")]:
        ax.hist(d, bins=BINS, color=c, alpha=0.75, edgecolor="white", linewidth=0.3, label=lb)
    ax.axvline(thr, color="black", ls="--", lw=2.5, label=f"Threshold ({thr:.2f})")
    # legend order TP, TN, FP, FN, thr (match example)
    h, l = ax.get_legend_handles_labels()
    order = [next(i for i, s in enumerate(l) if s.startswith(k)) for k in ("TP", "TN", "FP", "FN", "Thr")]
    ax.legend([h[i] for i in order], [l[i] for i in order], fontsize=13.5, loc="upper right")
    ax.set_title(f"{det}: Distribution of All Predictions", fontweight="bold", fontsize=18)
    ax.set_xlabel(xlabel, fontweight="bold", fontsize=16); ax.set_ylabel("Count", fontweight="bold", fontsize=16)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{det}_distribution.{ext}", bbox_inches="tight")
    plt.close(fig)


def pgd(det, rows, xf, thr, xlabel):
    corr = [r for r in rows if r[2] == r[0] and r[3] is not None]  # correct before, has adv
    fake = [r for r in corr if r[0] == 1]; real = [r for r in corr if r[0] == 0]
    fig, ax = plt.subplots(figsize=(12.0, 5.2))
    series = [
        (np.array([xf(r[1]) for r in real]), C_REAL_O, "Real (Original)"),
        (np.array([xf(r[3]) for r in real]), C_REAL_P, "Real (PGD)"),
        (np.array([xf(r[1]) for r in fake]), C_FAKE_O, "Fake (Original)"),
        (np.array([xf(r[3]) for r in fake]), C_FAKE_P, "Fake (PGD)"),
    ]
    for d, c, lb in series:
        ax.hist(d, bins=BINS, color=c, alpha=0.7, edgecolor="white", linewidth=0.3, label=lb)
    ax.axvline(thr, color="black", ls="--", lw=2.5, label=f"Threshold ({thr:.2f})")
    ax.set_title(f"{det}: Prediction Scores Before vs. After PGD", fontweight="bold", fontsize=17)
    ax.set_xlabel(xlabel, fontsize=15); ax.set_ylabel("Count", fontsize=15); ax.set_xlim(0, 1)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=13.5,
               frameon=True, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{det}_pgd.{ext}", bbox_inches="tight")
    plt.close(fig)


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    only = sys.argv[1:] or DETS
    for det in only:
        rows = load(det)
        xf, thr, xlabel = transform(rows)
        distribution(det, rows, xf, thr, xlabel)
        pgd(det, rows, xf, thr, xlabel)
        nc = sum(1 for r in rows if r[2] == r[0])
        print(f"{det:17} n={len(rows)} correct_before={nc}  thr={thr:.3f}  -> {det}_distribution.pdf, {det}_pgd.pdf")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Build a HUMAN-subset accuracy matrix from the EXISTING per-image scores (no new runs).

Images are named after their caption; an image is in the "human" subset if its caption
mentions a person. We reuse each detector's raw scores from scores*/ , recover the decision
threshold (prob -> 0.5, logit -> 0.0), VALIDATE it reproduces acc_matrix.csv on the full set,
then recompute per detector x generator accuracy on the human subset only.

Writes: acc_matrix_faces.csv (detector,generator,acc) and faces_counts.csv (generator,real,fake).
"""
import csv, os, glob

B = "/shared/home/sarim.hashmi/usenix/benchmark"
SCORE_DIRS = ["scores", "scores_new_gens", "scores_flux2", "scores_gpt2_full"]
DEGEN = {"FIRE", "WaRPAD", "AEROBLADE"}          # constant ~0.5; reported but not meaningful

HUMAN = {
    "man", "mans", "men", "mens", "woman", "womans", "women", "person", "persons",
    "people", "peoples", "boy", "boys", "girl", "girls", "lady", "ladies", "guy", "guys",
    "child", "children", "kid", "kids", "baby", "babies", "toddler", "toddlers",
    "gentleman", "gentlemen", "player", "players", "rider", "riders", "skier", "skiers",
    "surfer", "surfers", "snowboarder", "snowboarders", "skater", "skaters", "worker",
    "workers", "couple", "couples", "crowd", "crowds", "pedestrian", "pedestrians",
    "face", "faces", "portrait", "teenager", "teenagers", "human", "humans",
    "family", "families", "someone",
}


def is_human(path):
    name = os.path.splitext(os.path.basename(path))[0].lower()
    toks = name.split("_")
    return any(t in HUMAN for t in toks)


def load_detector(det):
    """rows: list of (generator, label:int, path, score:float)."""
    rows = []
    for d in SCORE_DIRS:
        p = f"{B}/{d}/{det}.csv"
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            if r["score"] in ("", None):
                continue
            rows.append((r["generator"], int(r["label"]), r["path"], float(r["score"])))
    return rows


def threshold(rows):
    s = [x[3] for x in rows]
    lo, hi = min(s), max(s)
    return 0.5 if (lo >= -1e-6 and hi <= 1 + 1e-6) else 0.0    # prob vs logit


def acc(rows, thr, gen, subset=None):
    sel = [r for r in rows if r[0] == gen and (subset is None or subset(r[2]))]
    if not sel:
        return None, 0, 0
    correct = sum(1 for _, lab, _, sc in sel if int(sc > thr) == lab)
    nreal = sum(1 for _, lab, _, _ in sel if lab == 0)
    nfake = len(sel) - nreal
    return correct / len(sel), nreal, nfake


def main():
    detectors = sorted(os.path.splitext(os.path.basename(p))[0]
                       for p in glob.glob(f"{B}/scores/*.csv"))
    # reference full-set matrix
    ref = {}
    gens = []
    for r in csv.DictReader(open(f"{B}/acc_matrix.csv")):
        ref[(r["detector"], r["generator"])] = float(r["acc"])
        if r["generator"] not in gens:
            gens.append(r["generator"])

    faces, counts = {}, {}
    worst_cal = 0.0
    for det in detectors:
        rows = load_detector(det)
        thr = threshold(rows)
        for g in gens:
            # calibration: full-set accuracy vs acc_matrix (skip degenerate)
            full, _, _ = acc(rows, thr, g)
            if det not in DEGEN and full is not None and (det, g) in ref:
                worst_cal = max(worst_cal, abs(full - ref[(det, g)]))
            fa, nr, nf = acc(rows, thr, g, subset=is_human)
            faces[(det, g)] = fa
            counts[g] = (nr, nf)

    print(f"calibration: worst |full-set acc - acc_matrix| over non-degenerate cells = {worst_cal:.4f}")
    print("(should be ~0 if the recovered thresholds match how acc_matrix was built)\n")

    with open(f"{B}/acc_matrix_faces.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["detector", "generator", "acc"])
        for det in detectors:
            for g in gens:
                if faces[(det, g)] is not None:
                    w.writerow([det, g, round(faces[(det, g)], 4)])
    with open(f"{B}/faces_counts.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["generator", "human_real", "human_fake"])
        for g in gens:
            w.writerow([g, counts[g][0], counts[g][1]])

    print("human-subset image counts per generator (real / fake):")
    for g in gens:
        print(f"  {g:22s} {counts[g][0]:3d} / {counts[g][1]:3d}")
    print("\nwrote acc_matrix_faces.csv, faces_counts.csv")


if __name__ == "__main__":
    main()

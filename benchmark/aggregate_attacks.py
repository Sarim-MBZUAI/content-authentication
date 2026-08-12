#!/usr/bin/env python3
"""Aggregate attacks_out/<Detector>.csv into the before/after PGD table.
Protocol (matches the paper's Table 1):
  - 1000 fake + 1000 real.
  - Before: correct_fake = #(label1 & pred_clean==1); correct_real = #(label0 & pred_clean==0).
  - After:  correct_* using pred_adv.
  - Acc = (correct_fake+correct_real)/2000 * 100.
  - Attack success on fake = (before_correct_fake - after_correct_fake)/before_correct_fake*100
    (same for real). Undefined (—) if nothing was correct before.
Writes results_attack.json, results_attack.md, and fills tab_attack_results values.
"""
import csv, os, json

AT = "attacks_out"
DYEAR = {"UFD":2023,"FreqNet":2024,"NPR":2024,"FatFormer":2024,"AEROBLADE":2024,"C2P-CLIP":2025,"D3":2025,"FIRE":2025,"DDA":2025,"FerretNet":2025,"WaRPAD":2025,"AllPatchesMatter":2026,"OmniAID":2026,"PGC":2026,"PROBE":2026,"DEAR":2026,"DGS-Net":2026,"SICA":2026,"IAPL":2026,"ForensicConcept":2026}
ORDER = ["UFD","FreqNet","NPR","FatFormer","AEROBLADE","C2P-CLIP","D3","FIRE","DDA","FerretNet","WaRPAD","AllPatchesMatter","OmniAID","PGC","PROBE","DEAR","DGS-Net","SICA","IAPL","ForensicConcept"]
DEGEN = {"FIRE","WaRPAD","AEROBLADE"}  # threshold-degenerate; note in output


def load(det):
    p = f"{AT}/{det}.csv"
    if not os.path.exists(p):
        return None
    rows = [r for r in csv.DictReader(open(p)) if r["pred_clean"] not in ("", None)]
    return rows or None


def stats(rows):
    cf0 = sum(1 for r in rows if r["label"] == "1" and r["pred_clean"] == "1")
    cr0 = sum(1 for r in rows if r["label"] == "0" and r["pred_clean"] == "0")
    cf1 = sum(1 for r in rows if r["label"] == "1" and r["pred_adv"] == "1")
    cr1 = sum(1 for r in rows if r["label"] == "0" and r["pred_adv"] == "0")
    n = len(rows)
    acc0 = 100.0 * (cf0 + cr0) / n
    acc1 = 100.0 * (cf1 + cr1) / n
    asf = 100.0 * (cf0 - cf1) / cf0 if cf0 else None
    asr = 100.0 * (cr0 - cr1) / cr0 if cr0 else None
    return dict(n=n, cf0=cf0, cr0=cr0, acc0=round(acc0, 2), cf1=cf1, cr1=cr1,
               acc1=round(acc1, 2),
               asf=round(asf, 1) if asf is not None else None,
               asr=round(asr, 1) if asr is not None else None)


def main():
    out = {}
    md = ["# PGD attack results (ε = 8/255, 10 steps, α = 2/255) — New_benchmark (1000 real + 1000 fake, 512px)",
          "",
          "Before/after white-box ℓ∞ PGD. `†` = threshold-degenerate detector (FIRE/WaRPAD/AEROBLADE); read with care.",
          "",
          "| Year | Model | Correct fake | Correct real | Acc% | Correct fake (adv) | Correct real (adv) | Acc% (adv) | ASR fake | ASR real |",
          "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for det in ORDER:
        rows = load(det)
        tag = "†" if det in DEGEN else ""
        if not rows:
            md.append(f"| {DYEAR[det]} | {det}{tag} | - | - | - | - | - | - | - | - |")
            out[det] = None
            continue
        s = stats(rows); out[det] = s
        asf = "-" if s["asf"] is None else f"{s['asf']:.1f}"
        asr = "-" if s["asr"] is None else f"{s['asr']:.1f}"
        md.append(f"| {DYEAR[det]} | {det}{tag} | {s['cf0']} | {s['cr0']} | {s['acc0']:.2f} | "
                  f"{s['cf1']} | {s['cr1']} | {s['acc1']:.2f} | {asf} | {asr} |")
    json.dump(out, open("results_attack.json", "w"), indent=1)
    open("results_attack.md", "w").write("\n".join(md) + "\n")
    # LaTeX rows for the 14 new detectors (paste under the existing 6)
    latex = []
    for det in ORDER:
        if det in {"UFD","FreqNet","NPR","FatFormer","D3","C2P-CLIP"}:
            continue
        s = out[det]
        key = det.lower().replace("-", "")
        if not s:
            latex.append(f"        {det}~\\cite{{{key}}} & - & - & - & - & - & - & - & - \\\\")
        else:
            asf = "-" if s["asf"] is None else f"{s['asf']:.1f}"
            asr = "-" if s["asr"] is None else f"{s['asr']:.1f}"
            latex.append(f"        {det}~\\cite{{{key}}} & {s['cf0']} & {s['cr0']} & {s['acc0']:.2f} "
                         f"& {s['cf1']} & {s['cr1']} & {s['acc1']:.2f} & {asf} & {asr} \\\\")
    open("tab_attack_newrows.tex", "w").write("\n".join(latex) + "\n")
    print("wrote results_attack.{json,md} + tab_attack_newrows.tex")
    print("\n".join(md))


if __name__ == "__main__":
    main()

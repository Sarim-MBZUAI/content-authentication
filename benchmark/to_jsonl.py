#!/usr/bin/env python3
"""Convert scores/<detector>.csv -> scores_jsonl/<detector>.jsonl for auditing.
One JSON object per image with the raw adapter output and its meaning."""
import csv, json, os

B = "/shared/home/sarim.hashmi/usenix/benchmark"
os.makedirs(f"{B}/scores_jsonl", exist_ok=True)

# what the raw `score` field actually is, per adapter (from adapter implementations)
SCORE_TYPE = {
    "UFD": "sigmoid P(fake), CLIP-L/14 linear probe",
    "FreqNet": "sigmoid P(fake)",
    "NPR": "sigmoid P(fake)",
    "FatFormer": "softmax[:,1] P(fake)",
    "C2P-CLIP": "sigmoid P(fake)",
    "D3": "sigmoid P(fake)",
    "FIRE": "RAW LOGIT (higher=fake; sigmoid saturates on this data)",
    "DDA": "sigmoid P(fake)",
    "FerretNet": "sigmoid P(fake)",
    "WaRPAD": "NEGATED DINOv2 cosine similarity (training-free; higher=fake)",
    "AllPatchesMatter": "softmax[:,1] P(fake), GenImage CLIP-LoRA variant",
    "OmniAID": "softmax P(fake), dino_v2 hybrid MoE",
    "PGC": "sigmoid P(fake), progan+sdv1.4 ckpt",
    "PROBE": "sigmoid of mean patch logit, DINOv2 variant",
    "DEAR": "sigmoid P(fake), dear_c variant",
    "DGS-Net": "sigmoid P(fake), step2 ckpt",
    "SICA": "sigmoid P(fake), full ckpt",
    "IAPL": "sigmoid P(fake), non-adaptive forward (no TTA), progan ckpt",
    "ForensicConcept": "sigmoid of main-head logit, CLIP variant (DINOv3 gated)",
}

done = 0
for f in sorted(os.listdir(f"{B}/scores")):
    if not f.endswith(".csv"):
        continue
    det = f[:-4]
    with open(f"{B}/scores/{f}") as src, open(f"{B}/scores_jsonl/{det}.jsonl", "w") as dst:
        for r in csv.DictReader(src):
            dst.write(json.dumps({
                "detector": det,
                "score_type": SCORE_TYPE.get(det, "unknown"),
                "image": os.path.basename(r["path"]),
                "path": r["path"],
                "generator": r["generator"],
                "gen_year": int(r["gen_year"]),
                "label": int(r["label"]),           # 0=real, 1=fake
                "score": float(r["score"]) if r["score"] not in ("", None) else None,
            }) + "\n")
    done += 1
    print(f"{det}.jsonl written")
print(f"{done} detectors exported to {B}/scores_jsonl/")

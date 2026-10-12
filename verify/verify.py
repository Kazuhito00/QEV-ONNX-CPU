# -*- coding: utf-8 -*-
r"""GATE 2: ONNX runtime vs the goldens (QEV's own torch fp32 CPU runtime), text and image cases.

    uv run verify/verify.py [--threads 6] [--models DIR] [--only a,b]

Per question: token ids (exact), the two hidden states, condition logits, the option logits, calibrated probabilities,
the decision (choice key / noul > 0.5 / score argmax), and the abstain flag. DECISION flips and ABSTAIN flips are counted
separately (a probability within ~1e-4 of a threshold can flip the abstain flag without any real disagreement).
Also compares the vision tower's features with the oracle's for every golden picture.
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qev_decide import Decider, default_model_dir  # noqa: E402


def decision(ans):
    if ans["type"] == "choice":
        return ans["choice"]
    if ans["type"] == "noul":
        return ans["noul"] > 0.5
    p = ans["probabilities"]
    return max(p, key=p.get)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--models", default=None, help="model directory (default: $QEV_MODEL_DIR, else models/onnx)")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    gold = json.loads((ROOT / "artifacts" / "golden.json").read_text(encoding="utf-8"))["cases"]
    hid = np.load(ROOT / "artifacts" / "golden_hidden.npz")
    gimg = np.load(ROOT / "artifacts" / "golden_image.npz")
    d = Decider(args.models or default_model_dir(), intra_op_threads=args.threads, load_vision=True)
    only = set(filter(None, args.only.split(",")))
    cases = [(k, v) for k, v in gold.items() if not only or k in only]
    # sort by graph length so only one session is loaded at a time without thrashing
    cases.sort(key=lambda kv: max(q["n_tokens"] for q in kv[1]["questions"].values()))
    n = id_bad = dec_flip = abs_flip = 0
    worst = dict(hidden=0.0, cond_logits=0.0, logits=0.0, prob=0.0, vision=0.0)
    t0 = time.time()
    for name, c in cases:
        req = json.loads(json.dumps(c["request"]))
        if req["state"].get("images"):
            req["state"]["images"] = [{"path": str(ROOT / i["path"])} for i in req["state"]["images"]]
            if not all(pathlib.Path(i["path"]).exists() for i in req["state"]["images"]):
                print("%-16s skipped (the picture is not part of this repository)" % name)
                continue
        answers, info, _ = d._run(req, details=True)
        for qn, g in c["questions"].items():
            r, a = info[qn], answers[qn]
            n += 1
            ids_ok = r["ids"] == g["input_ids"]
            id_bad += not ids_ok
            gh = hid["%s__%s" % (name, qn)]
            dh = float(np.abs(r["hidden"] - gh).max())
            dc = float(np.abs(r["condition_logits"] - np.array(g["condition_logits"])).max())
            gl = np.array(g["option_logits_all16"])
            dl = float(np.abs(r["slot_logits"] - gl).max())
            ga = g["answer"]
            dp = max(abs(a["probabilities"][k] - ga["probabilities"][k]) for k in ga["probabilities"])
            flip = decision(a) != decision(ga)
            aflip = a["abstained"] != ga["abstained"]
            dec_flip += flip
            abs_flip += aflip
            worst["hidden"] = max(worst["hidden"], dh); worst["cond_logits"] = max(worst["cond_logits"], dc)
            worst["logits"] = max(worst["logits"], dl); worst["prob"] = max(worst["prob"], dp)
            print("%-16s %-8s ids %s  dh %.1e  dlogit %.1e  dp %.1e  decision %s%s  conf %.3f/%.3f %s" % (
                name, qn, "exact" if ids_ok else "MISMATCH", dh, dl, dp, decision(a), " FLIP(ref %s)" % decision(ga) if flip else "",
                a["confidence"], ga["confidence"], "ABSTAIN-FLIP" if aflip else ""), flush=True)
        if c["grid_thw"] and "%s__features" % name in gimg.files:
            feats = next(iter(info.values()))["image_features"]
            dv = float(np.abs(feats - gimg["%s__features" % name]).max())
            worst["vision"] = max(worst["vision"], dv)
            print("   vision features vs oracle: max|d| %.2e (scale %.1f)" % (dv, np.abs(gimg["%s__features" % name]).max()))
    print("\nquestions %d   token-id mismatches %d   decision flips %d   abstain flips %d   (%.0f s)" % (n, id_bad, dec_flip, abs_flip, time.time() - t0))
    print("worst |d|: hidden %.2e  condition-logits %.2e  option-logits %.2e  probability %.2e  vision-features %.2e" % tuple(worst[k] for k in ("hidden", "cond_logits", "logits", "prob", "vision")))
    return 0 if not (id_bad or dec_flip) else 1


if __name__ == "__main__":
    raise SystemExit(main())

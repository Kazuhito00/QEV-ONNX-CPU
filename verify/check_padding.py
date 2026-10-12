# -*- coding: utf-8 -*-
"""GATE 3: padding / graph length cannot change the result. The same prompts run through EVERY exported fixed-length graph
(different amounts of right padding) must reproduce the oracle's two hidden states."""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qev_decide import Decider, default_model_dir  # noqa: E402


def main():
    d = Decider(default_model_dir(), intra_op_threads=6)
    gold = json.loads((ROOT / "artifacts" / "golden.json").read_text(encoding="utf-8"))["cases"]
    hid = np.load(ROOT / "artifacts" / "golden_hidden.npz")
    picks = [("choice-basic", "team"), ("multi-4", "urgency"), ("special-chars", "q"), ("noul-basic", "angry")]
    worst = 0.0
    for L in d.lengths:                                   # one session at a time
        row = []
        for case, qn in picks:
            g = gold[case]["questions"][qn]
            h = d.hidden(g["input_ids"], [g["decision_pos"], g["condition_pos"]], length=L)
            row.append(float(np.abs(h - hid["%s__%s" % (case, qn)]).max()))
        worst = max(worst, max(row))
        print("L=%-5d vs oracle: max|dh| per prompt %s" % (L, " ".join("%.1e" % r for r in row)), flush=True)
    print("check_padding: worst %.2e  %s" % (worst, "PASS" if worst < 5e-3 else "FAIL"))
    return 0 if worst < 5e-3 else 1


if __name__ == "__main__":
    raise SystemExit(main())

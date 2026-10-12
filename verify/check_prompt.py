# -*- coding: utf-8 -*-
"""GATE 1 (token ids, no model needed): prompt text, token ids, option slots and read positions of qev_decide against the
goldens produced by QEV's own runtime. Exact match required."""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qev_decide import default_model_dir  # noqa: E402
from qev_decide.tokenizer import QwenBPE  # noqa: E402
from qev_decide.protocol import condition_position, option_prompt, render_prompt, validate_request, RESULT_MARKER  # noqa: E402


def main():
    gold = json.loads((ROOT / "artifacts" / "golden.json").read_text(encoding="utf-8"))["cases"]
    tok = QwenBPE(str(default_model_dir() / "tokenizer.json"))
    enc = lambda t: tok.encode(t)  # noqa: E731
    marker = enc(RESULT_MARKER)
    bad = n = 0
    for case, c in gold.items():
        req = c["request"]
        has_img = bool(req["state"].get("images"))
        text, specs = validate_request(req["state"], req["questions"], 1 if has_img else 0)
        n_img = (c["grid_thw"][0][1] * c["grid_thw"][0][2]) // 4 if has_img else 0
        for qn, spec in specs.items():
            g = c["questions"][qn]
            user, pos = option_prompt(text, spec)
            prompt = render_prompt(user, has_img)
            if n_img:
                prompt = prompt.replace("<|image_pad|>", "<|image_pad|>" * n_img, 1)
            ids = enc(prompt)
            cp = condition_position(ids, marker)
            ok = (g["option_text"] == user and g["prompt_text"].replace("<|image_pad|>", "") == prompt.replace("<|image_pad|>", "")
                  and ids == g["input_ids"] and pos == g["positions"] and len(ids) - 1 == g["decision_pos"]
                  and cp == g["condition_pos"] and cp == len(ids) - 1 - len(marker))
            n += 1
            bad += not ok
            print("%-16s %-8s %4d tok  %s" % (case, qn, len(ids), "exact" if ok else "MISMATCH"))
    print("check_prompt: %d/%d exact" % (n - bad, n))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

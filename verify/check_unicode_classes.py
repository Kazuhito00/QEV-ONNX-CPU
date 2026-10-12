# -*- coding: utf-8 -*-
r"""Exhaustive: every Unicode code point through the pure-Python tokenizer vs transformers' tokenizer, in 4 contexts that
expose its \p{L} / \p{N} / whitespace classification.   uv run --group oracle verify/check_unicode_classes.py"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qev_decide import default_model_dir  # noqa: E402
from qev_decide.tokenizer import QwenBPE  # noqa: E402


def main():
    from transformers import AutoTokenizer
    from export.common import BASE
    mine = QwenBPE(str(default_model_dir() / "tokenizer.json"))
    hf = AutoTokenizer.from_pretrained(str(BASE))
    bad = []
    cps = [c for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF]
    for fmt in ("{}", "a{}b", "-{}-T", " {} x"):
        for i in range(0, len(cps), 4000):
            chunk = [fmt.format(chr(c)) for c in cps[i:i + 4000]]
            want = hf(chunk, add_special_tokens=False)["input_ids"]
            for c, t, w in zip(cps[i:i + 4000], chunk, want):
                if mine.encode(t) != w:
                    bad.append((c, fmt))
    print("code points %d x 4 contexts: %d mismatches %s" % (len(cps), len(bad), [("U+%04X" % c, f) for c, f in bad[:10]]))
    print("check_unicode_classes:", "PASS" if not bad else "FAIL")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

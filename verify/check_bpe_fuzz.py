# -*- coding: utf-8 -*-
r"""GATE 1a: the pure-Python Qwen BPE (qev_decide.tokenizer) against transformers' AutoTokenizer (the oracle QEV itself uses) AND the Rust
`tokenizers` object behind it (both oracles only; qev_decide never imports them). Token ids must match exactly.

    uv run --group oracle verify/check_bpe_fuzz.py [N=100000]   (needs QEV_HOME with the pinned backbone: see README)

Corpus: QEV's real prompts (the golden prompt texts) plus random mixes of CJK, emoji
(ZWJ, flags, keycaps), combining marks, ligatures, 20+ Unicode whitespace kinds, control chars, special tokens inside text,
the result marker, digits, URLs, and very long strings. Also prints the per-request tokenization time of both.
"""
import json
import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qev_decide import default_model_dir  # noqa: E402
from qev_decide.tokenizer import QwenBPE  # noqa: E402

WS = ["\t", "\n", "\x0b", "\x0c", "\r", " ", "\x1c", "\x1d", "\x1e", "\x1f", "\x85", "\xa0", "\u1680", "\u2000", "\u2002", "\u2003", "\u2009",
      "\u200a", "\u2028", "\u2029", "\u202f", "\u205f", "\u3000", "\ufeff", "\u200b", "\u200c", "\u200d", "\r\n", "\n\n\n", "   "]
SPECIAL = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<tool_response>", "<|vision_start|>", "<|vision_end|>",
           "<|image_pad|>", "<|video_pad|>", "<|fim_prefix|>", "<tool_call>", "<|im_", "<|", "|>", "VeyraResult:", "Condition state: . . . ."]
POOL = ["hello", "world", "don't", "I'll", "WE'RE", "x" * 50, "日本語", "これはテストです。", "ひらがなカタカナ漢字", "한국어", "العربية", "עברית", "ไทย", "हिन्दी",
        "\U0001F600", "\U0001F468\u200d\U0001F469\u200d\U0001F467", "\U0001F1EF\U0001F1F5", "\U0001F1FA\U0001F1F8", "1\ufe0f\u20e3", "\U0001F44D\U0001F3FD",
        "e\u0301", "a\u0308", "\u0958", "\u0915\u094d\u0937", "\ufb01", "\ufb02", "\ufb03", "\u00df", "\u0130", "\uff21\uff22", "\uff71\uff72", "\u2460", "\u00bd",
        "\u212b", "\u1e9b\u0323", "\U00020bb7", "\ud7ff", "\ue000", "\U0010ffff", "123", "1,234.5", "3.14159", "0x1F", "#!/bin/sh", "https://a.b/c?d=e#f",
        "{\"k\": [1, 2]}", "<b>x</b>", "&amp;", "\n", "\"", "'", "...", "--", "\u2014", "\u2026", "$100", "50%", "a_b", "CamelCase", "\x00", "\x01", "\x7f"]


def real_corpus():
    texts = []
    g = json.loads((ROOT / "artifacts" / "golden.json").read_text(encoding="utf-8"))["cases"]
    for c in g.values():
        for q in c["questions"].values():
            texts.append(q["prompt_text"])
            texts.append(q["option_text"])
    return texts


def rnd(rng):
    mode = rng.random()
    n = rng.randint(0, 40)
    if mode < 0.1:
        return "".join(chr(rng.choice([rng.randint(0, 0x7f), rng.randint(0x80, 0x7ff), rng.randint(0x800, 0xd7ff), rng.randint(0xe000, 0xffff), rng.randint(0x10000, 0x10ffff)])) for _ in range(n))
    parts = []
    for _ in range(n):
        r = rng.random()
        parts.append(rng.choice(POOL) if r < 0.5 else rng.choice(WS) if r < 0.75 else rng.choice(SPECIAL) if r < 0.85 else rng.choice([" ", ""]))
        if rng.random() < 0.5:
            parts.append(rng.choice([" ", "", "\n"]))
    return "".join(parts)


def main():
    n_cases = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    from export.common import BASE
    path = default_model_dir() / "tokenizer.json"
    mine = QwenBPE(str(path))
    hf = AutoTokenizer.from_pretrained(str(BASE))
    rust = hf.backend_tokenizer                  # the Rust `tokenizers` object transformers built (HF's split regex)
    rust_file = Tokenizer.from_file(str(path))   # informational: tokenizer.json as stored (its regex has \p{M}, HF's does not)
    file_diff = 0
    rng = random.Random(7)
    corpus = real_corpus()
    real = list(corpus)
    for t in real[:300]:                                   # real texts, also cut / concatenated / lightly mutated
        corpus.append(t[: rng.randint(0, len(t))])
        corpus.append(t + rng.choice(WS) + rnd(rng))
    corpus.append("word " * 6000)
    corpus.append("日本語のテスト。" * 3000)
    corpus.append("\U0001F600 " * 5000 + "\n" * 500 + " " * 700)
    corpus.append("".join(rng.choice(POOL + WS) for _ in range(60000)))
    while len(corpus) < n_cases:
        corpus.append(rnd(rng))
    bad = 0
    for i, t in enumerate(corpus):
        a = mine.encode(t)
        b = rust.encode(t, add_special_tokens=False).ids
        c = hf.encode(t, add_special_tokens=False) if (i % 5 == 0 or len(t) > 5000) else b
        file_diff += rust_file.encode(t, add_special_tokens=False).ids != b
        if not (a == b == c):
            bad += 1
            if bad <= 8:
                k = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                print("MISMATCH #%d len %d: mine==rust %s rust==hf %s  first diff @%d  text=%r" % (i, len(t), a == b, b == c, k, t[:120]))
        if i and i % 20000 == 0:
            print("  %d done, %d mismatches" % (i, bad), flush=True)
    print("cases %d (real %d), mismatches %d   [informational: raw tokenizer.json via tokenizers differs from transformers on %d cases]" % (len(corpus), len(real), bad, file_diff))
    # timing: per-request tokenization of the real prompts (one question = one prompt)
    prompts = [t for t in real if t.startswith("<|im_start|>")] or real
    for name, f in (("pure-Python", mine.encode), ("tokenizers (Rust)", lambda s: rust.encode(s, add_special_tokens=False).ids)):
        f(prompts[0])
        t0 = time.perf_counter()
        for _ in range(5):
            for p in prompts:
                f(p)
        dt = (time.perf_counter() - t0) / (5 * len(prompts))
        print("%-18s %.2f ms per prompt (%d golden prompts, mean %d chars)" % (name, dt * 1000, len(prompts), sum(map(len, prompts)) // len(prompts)))
    long_p = max(prompts, key=len)
    for name, f in (("pure-Python", mine.encode), ("tokenizers (Rust)", lambda s: rust.encode(s, add_special_tokens=False).ids)):
        t0 = time.perf_counter(); f(long_p); print("%-18s longest golden prompt (%d chars, %d tokens): %.1f ms" % (name, len(long_p), len(f(long_p)), (time.perf_counter() - t0) * 1000))
    print("check_bpe_fuzz:", "PASS" if not bad else "FAIL")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

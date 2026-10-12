# -*- coding: utf-8 -*-
"""The pure-Python Qwen byte-level BPE: a tiny tokenizer.json always, the real one when models/onnx has it."""
import json
import os
import pathlib

import pytest

from qev_decide.protocol import RESULT_MARKER, condition_position, option_prompt, render_prompt, validate_request
from qev_decide.tokenizer import QwenBPE

ROOT = pathlib.Path(__file__).resolve().parents[1]
REAL = pathlib.Path(os.environ.get("QEV_MODEL_DIR") or ROOT / "models" / "onnx") / "tokenizer.json"


def test_merges_follow_rank_order(tiny_tokenizer_json):
    tok = QwenBPE(tiny_tokenizer_json)
    assert tok.encode("the") == [tok.vocab["the"]]
    assert tok.encode("and") == [tok.vocab["and"]]
    assert tok.encode(" the") == [tok.vocab["Ġthe"]]


def test_bytes_outside_ascii_become_byte_tokens(tiny_tokenizer_json):
    tok = QwenBPE(tiny_tokenizer_json)
    assert len(tok.encode("é")) == 2                         # two UTF-8 bytes, no merge for them
    assert tok.encode("") == []


def test_added_tokens_are_cut_out_of_the_text(tiny_tokenizer_json):
    tok = QwenBPE(tiny_tokenizer_json)
    eot, pad = tok._added["<|endoftext|>"], tok._added["<|image_pad|>"]
    assert tok.encode("<|endoftext|>") == [eot]
    assert tok.encode("a<|image_pad|><|image_pad|>b") == [tok.vocab["a"], pad, pad, tok.vocab["b"]]


def test_unsupported_pipeline_is_refused(tiny_tokenizer_json, tmp_path):
    spec = json.loads(pathlib.Path(tiny_tokenizer_json).read_text(encoding="utf-8"))
    spec["normalizer"] = None
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="NFC"):
        QwenBPE(str(bad))


@pytest.mark.skipif(not REAL.exists(), reason="models/onnx/tokenizer.json not downloaded (download_model.py)")
def test_golden_prompt_ids_with_the_real_tokenizer():
    """Like verify/check_prompt.py: token ids of the golden text prompts, exactly as QEV's processor made them."""
    golden = json.loads((ROOT / "artifacts" / "golden.json").read_text(encoding="utf-8"))
    tok = QwenBPE(str(REAL))
    marker = tok.encode(RESULT_MARKER)
    checked = 0
    for case in golden["cases"].values():
        req = case["request"]
        if req.get("images") or (isinstance(req["state"], dict) and req["state"].get("images")):
            continue                                            # image prompts need the expanded placeholders: see verify.py
        text, specs = validate_request(req["state"], req["questions"], 0)
        for name, spec in specs.items():
            user_text, _ = option_prompt(text, spec)
            ids = tok.encode(render_prompt(user_text, False))
            want = case["questions"][name]
            assert ids == want["input_ids"] and condition_position(ids, marker) == want["condition_pos"]
            checked += 1
    assert checked >= 10

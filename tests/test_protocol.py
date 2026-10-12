# -*- coding: utf-8 -*-
"""Request validation, the prompt text and the typed answers (QEV's rules), without a model."""
import numpy as np
import pytest

from qev_decide import choice, noul, score
from qev_decide.protocol import (RESULT_MARKER, QevHead, candidates, condition_position, option_prompt, render_prompt,
                                 typed_answer, validate_question, validate_request)

TEMPS = [2.0, 4.0, 8.0]
THRESHOLDS = [0.5, 0.5, 0.7]


def test_validation_strips_with_unicode_white_space_and_fills_noul_defaults():
    spec = validate_question("q", noul("　 angry?  "))
    assert spec["instructions"] == "angry?"
    assert spec["criteria"] == {"true": "The proposition is true.", "false": "The proposition is false."}


@pytest.mark.parametrize("bad", [
    choice("x", {"a": "same", "b": "same"}),                    # descriptions must be distinct
    choice("x", {"a": "only one"}),                             # 2..16 options
    choice("x", {"k%d" % i: "d%d" % i for i in range(17)}),
    score("x", ["one"]),
    noul(""),
])
def test_invalid_questions_are_refused(bad):
    with pytest.raises(ValueError):
        validate_question("q", bad)


def test_request_limits():
    q = {"q": noul("yes?")}
    assert validate_request("text", q)[0] == "text"
    with pytest.raises(ValueError):
        validate_request("   ", q)                              # no evidence at all
    assert validate_request("   ", q, n_images=1)[0] == "   "   # an image is evidence
    with pytest.raises(ValueError, match="at most one image"):
        validate_request("t", q, n_images=2)
    with pytest.raises(ValueError, match="1 to 4 questions"):
        validate_request("t", {})
    with pytest.raises(ValueError, match="1 to 4 questions"):
        validate_request("t", {"q%d" % i: noul("x") for i in range(5)})


def test_choice_options_are_sorted_by_description_and_positions_map_back():
    spec = validate_question("q", choice("Which?", {"first": "zebra", "second": "apple"}))
    text, positions = option_prompt("evidence", spec)
    assert "A. apple\nB. zebra" in text
    assert positions == [1, 0]                                  # the caller's "first" (zebra) is letter B


def test_score_and_noul_keep_their_order():
    s, positions = option_prompt("e", validate_question("q", score("How?", ["low", "high"])))
    assert positions == [0, 1] and "A. low\nB. high" in s
    n, positions = option_prompt("e", validate_question("q", noul("Is it?")))
    assert positions == [0, 1]                                  # candidates are (false, true): "true" is letter B
    assert "A. false: " in n and "B. true: " in n


def test_prompt_template_and_condition_marker():
    prompt = render_prompt("Evidence:\nx\nAnswer with only the option letter.", False)
    assert prompt.startswith("<|im_start|>system\n") and prompt.endswith(RESULT_MARKER)
    assert "<think>\n\n</think>\n\n" in prompt and prompt.count(" .") == 4
    assert "<|image_pad|>" in render_prompt("t", True) and "<|image_pad|>" not in prompt
    assert condition_position([9, 1, 2, 3, 4], [3, 4]) == 2     # the token BEFORE the last marker occurrence
    with pytest.raises(ValueError):
        condition_position([1, 2, 3], [7, 8])


def test_typed_answers():
    c = typed_answer(validate_question("q", choice("x", {"a": "d1", "b": "d2"})), np.array([0.0, 3.0]), TEMPS, THRESHOLDS)
    assert c["choice"] == "b" and abs(sum(c["probabilities"].values()) - 1) < 1e-12 and c["confidence"] == c["probabilities"]["b"]
    n = typed_answer(validate_question("q", noul("x")), np.array([0.0, 0.0]), TEMPS, THRESHOLDS)
    assert n["noul"] == pytest.approx(0.5) and n["abstained"]    # 0.5 < the noul threshold 0.7
    s = typed_answer(validate_question("q", score("x", ["l", "m", "h"])), np.array([0.0, 0.0, 40.0]), TEMPS, THRESHOLDS)
    assert s["score"] > 1.9 and s["legend"] == {"0": "l", "1": "m", "2": "h"}
    assert candidates(validate_question("q", noul("x")))[1][0] == "true"


def test_head_shapes():
    rng = np.random.default_rng(0)
    head = QevHead({"readout.weight": rng.normal(size=(16, 8)), "condition_readout.weight": rng.normal(size=(3, 8)),
                    "binding_head.normalize.weight": np.ones(8), "binding_head.normalize.bias": np.zeros(8),
                    "binding_head.project.weight": rng.normal(size=(64, 8)), "binding_head.project.bias": np.zeros(64),
                    "binding_head.experts.weight": rng.normal(size=(48, 64))})
    h = rng.normal(size=8)
    assert head.logits16(h, h).shape == (16,) and head.condition_logits(h).shape == (3,)

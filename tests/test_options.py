# -*- coding: utf-8 -*-
"""Command-line option parsing of the demos (no model is loaded)."""
import argparse

import demo_inference_text as demo
from qev_decide import add_device_args, choice, noul, score
from qev_decide.protocol import validate_question


def _args(**kw):
    base = dict(choice=None, score=None, noul=None, instructions="")
    base.update(kw)
    return argparse.Namespace(**base)


def test_parse_questions_builds_each_kind():
    q = demo.parse_questions(_args(choice=["team=billing, tech"], score=["urgency=low,high"],
                                   noul=["angry=The customer is angry"]))
    assert q["team"] == choice("team", {"billing": "billing", "tech": "tech"})
    assert q["urgency"] == score("urgency", ["low", "high"])
    assert q["angry"] == noul("The customer is angry")


def test_instructions_apply_to_choice_and_score_only():
    q = demo.parse_questions(_args(choice=["t=a,b"], noul=["n=statement"], instructions="Be strict."))
    assert q["t"].instructions == "Be strict." and q["n"].instructions == "statement"


def test_builtin_example_is_valid():
    for name, question in demo.EXAMPLE_QUESTIONS.items():
        validate_question(name, question)


def test_device_options():
    import demo_inference_image  # noqa: F401  (imports fine without OpenCV)
    parser = argparse.ArgumentParser()
    add_device_args(parser)
    args = parser.parse_args(["--device", "cuda", "--require-gpu"])
    assert args.device == "cuda" and args.require_gpu and not args.gpu

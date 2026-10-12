# -*- coding: utf-8 -*-
"""Text decisions with QEV 0.2.1, on onnxruntime + numpy.

    uv run demo_inference_text.py                                           # built-in example
    uv run demo_inference_text.py --state "..." --choice "team=billing,technical"
    uv run --no-sync demo_inference_text.py --device cuda --require-gpu     # GPU (onnxruntime-gpu installed; see the README)

One question is one forward pass of the 2B torso graph (a prefill; nothing is generated). Each answer is one of the options you
supply, with probabilities, a confidence and an ``abstained`` flag. A request holds 1 to 4 questions; choice / score take 2 to 16
options. The graphs have a fixed length: --seq-len is the smallest one used, a longer prompt loads the next larger graph
(a reload of about a minute), so keep prompts short and similar in length.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from qev_decide import DEFAULT_SEQ_LEN, Decider, add_device_args, choice, default_model_dir, noul, providers_from_args, score  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent

EXAMPLE_STATE = "My card was charged twice for the same order and I want my money back. This is urgent!"
EXAMPLE_QUESTIONS = {
    "angry": noul("The customer is angry."),
    "team": choice("Which team should handle this?", {"billing": "charges and refunds", "tech": "software bugs and outages"}),
    "urgency": score("How urgent is it?", ["low", "medium", "high", "critical"]),
}


def parse_questions(args):
    """--choice NAME=a,b,c / --score NAME=low,...,high / --noul NAME=statement -> {name: Question}."""
    questions = {}
    for spec in args.choice or []:
        name, _, rest = spec.partition("=")
        options = [v.strip() for v in rest.split(",") if v.strip()]
        questions[name.strip()] = choice(args.instructions or name.strip(), {o: o for o in options})
    for spec in args.score or []:
        name, _, rest = spec.partition("=")
        levels = [v.strip() for v in rest.split(",") if v.strip()]
        questions[name.strip()] = score(args.instructions or name.strip(), levels)
    for spec in args.noul or []:
        name, _, rest = spec.partition("=")
        questions[name.strip()] = noul(rest.strip() or name.strip())
    return questions


def show(name: str, answer: dict) -> None:
    flag = "  (abstained)" if answer["abstained"] else ""
    if answer["type"] == "noul":
        print("%-10s noul   p(true)=%.3f  confidence %.3f%s" % (name, answer["noul"], answer["confidence"], flag))
        return
    spread = "  ".join("%s=%.3f" % kv for kv in sorted(answer["probabilities"].items(), key=lambda kv: -kv[1]))
    if answer["type"] == "choice":
        print("%-10s choice %-20s confidence %.3f%s\n           %s" % (name, answer["choice"], answer["confidence"], flag, spread))
    else:
        best = max(answer["probabilities"], key=answer["probabilities"].get)
        print("%-10s score  %-20s expected level %.2f, confidence %.3f%s\n           %s"
              % (name, answer["legend"][best], answer["score"], answer["confidence"], flag, spread))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", help="the evidence text; omit to run the built-in example")
    parser.add_argument("--choice", action="append", metavar="NAME=a,b,c")
    parser.add_argument("--score", action="append", metavar="NAME=low,...,high")
    parser.add_argument("--noul", action="append", metavar="NAME=statement")
    parser.add_argument("--instructions", default="")
    parser.add_argument("--model", default=None, help="model directory (default: $QEV_MODEL_DIR, else models/onnx)")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN,
                        help="smallest torso graph (256/384/512/1024/1280/2048); a longer prompt loads a larger one")
    add_device_args(parser)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.state is None:
        state, questions = EXAMPLE_STATE, EXAMPLE_QUESTIONS
    else:
        state, questions = args.state, parse_questions(args)
        if not questions:
            parser.error("give at least one --choice, --score or --noul")

    providers, require_gpu = providers_from_args(args)
    started = time.perf_counter()
    decider = Decider(args.model or default_model_dir(), intra_op_threads=args.threads, providers=providers,
                      require_gpu=require_gpu, seq_len=args.seq_len)
    load = time.perf_counter() - started
    started = time.perf_counter()
    answers = decider.decide(state, questions)
    elapsed = time.perf_counter() - started    # includes loading the first torso graph (it is created on the first question)

    if args.json:
        print(json.dumps(answers, ensure_ascii=False, indent=2))
        return 0
    print("state:\n  %s\n" % state[:160])
    for name, answer in answers.items():
        show(name, answer)
    print("\n%d questions, one forward pass each" % len(answers))
    print("setup %.1f s, first request incl. graph load %.1f s, provider: %s" % (load, elapsed, decider.provider))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

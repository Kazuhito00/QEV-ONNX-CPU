# -*- coding: utf-8 -*-
"""Typed decisions about a picture with QEV 0.2.1, on onnxruntime + numpy + Pillow.

    uv run demo_inference_image.py                          # built-in example: artifacts/images/bars.png (drawn by a script)
    uv run demo_inference_image.py photo.jpg --noul "dog=There is a dog in the picture"
    uv run demo_inference_image.py photo.jpg --state "Checkout page." --choice "next=retry,back,none"

ONE image per request (QEV's rule; PNG, JPEG, WebP or GIF, at most 20 MiB / 40 MP). QEV's processor works with 65,536 to 262,144
pixels and 32 px alignment, so a small picture is scaled UP (224x224 -> 256x256 = 64 visual tokens). Needs vision_encoder.onnx in the
model directory (download_model.py). The torso graph must be at least as long as the prompt: when the default L=384 is too short,
the runtime loads the next larger graph.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from demo_inference_text import parse_questions, show  # noqa: E402
from qev_decide import DEFAULT_SEQ_LEN, Decider, add_device_args, choice, default_model_dir, noul, providers_from_args  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
SAMPLE = ROOT / "artifacts" / "images" / "bars.png"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", help="one image file; omit to run the built-in example")
    ap.add_argument("--state", default="Look at the attached image.", help="evidence text that goes with the image")
    ap.add_argument("--choice", action="append", metavar="NAME=a,b,c")
    ap.add_argument("--score", action="append", metavar="NAME=low,...,high")
    ap.add_argument("--noul", action="append", metavar="NAME=statement")
    ap.add_argument("--instructions", default="")
    ap.add_argument("--model", default=None, help="model directory (default: $QEV_MODEL_DIR, else models/onnx)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN,
                    help="smallest torso graph (256/384/512/1024/1280/2048); a longer prompt loads a larger one")
    add_device_args(ap)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    questions = parse_questions(args)
    image = args.image
    if image is None:
        if questions:
            ap.error("give an image, or omit the questions to run the built-in example")
        image = str(SAMPLE)
        questions = {"chart": noul("The picture is a bar chart."),
                     "kind": choice("What does the picture show?", {"bars": "vertical bars of different heights",
                                                                     "circle": "a single circle", "text": "a paragraph of text"})}
    elif not questions:
        ap.error("give at least one --choice, --score or --noul")

    providers, require_gpu = providers_from_args(args)
    decider = Decider(args.model or default_model_dir(), intra_op_threads=args.threads, load_vision=True, providers=providers,
                      require_gpu=require_gpu, seq_len=args.seq_len)
    started = time.perf_counter()
    answers = decider.decide(args.state, questions, images=[image])
    elapsed = time.perf_counter() - started
    if args.json:
        print(json.dumps(answers, ensure_ascii=False, indent=2))
        return 0
    print("image: %s\n" % image)
    for name, answer in answers.items():
        show(name, answer)
    print("\nfirst request %.2f s incl. graph load (vision tower + %d questions), provider: %s" % (elapsed, len(answers), decider.provider))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

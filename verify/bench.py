# -*- coding: utf-8 -*-
"""Warm time of one question per torso bucket (indicative only: run it on a quiet machine).

    python verify/bench.py [--seq-len 384 ...] [--gpu] [--threads 6] [--repeat 3]

Without --seq-len every exported bucket is timed (one session at a time: load time and warm run are reported apart).
"""
import argparse
import json
import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qev_decide import Decider, default_model_dir  # noqa: E402
from qev_decide.runtime import DEFAULT_SEQ_LEN, add_device_args, providers_from_args  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, nargs="*", help="buckets to time (default: all exported)")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--repeat", type=int, default=3)
    add_device_args(ap)
    args = ap.parse_args()
    providers, require_gpu = providers_from_args(args)
    d = Decider(default_model_dir(), intra_op_threads=args.threads, providers=providers, require_gpu=require_gpu)
    g = json.loads((ROOT / "artifacts" / "golden.json").read_text(encoding="utf-8"))["cases"]["choice-basic"]["questions"]["team"]
    ids, reads = g["input_ids"], [g["decision_pos"], g["condition_pos"]]
    print("default bucket: %d; exported: %s; prompt %d tokens; threads %d" % (DEFAULT_SEQ_LEN, d.lengths, len(ids), args.threads))
    for L in (args.seq_len or d.lengths):
        t = time.perf_counter()
        d.hidden(ids, reads, length=L)                       # load + first run
        first = time.perf_counter() - t
        runs = []
        for _ in range(args.repeat):
            t = time.perf_counter()
            d.hidden(ids, reads, length=L)
            runs.append(time.perf_counter() - t)
        print("L=%-5d load+first %.1f s   warm median %.2f s (min %.2f)   provider %s" % (L, first, statistics.median(runs), min(runs), d.provider), flush=True)


if __name__ == "__main__":
    main()

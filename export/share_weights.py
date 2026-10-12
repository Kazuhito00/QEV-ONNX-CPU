# -*- coding: utf-8 -*-
"""Make every torso_L*.onnx read its weights from ONE external-data file (models/onnx/torso_weights.data).

    uv run --group export export/share_weights.py [--base 512] [--apply]

The graphs differ only in the unrolled sequence length; the weights are the same tensors. For each graph, every external
initializer is matched BY NAME to the base graph's and compared BYTE FOR BYTE; only if all match is the graph rewritten to point at
the shared file (without --apply nothing is changed). The per-graph .data files can then be deleted. Unmatched or different
initializers abort that graph (it keeps its own file).
"""
import argparse
import hashlib
import pathlib
import sys

import onnx
from onnx.external_data_helper import ExternalDataInfo

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "onnx"


def ext(init):
    d = {e.key: e.value for e in init.external_data}
    return d["location"], int(d.get("offset", 0)), int(d["length"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=int, default=512)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    base_path = OUT / ("torso_L%d.onnx" % args.base)
    base = onnx.load(str(base_path), load_external_data=False)
    binit = {i.name: ext(i) for i in base.graph.initializer if i.external_data}
    shared = OUT / "torso_weights.data"
    if not shared.exists():
        if not args.apply:
            print("would rename %s -> %s" % (binit and next(iter(binit.values()))[0], shared.name))
        else:
            (OUT / next(iter(binit.values()))[0]).rename(shared)
    src_base = shared if shared.exists() else OUT / next(iter(binit.values()))[0]
    fb = open(src_base, "rb")
    for g in sorted(OUT.glob("torso_L*.onnx")):
        m = onnx.load(str(g), load_external_data=False)
        inits = [i for i in m.graph.initializer if i.external_data]
        loc0 = ext(inits[0])[0]
        if loc0 == shared.name:
            print(g.name, "already shared"); continue
        own = OUT / loc0
        if not own.exists():
            own = shared
        ok, same_bytes = True, 0
        with open(own, "rb") as fo:
            for i in inits:
                loc, off, ln = ext(i)
                if i.name not in binit or binit[i.name][2] != ln:
                    print(g.name, "initializer %s not in base / size differs" % i.name); ok = False; break
                fo.seek(off); fb.seek(binit[i.name][1])
                if fo.read(ln) != fb.read(ln):
                    print(g.name, "initializer %s differs" % i.name); ok = False; break
                same_bytes += ln
        if not ok:
            continue
        print("%s: %d initializers, %.2f GB identical to the base" % (g.name, len(inits), same_bytes / 1e9))
        if args.apply:
            for i in inits:
                for e in i.external_data:
                    if e.key == "location":
                        e.value = shared.name
                    elif e.key == "offset":
                        e.value = str(binit[i.name][1])
            onnx.save(m, str(g))
            if g != base_path:
                own.unlink()
    print("done" if args.apply else "dry run only")


if __name__ == "__main__":
    main()

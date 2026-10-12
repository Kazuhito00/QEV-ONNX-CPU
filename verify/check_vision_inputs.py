# -*- coding: utf-8 -*-
"""Gate: the numpy/Pillow image path against QEV's own processor (artifacts/golden_image.npz).

pixel_values and the grid (ids of the processor's output), the M-RoPE ids, and (needs torch: run with the export or oracle group) the
bilinear taps / vision position ids against transformers' own functions.
"""
import json
import pathlib
import sys
import tempfile

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qev_decide.image import ImageConfig, bilinear_inputs, position_ids, prepare_images, vision_position_ids  # noqa: E402


# The x-* pictures are random noise of odd sizes, made here from a fixed seed in the order export/make_golden_image.py made them
# (they exercise smart_resize: wide, tall, large, tiny, RGBA). They are not stored in the repository.
EXTRA = {"x-wide": (640, 200, "RGB"), "x-tall": (150, 700, "RGB"), "x-big": (1600, 1200, "RGB"),
         "x-small": (40, 56, "RGB"), "x-rgba": (300, 220, "RGBA")}


def make_extra(folder):
    rng = np.random.default_rng(0)
    paths = {}
    for name, (w, h, mode) in EXTRA.items():
        arr = rng.integers(0, 256, (h, w, 4 if mode == "RGBA" else 3), dtype=np.uint8)
        paths[name] = pathlib.Path(folder) / (name + ".png")
        Image.fromarray(arr, mode).save(paths[name])
    return paths


def main():
    meta = json.loads((ROOT / "artifacts" / "golden_image.json").read_text(encoding="utf-8"))
    gold = np.load(ROOT / "artifacts" / "golden_image.npz")
    cfg = ImageConfig()
    bad = 0
    worst_px = 0.0
    tmp = tempfile.TemporaryDirectory()
    extra = make_extra(tmp.name)
    for name, info in meta["cases"].items():
        p = extra[name] if name in extra else ROOT / info["path"]
        if not p.exists():
            print("%-16s skipped (the picture is not part of this repository)" % name)
            continue
        prep = prepare_images([str(p)], cfg)
        ok_grid = np.array_equal(prep.image_grid_thw, gold[name + "__grid"])
        px = gold[name + "__pixel_values"]
        d = float(np.abs(prep.pixel_values - px).max()) if prep.pixel_values.shape == px.shape else float("inf")
        worst_px = max(worst_px, d)
        ids = gold[name + "__input_ids"]
        pos = position_ids(ids, prep.image_grid_thw)
        ok_pos = np.array_equal(pos, gold[name + "__position_ids"])
        flag = ok_grid and ok_pos and d < 2e-2
        bad += not flag
        print("%-16s grid %s pos_ids %s  max|dpx| %.3e  %s" % (name, "ok" if ok_grid else "BAD", "ok" if ok_pos else "BAD", d, "" if flag else "<-- FAIL"))
    try:
        import torch
        from transformers.vision_utils import get_vision_bilinear_indices_and_weights, get_vision_position_ids
    except Exception as e:
        print("transformers helpers not available (%s); bilinear/position checks skipped" % e)
    else:
        for grid in ([1, 16, 16], [1, 18, 16], [1, 20, 14], [1, 12, 40], [1, 44, 10], [1, 26, 36], [1, 14, 20]):
            g = torch.tensor([grid])
            bi, bw = get_vision_bilinear_indices_and_weights(g, num_grid_per_side=48, spatial_merge_size=2)
            mi, mw = bilinear_inputs(np.array([grid]), 48, 2)
            vp = get_vision_position_ids(g, 2)
            ok = (np.array_equal(mi, bi.numpy()) and np.abs(mw - bw.numpy()).max() < 1e-5
                  and np.array_equal(vision_position_ids(np.array([grid]), 2), vp.numpy()))
            bad += not ok
            print("grid", grid, "bilinear+vision position ids", "ok" if ok else "FAIL")
    print("worst |dpixel| %.3e" % worst_px)
    print("RESULT:", "PASS" if not bad else "FAIL (%d)" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

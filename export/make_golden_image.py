# -*- coding: utf-8 -*-
"""Stage-by-stage image goldens from QEV's own encoder (torch, CPU fp32): pixel_values, grid, vision features, M-RoPE ids.

    uv run --group oracle export/make_golden_image.py

Pictures are listed in artifacts/golden.json (cases with state.images). Writes artifacts/golden_image.npz
(keys <case>__pixel_values, __grid, __features, __position_ids) and artifacts/golden_image.json (processor class, sizes).
Also adds a few extra picture shapes (non-square, large, EXIF-less RGBA) to cover smart_resize branches.
"""
import json
import os
import pathlib
import sys

import numpy as np
import torch
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QEV_HOME", str(ROOT / "models" / "qev_home"))
torch.set_num_threads(6)

import qev


def main():
    model = qev.load(device="cpu")
    enc = model.encoder
    proc = enc.processor
    gold = json.loads((ROOT / "artifacts" / "golden.json").read_text(encoding="utf-8"))["cases"]
    items = {}
    for name, c in gold.items():
        for im in c["request"]["state"].get("images", []):
            items[name] = ROOT / im["path"]
    # extra synthetic shapes (saved under artifacts/_tmp_images, not committed)
    tmp = ROOT / "artifacts" / "_tmp_images"
    tmp.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for nm, (w, h, mode) in {"x-wide": (640, 200, "RGB"), "x-tall": (150, 700, "RGB"), "x-big": (1600, 1200, "RGB"),
                             "x-small": (40, 56, "RGB"), "x-rgba": (300, 220, "RGBA")}.items():
        arr = rng.integers(0, 256, (h, w, 4 if mode == "RGBA" else 3), dtype=np.uint8)
        p = tmp / (nm + ".png")
        Image.fromarray(arr, mode).save(p)
        items[nm] = p
    out, meta = {}, {"processor": type(proc.image_processor).__name__, "size": dict(proc.image_processor.size), "cases": {}}
    prompt = "<|vision_start|><|image_pad|><|vision_end|>hello"
    for name, path in items.items():
        from PIL import ImageOps
        with Image.open(path) as src:
            image = ImageOps.exif_transpose(src).convert("RGB")
        inp = proc(text=[prompt], images=[image], return_tensors="pt", padding=True)
        with torch.inference_mode():
            feats = enc.model.get_image_features(inp["pixel_values"], inp["image_grid_thw"])
            feats = feats.pooler_output if hasattr(feats, "pooler_output") else feats
            if isinstance(feats, (list, tuple)):
                feats = torch.cat(list(feats))
            pos, _ = enc.model.get_rope_index(inp["input_ids"], inp["mm_token_type_ids"], inp["image_grid_thw"], None, inp["attention_mask"])
        out[name + "__pixel_values"] = inp["pixel_values"].numpy()
        out[name + "__grid"] = inp["image_grid_thw"].numpy()
        out[name + "__features"] = feats.float().numpy()
        out[name + "__position_ids"] = pos[:, 0].numpy()
        out[name + "__input_ids"] = inp["input_ids"][0].numpy()
        meta["cases"][name] = {"path": path.relative_to(ROOT).as_posix(),
                               "grid": inp["image_grid_thw"].tolist(), "feat_shape": list(feats.shape)}
        print(name, meta["cases"][name], flush=True)
    np.savez(ROOT / "artifacts" / "golden_image.npz", **out)
    (ROOT / "artifacts" / "golden_image.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Paths of the pinned downloads (made by `qev download` into models/qev_home) and the LoRA/head readers."""
from __future__ import annotations

import json
import os
import pathlib

import numpy as np
from safetensors import safe_open

ROOT = pathlib.Path(__file__).resolve().parents[1]
QEV_HOME = pathlib.Path(os.environ.get("QEV_HOME") or ROOT / "models" / "qev_home")
BACKBONE_REV = "15852e8c16360a2fea060d615a32b45270f8a8fc"
CHECKPOINT_REV = "0d2d13ffb3c392071ea00b6bdb903c4b84dff48d"
BASE = QEV_HOME / "hub" / "models--Qwen--Qwen3.5-2B" / "snapshots" / BACKBONE_REV
CKPT = QEV_HOME / "checkpoints" / CHECKPOINT_REV
BASE_WEIGHTS = BASE / "model.safetensors-00001-of-00001.safetensors"
LORA_PREFIX = "encoder.model.language_model."
N_LORA_MODULES = 150          # 18 linear-attention layers x 6 + 6 full-attention layers x 7 ... checked at load


def manifest() -> dict:
    return json.loads((CKPT / "manifest.json").read_text(encoding="utf-8"))


def lora_pairs():
    """({decoder weight name: (A [r,in], B [out,r])}, scale). Names are the bare text decoder's (layers.N.mlp.up_proj.weight)."""
    ad = manifest()["adaptation"]
    scale = float(ad["alpha"]) / float(ad["rank"])
    import torch
    pairs = {}
    with safe_open(str(CKPT / "head.safetensors"), framework="pt") as h:
        for key in h.keys():
            if not key.startswith(LORA_PREFIX):
                continue
            rest = key[len(LORA_PREFIX):]
            module, _, ab = rest.rpartition(".lora_")
            pairs.setdefault(module + ".weight", {})[ab] = h.get_tensor(key).float()
    return {k: (v["a"], v["b"]) for k, v in pairs.items()}, scale


def head_arrays() -> dict:
    """The numpy head: readout, condition_readout and the binding head (fp32) straight from head.safetensors."""
    out = {}
    with safe_open(str(CKPT / "head.safetensors"), framework="np") as h:
        for key in h.keys():
            if key.startswith(LORA_PREFIX):
                continue
            out[key] = h.get_tensor(key).astype(np.float32)
    return out

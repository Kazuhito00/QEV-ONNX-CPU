# -*- coding: utf-8 -*-
"""Export the QEV torso (Qwen3.5-2B + merged LoRA, scale alpha/r = 2) to ONNX: one prefill, hidden states out.

    uv run --group export export/export_onnx.py --seq-len 256

    input_ids [1, L], attention_mask [1, L], position_ids [3, 1, L], read_positions [R],
    image_features [N, 2048], image_slots [N]          ->  hidden [R, 2048]  (post final RMSNorm)

``read_positions`` = [decision token (the last one), condition token (the "\\n" before "VeyraResult:")]; both come from
ONE forward. The heads (readout, condition_readout, binding head) run in numpy from head.npz.

The length L is FIXED (a multiple of 64): the linear-attention layers loop over 64-token chunks in Python, which the
exporter unrolls at the traced length. The runtime right-pads to the smallest exported L; the model is causal and every
read position precedes the padding (verify/check_padding.py).

Also writes models/onnx/head.npz, tokenizer.json, preprocessor_config.json, manifest.json and graph_meta_L*.json.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from export.common import BASE, BASE_WEIGHTS, N_LORA_MODULES, head_arrays, lora_pairs, manifest  # noqa: E402

CHUNK = 64
PAD = 248044


class TorsoGraph(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, position_ids, read_positions, image_features, image_slots):
        embeds = self.model.get_input_embeddings()(input_ids)
        embeds = embeds.index_copy(1, image_slots, image_features.unsqueeze(0).to(embeds.dtype))
        out = self.model(inputs_embeds=embeds, attention_mask=attention_mask, position_ids=position_ids,
                         use_cache=False, return_dict=True)
        return out.last_hidden_state[0].index_select(0, read_positions)


def load_merged_model(dtype=torch.float32):
    """Qwen3_5TextModel with the LoRA folded in, filled one weight at a time (peak = the model, ~7.6 GB in fp32)."""
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    config = AutoConfig.from_pretrained(str(BASE))
    torch.set_default_dtype(torch.float32)
    model = Qwen3_5TextModel(config.text_config).eval()
    params = dict(model.named_parameters())
    pairs, scale = lora_pairs()
    if len(pairs) != N_LORA_MODULES or scale != 2.0:
        raise SystemExit("expected %d LoRA modules at scale 2.0, got %d at %s" % (N_LORA_MODULES, len(pairs), scale))
    seen, merged = set(), 0
    prefix = "model.language_model."
    with torch.no_grad(), safe_open(str(BASE_WEIGHTS), framework="pt") as handle:
        for key in handle.keys():
            if not key.startswith(prefix):
                continue
            name = key[len(prefix):]
            if name not in params:
                raise SystemExit("base weight %s has no parameter in the decoder" % name)
            w = handle.get_tensor(key).float()
            if name in pairs:
                a, b = pairs[name]
                w = w + (b @ a) * scale
                merged += 1
            params[name].copy_(w)
            seen.add(name)
    missing = [n for n in params if n not in seen]
    if missing:
        raise SystemExit("decoder parameters never loaded: %s" % missing[:5])
    if merged != len(pairs):
        raise SystemExit("merged %d of %d adapter modules (a key found no weight)" % (merged, len(pairs)))
    print("loaded %d tensors, merged %d LoRA modules (scale %.1f)" % (len(seen), merged, scale), flush=True)
    return config, model.to(dtype)


def padded_inputs(ids, length, position_ids=None):
    real = len(ids)
    if real > length:
        raise ValueError("%d tokens exceeds the graph's fixed length %d" % (real, length))
    input_ids = torch.tensor([list(ids) + [PAD] * (length - real)], dtype=torch.long)
    mask = torch.tensor([[1] * real + [0] * (length - real)], dtype=torch.long)
    if position_ids is None:
        pos = torch.arange(length, dtype=torch.long)[None, None, :].repeat(3, 1, 1)
    else:
        pos = torch.zeros(3, 1, length, dtype=torch.long)
        pos[:, 0, :real] = torch.as_tensor(position_ids)
        pos[:, 0, real:] = torch.arange(real, length) + int(position_ids.max()) - real + 1
    return input_ids, mask, pos


def golden_check(graph, length, hidden):
    """The merged torso (in torch) against the engine's own hidden states, before anything is exported."""
    gold = json.loads((ROOT / "artifacts" / "golden.json").read_text(encoding="utf-8"))["cases"]
    hid = np.load(ROOT / "artifacts" / "golden_hidden.npz")
    img = np.load(ROOT / "artifacts" / "golden_image.npz")
    feats0 = torch.zeros(0, hidden)
    slots0 = torch.zeros(0, dtype=torch.long)
    worst = 0.0
    todo = [("choice-basic", "team"), ("multi-4", "urgency"), ("special-chars", "q"), ("injection", "q"), ("img-shape", "shape")]
    for case, qid in todo:
        q = gold[case]["questions"][qid]
        ids = q["input_ids"]
        if len(ids) > length:
            continue
        reads = torch.tensor([q["decision_pos"], q["condition_pos"]])
        if case.startswith("img"):
            from qev_decide.image import position_ids as mrope
            pos = mrope(ids, np.array(gold[case]["grid_thw"]))
            slots = torch.tensor(np.flatnonzero(np.asarray(ids) == 248056))
            feats = torch.from_numpy(img[case + "__features"])
        else:
            pos, slots, feats = None, slots0, feats0
        input_ids, mask, positions = padded_inputs(ids, length, pos)
        with torch.no_grad():
            got = graph(input_ids, mask, positions, reads, feats, slots).float().numpy()
        want = hid["%s__%s" % (case, qid)]
        diff = float(np.abs(got - want).max())
        worst = max(worst, diff)
        print("  torch torso vs engine  %-22s max|dh| %.2e  (scale %.1f)" % (case + "/" + qid, diff, np.abs(want).max()), flush=True)
    return worst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="models/onnx")
    ap.add_argument("--seq-len", type=int, default=256, help="fixed prompt length (a multiple of 64)")
    ap.add_argument("--no-graph", action="store_true", help="only write the small files (head.npz, tokenizer, meta)")
    ap.add_argument("--skip-check", action="store_true")
    args = ap.parse_args()
    if args.seq_len % CHUNK:
        raise SystemExit("--seq-len must be a multiple of %d" % CHUNK)
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    man = manifest()
    heads = head_arrays()
    np.savez(out_dir / "head.npz", **heads)
    for name in ("tokenizer.json", "preprocessor_config.json"):
        shutil.copy(BASE / name, out_dir / name)
    cal = man["calibration"]
    meta = {"graph": None, "seq_len": args.seq_len, "dtype": "float32", "chunk": CHUNK, "pad_token_id": PAD,
            "max_total_tokens": man["encoder"]["max_total_tokens"], "image_min_pixels": man["encoder"]["image_min_pixels"],
            "image_max_pixels": man["encoder"]["image_max_pixels"], "reasoning_slots": man["reasoning_slots"],
            "question_types": ["choice", "score", "noul"], "temperatures": cal["temperatures"],
            "abstain_thresholds": cal["abstain_thresholds"], "always_abstain_types": cal["always_abstain_types"],
            "backbone": man["backbone"], "checkpoint_revision": CKPT_REV()}
    (out_dir / "qev_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if args.no_graph:
        return 0

    started = time.time()
    config, model = load_merged_model()
    print("model ready in %.0f s" % (time.time() - started), flush=True)
    graph = TorsoGraph(model).eval()
    hidden = config.text_config.hidden_size
    if not args.skip_check:
        worst = golden_check(graph, args.seq_len, hidden)
        if worst > 5e-2:
            raise SystemExit("the merged torso does not reproduce the engine's hidden states (%.2e); not exporting" % worst)

    q = json.loads((ROOT / "artifacts" / "golden.json").read_text(encoding="utf-8"))["cases"]["choice-basic"]["questions"]["team"]
    input_ids, mask, positions = padded_inputs(q["input_ids"], args.seq_len)
    reads = torch.tensor([q["decision_pos"], q["condition_pos"]], dtype=torch.long)
    feats = torch.zeros(4, hidden)
    slots = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    target = out_dir / ("torso_L%d.onnx" % args.seq_len)
    started = time.time()
    torch.onnx.export(
        graph, (input_ids, mask, positions, reads, feats, slots), str(target),
        input_names=["input_ids", "attention_mask", "position_ids", "read_positions", "image_features", "image_slots"],
        output_names=["hidden"],
        dynamic_shapes={"input_ids": {}, "attention_mask": {}, "position_ids": {}, "read_positions": {0: "reads"},
                        "image_features": {0: "visual"}, "image_slots": {0: "visual"}},
        opset_version=18, dynamo=True, external_data=True, optimize=False)
    print("exported in %.0f s -> %s" % (time.time() - started, target), flush=True)
    meta["graph"] = target.name
    (out_dir / ("graph_meta_L%d.json" % args.seq_len)).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return 0


def CKPT_REV():
    from export.common import CHECKPOINT_REV
    return CHECKPOINT_REV


if __name__ == "__main__":
    raise SystemExit(main())

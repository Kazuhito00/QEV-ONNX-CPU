# -*- coding: utf-8 -*-
"""Run QEV's OWN runtime (torch, CPU fp32) on export/cases.py and record everything the port must reproduce.

    uv run --group oracle export/make_golden.py

Per question: the prompt text, token ids, decision/condition positions, the two hidden states, condition logits, final
logits (16 slots), the model.predict() answer (calibrated probabilities, abstain flag). Writes artifacts/golden.json and
artifacts/golden_hidden.npz (hidden states: [2, 2048] per question = decision, condition).
"""
import json
import os
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QEV_HOME", str(ROOT / "models" / "qev_home"))
torch.set_num_threads(6)

import qev
from veyra.option_model import option_prompt
from veyra.reasoning_workspace import RESULT_MARKER, condition_positions
from veyra.schema import DecisionRequest
from export.cases import build


class Spy:
    def __init__(self, proc):
        self._p = proc
        self.last = None

    def __getattr__(self, k):
        return getattr(self._p, k)

    def __call__(self, *a, **k):
        r = self._p(*a, **k)
        self.last = (r, k.get("text"))
        return r


def main():
    model = qev.load(device="cpu")
    proc = model.encoder.processor
    spy = Spy(proc)
    model.encoder.processor = spy
    out, hid = {}, {}
    for name, req in build().items():
        req = json.loads(json.dumps(req))
        if req.get("images"):
            req["state"]["images"] = [{"path": str(ROOT / p)} for p in req["images"]]
        rel = req.pop("images", None)
        request = DecisionRequest.model_validate(req)
        with torch.inference_mode():
            states, mappings, tokens, cond_states = model.encode(request, None, capture_condition=True)
            inp, texts = spy.last
            cond_logits = model.condition_readout(cond_states)
            all_logits = model.readout(states) + model.binding_head(states, cond_logits)
            ans = model.predict(request, None)
        marker = proc.tokenizer.encode(RESULT_MARKER, add_special_tokens=False)
        cpos = condition_positions(inp["input_ids"], inp["attention_mask"], marker).tolist()
        ids_all, mask = inp["input_ids"].tolist(), inp["attention_mask"].tolist()
        qrec = {}
        for i, (qname, q) in enumerate(request.questions.items()):
            first = min(j for j, m in enumerate(mask[i]) if m)
            last = max(j for j, m in enumerate(mask[i]) if m)
            text, positions = option_prompt(request, q)
            qrec[qname] = dict(prompt_text=texts[i], option_text=text, positions=positions,
                               input_ids=[t for t, m in zip(ids_all[i], mask[i]) if m], n_tokens=sum(mask[i]),
                               pad_left=first, decision_pos=last - first, condition_pos=cpos[i] - first,
                               option_logits_all16=all_logits[i].tolist(), condition_logits=cond_logits[i].tolist(),
                               answer=ans["answers"][qname])
            hid["%s__%s" % (name, qname)] = np.stack([states[i].numpy(), cond_states[i].numpy()])
        req_out = dict(req)
        if rel:
            req_out["state"] = dict(req["state"], images=[{"path": p} for p in rel])
        out[name] = dict(request=req_out, tokens_total=tokens,
                         grid_thw=inp["image_grid_thw"].tolist() if "image_grid_thw" in inp else None, questions=qrec)
        print(name, {k: (v["n_tokens"], v["answer"].get("choice", v["answer"].get("noul", v["answer"].get("score"))),
                         v["answer"]["abstained"]) for k, v in qrec.items()}, flush=True)
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "artifacts" / "golden.json").write_text(json.dumps(dict(
        qev_version=qev.__version__, backbone="Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc",
        torch=torch.__version__, device="cpu fp32", cases=out), ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez(ROOT / "artifacts" / "golden_hidden.npz", **hid)


if __name__ == "__main__":
    main()

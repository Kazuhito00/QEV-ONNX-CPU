# -*- coding: utf-8 -*-
"""QEV 0.2.1 typed decisions on onnxruntime + numpy (+ Pillow for images). No torch, no transformers.

One question = one prefill of the torso graph (Qwen3.5-2B with QEV's LoRA folded in) that returns the post-final-norm
hidden state at two positions: the last prompt token (decision) and the token before ``VeyraResult:`` (condition).
The heads (readout, condition_readout, binding head), the per-type temperature and the abstain threshold run in numpy.

The graphs have a FIXED length L (a multiple of 64; see export/export_onnx.py): the prompt is right-padded to the smallest
exported L that holds it (padding cannot change the result: the model is causal and both read positions precede it).
A prompt longer than the largest exported graph raises ValueError (QEV itself refuses requests above 2048 tokens in
total). Only ONE session is kept in memory at a time (each fp32 graph holds 7.5 GB of weights); ``keep_sessions`` raises it.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import onnxruntime as ort

from ._ort import add_device_args, gpu_providers, make_session, providers_from_args, select_providers

from .tokenizer import QwenBPE
from .protocol import (QevHead, Question, QUESTION_TYPES, RESULT_MARKER, condition_position, option_prompt, render_prompt,
                       typed_answer, validate_request)

__all__ = ["DEFAULT_SEQ_LEN", "Decider", "add_device_args", "default_model_dir", "gpu_providers", "providers_from_args",
           "select_providers"]

MODEL_DIR_ENV = "QEV_MODEL_DIR"

CHUNK = 64
DEFAULT_SEQ_LEN = 384          # smallest bucket used by default; a longer prompt escalates to the next exported bucket


def default_model_dir() -> pathlib.Path:
    """``$QEV_MODEL_DIR`` if set, else ``models/onnx`` of this checkout (where download_model.py puts the files)."""
    env = os.environ.get(MODEL_DIR_ENV)
    return pathlib.Path(env) if env else pathlib.Path(__file__).resolve().parents[1] / "models" / "onnx"


class Decider:
    """Load once, then :meth:`decide` / :meth:`predict`.

    >>> d = Decider("models/onnx")
    >>> d.decide("My card was charged twice.", {"team": choice("Which team?", {"billing": "charges", "tech": "bugs"})})
    """

    def __init__(self, model_dir: str | pathlib.Path = "models/onnx", intra_op_threads: Optional[int] = None,
                 load_vision: bool = False, keep_sessions: int = 1, providers: Optional[Sequence[str]] = None,
                 seq_len: int = DEFAULT_SEQ_LEN, require_gpu: bool = False) -> None:
        """``providers``: onnxruntime execution providers, e.g. ``select_providers("cuda")`` (default: CPU only).
        ``require_gpu``: raise RuntimeError instead of falling back to the CPU when a session cannot use the GPU."""
        root = pathlib.Path(model_dir)
        self.root = root
        graphs = sorted(root.glob("torso_L*.onnx"), key=lambda p: int(p.stem.split("_L")[1]))
        if not graphs:
            raise FileNotFoundError("no torso_L*.onnx under %s; run export/export_onnx.py" % root)
        self.lengths = [int(p.stem.split("_L")[1]) for p in graphs]
        self._graphs = dict(zip(self.lengths, graphs))
        cfg = json.loads((root / "qev_config.json").read_text(encoding="utf-8"))
        self.config = cfg
        self.max_total_tokens = int(cfg["max_total_tokens"])
        self.slots = int(cfg["reasoning_slots"])
        self.pad_id = int(cfg["pad_token_id"])
        self.temperatures = [float(x) for x in cfg["temperatures"]]
        self.thresholds = [float(x) for x in cfg["abstain_thresholds"]]
        self.always_abstain = list(cfg["always_abstain_types"])
        self.tokenizer = QwenBPE(str(root / "tokenizer.json"))
        self.marker_ids = self.encode(RESULT_MARKER)
        head = np.load(root / "head.npz")
        self.head = QevHead({k: head[k] for k in head.files})
        self._threads = intra_op_threads
        self.providers = list(providers) if providers else None
        self.require_gpu = require_gpu
        self.seq_len = int(seq_len)
        self.keep_sessions = max(1, keep_sessions)
        self._sessions: Dict[int, ort.InferenceSession] = {}
        self.vision = None
        self.image_config = None
        if load_vision:
            self._load_vision()

    @property
    def provider(self) -> str:
        """The execution provider the most recently used torso session actually runs on (what it really got, not what was
        asked). Sessions are created on the first question; before that there is nothing to report."""
        if not self._sessions:
            return "(no graph loaded yet)"
        return next(reversed(self._sessions.values())).get_providers()[0]

    # ------------------------------------------------------------------ helpers

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def _load_vision(self):
        from .image import ImageConfig, VisionEncoder
        cfg = ImageConfig.from_pretrained(self.root, min_pixels=int(self.config["image_min_pixels"]),
                                          max_pixels=int(self.config["image_max_pixels"]))
        self.vision = VisionEncoder(self.root, self._threads, cfg, self.providers, self.require_gpu)

    def _session(self, length: int) -> ort.InferenceSession:
        if length not in self._sessions:
            while len(self._sessions) >= self.keep_sessions:
                self._sessions.pop(next(iter(self._sessions)))
            self._sessions[length] = make_session(self._graphs[length], self._threads, self.providers, self.require_gpu)
        return self._sessions[length]

    def graph_length(self, n_tokens: int) -> int:
        need = -(-n_tokens // CHUNK) * CHUNK
        length = next((L for L in self.lengths if L >= max(need, self.seq_len)), None)
        if length is None:                  # seq_len above every bucket that fits: any bucket that holds the prompt
            length = next((L for L in self.lengths if L >= need), None)
        if length is None:
            raise ValueError("a prompt of %d tokens needs a graph of at least %d; exported lengths: %s"
                             % (n_tokens, need, self.lengths))
        return length

    def hidden(self, ids: Sequence[int], read_positions: Sequence[int], image_features: Optional[np.ndarray] = None,
               image_slots: Optional[np.ndarray] = None, position_ids: Optional[np.ndarray] = None,
               length: Optional[int] = None) -> np.ndarray:
        """Torso hidden states [R, 2048] at ``read_positions`` for one prompt."""
        n = len(ids)
        length = length or self.graph_length(n)
        if length < n or length not in self._graphs:
            raise ValueError("no exported graph of length %s holds %d tokens" % (length, n))
        input_ids = np.full((1, length), self.pad_id, dtype=np.int64)
        input_ids[0, :n] = ids
        mask = np.zeros((1, length), dtype=np.int64)
        mask[0, :n] = 1
        if position_ids is None:
            positions = np.broadcast_to(np.arange(length, dtype=np.int64), (3, 1, length)).copy()
        else:
            positions = np.zeros((3, 1, length), dtype=np.int64)
            positions[:, 0, :n] = position_ids
            positions[:, 0, n:] = np.arange(n, length) + int(position_ids.max()) - n + 1
        feats = np.zeros((0, 2048), np.float32) if image_features is None else image_features.astype(np.float32)
        slots = np.zeros((0,), np.int64) if image_slots is None else image_slots.astype(np.int64)
        feeds = {"input_ids": input_ids, "attention_mask": mask, "position_ids": positions,
                 "read_positions": np.asarray(read_positions, dtype=np.int64), "image_features": feats,
                 "image_slots": slots}
        return self._session(length).run(["hidden"], feeds)[0]

    # ------------------------------------------------------------------ one question

    def question_ids(self, state_text: str, spec: Mapping[str, Any], n_image_tokens: int = 0):
        """(ids, positions) of one question's prompt: token ids exactly as QEV's processor produces them."""
        user_text, positions = option_prompt(state_text, spec)
        prompt = render_prompt(user_text, n_image_tokens > 0, self.slots)
        if n_image_tokens:
            from .image import IMAGE_PAD
            prompt = prompt.replace(IMAGE_PAD, IMAGE_PAD * n_image_tokens, 1)
        return self.encode(prompt), positions

    def logits_for(self, ids: Sequence[int], positions: Sequence[int], extra: Optional[Dict[str, Any]] = None):
        """(logits per caller option, hidden [2, 2048], 16 slot logits, condition logits) for one prompt."""
        cond = condition_position(ids, self.marker_ids)
        h = self.hidden(ids, [len(ids) - 1, cond], **(extra or {}))
        slot_logits = self.head.logits16(h[0], h[1])
        return slot_logits[list(positions)], h, slot_logits, self.head.condition_logits(h[1])

    # ------------------------------------------------------------------ public

    def decide(self, state: Any, questions: Mapping[str, Any], images: Sequence[Any] = ()) -> Dict[str, Dict[str, Any]]:
        return self.predict({"state": state, "questions": questions, "images": list(images)})["answers"]

    def predict(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        answers, _, total = self._run(request)
        return {"model": "qev-0.2.1-onnx", "answers": answers, "usage": {"input_tokens": total, "output_tokens": 0}}

    def _run(self, request: Mapping[str, Any], details: bool = False):
        """(answers, per-question details, total tokens). ``details`` keeps ids / hidden states / logits for verification."""
        images = list(request.get("images") or [])
        state = request["state"]
        if isinstance(state, Mapping) and state.get("images"):
            images = images + [i["path"] if isinstance(i, Mapping) else i for i in state["images"]]
        text, specs = validate_request(state, request["questions"], len(images))
        feats, prepared = None, None
        if images:
            from .image import prepare_images
            if self.vision is None:
                self._load_vision()
            prepared = prepare_images(images, self.vision.config)
            feats = self.vision.encode(prepared)
        prompts = []
        for name, spec in specs.items():
            ids, positions = self.question_ids(text, spec, 0 if feats is None else len(feats))
            prompts.append((name, spec, ids, positions))
        total = sum(len(p[2]) for p in prompts)
        if total > self.max_total_tokens:
            raise ValueError("request has %d tokens, exceeding %d; reduce input size explicitly" % (total, self.max_total_tokens))
        answers: Dict[str, Dict[str, Any]] = {}
        info: Dict[str, Dict[str, Any]] = {}
        for name, spec, ids, positions in prompts:
            image_kw = None
            if feats is not None:
                from .image import IMAGE_TOKEN_ID, position_ids
                slots = np.flatnonzero(np.asarray(ids) == IMAGE_TOKEN_ID)
                if len(slots) != len(feats):
                    raise ValueError("evidence text contains the image placeholder token; refusing")
                image_kw = {"image_features": feats, "image_slots": slots,
                            "position_ids": position_ids(ids, prepared.image_grid_thw)}
            logits, h, slot_logits, cond_logits = self.logits_for(ids, positions, image_kw)
            answers[name] = typed_answer(spec, logits, self.temperatures, self.thresholds,
                                         always_abstain_types=self.always_abstain)
            if details:
                info[name] = {"ids": ids, "hidden": h, "slot_logits": slot_logits, "condition_logits": cond_logits,
                              "positions": positions, "image_features": feats}
        return answers, info, total

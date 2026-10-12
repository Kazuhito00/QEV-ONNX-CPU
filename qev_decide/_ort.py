# -*- coding: utf-8 -*-
"""onnxruntime session creation shared by the torso and the vision tower, plus the CPU / GPU choice (--device, --gpu, --require-gpu).

The CPU is the default. A GPU run never pretends: the providers a session really got are checked, a fallback to the CPU is
announced on stderr, and ``require_gpu`` turns it into an error.
"""
from __future__ import annotations

import pathlib
import sys
from typing import List, Optional, Sequence, Tuple

import onnxruntime as ort

GPU_PROVIDERS = ("CUDAExecutionProvider", "DmlExecutionProvider")
CPU = "CPUExecutionProvider"


def gpu_providers() -> List[str]:
    """GPU execution providers available in this onnxruntime build, CPU last."""
    available = ort.get_available_providers()
    return [p for p in GPU_PROVIDERS if p in available] + [CPU]


def select_providers(device: str = "cpu", available: Optional[Sequence[str]] = None) -> List[str]:
    """Execution providers for ``device``: ``cpu`` (default), ``cuda`` (CUDA, then CPU) or ``auto`` (CUDA / DirectML if this
    onnxruntime build has them, then CPU). ``cuda`` lists CUDAExecutionProvider even when the build lacks it: the session
    then warns (or fails with ``require_gpu``) instead of silently pretending to be on the GPU."""
    available = list(ort.get_available_providers() if available is None else available)
    if device == "cpu":
        return [CPU]
    if device == "cuda":
        return ["CUDAExecutionProvider", CPU]
    if device == "auto":
        return [p for p in GPU_PROVIDERS if p in available] + [CPU]
    raise ValueError("device must be cpu, cuda or auto, got %r" % (device,))


def add_device_args(parser) -> None:
    """``--device cpu|cuda|auto`` (default cpu), ``--gpu`` (= ``--device auto``) and ``--require-gpu`` for a demo's parser."""
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu",
                        help="cpu (default); cuda = CUDAExecutionProvider then CPU; auto = CUDA / DirectML when this "
                             "onnxruntime build has it, else CPU")
    parser.add_argument("--gpu", action="store_true", help="same as --device auto")
    parser.add_argument("--require-gpu", action="store_true",
                        help="fail instead of falling back to the CPU when no GPU provider can be used")


def providers_from_args(args, available: Optional[Sequence[str]] = None) -> Tuple[Optional[List[str]], bool]:
    """(providers, require_gpu) for ``Decider(...)`` from the arguments added by :func:`add_device_args`."""
    device = "auto" if args.gpu and args.device == "cpu" else args.device
    if args.require_gpu and device == "cpu":
        device = "auto"
    providers = select_providers(device, available)
    have = list(ort.get_available_providers() if available is None else available)
    if device != "cpu" and (providers[0] == CPU or providers[0] not in have):
        message = ("no %s in this onnxruntime build (install onnxruntime-gpu or onnxruntime-directml "
                   "instead of onnxruntime)" % ("GPU provider" if providers[0] == CPU else providers[0]))
        if args.require_gpu:
            raise SystemExit("error: " + message)
        print("warning: %s; running on CPU" % message, file=sys.stderr)
    return (None if device == "cpu" else providers), bool(args.require_gpu)


def make_session(path, threads: Optional[int], providers: Optional[Sequence[str]] = None,
                 require_gpu: bool = False) -> "ort.InferenceSession":
    """``providers`` None = CPU only. A GPU that cannot hold the graph (or lacks its DLLs) warns and falls back to the CPU,
    or raises RuntimeError with ``require_gpu``. The providers the session really got are checked, not assumed."""
    path = pathlib.Path(path)
    options = ort.SessionOptions()
    options.log_severity_level = 3
    if threads:
        options.intra_op_num_threads = threads
    wanted = list(providers) if providers else [CPU]
    if wanted[0] == CPU:
        if require_gpu:
            raise RuntimeError("require_gpu is set but no GPU provider was asked for")
        return ort.InferenceSession(str(path), sess_options=options, providers=wanted)
    try:
        session = ort.InferenceSession(str(path), sess_options=options, providers=wanted)
        if session.get_providers()[0] != CPU:
            return session
        reason = "the provider was dropped while loading (missing CUDA / cuDNN DLLs on PATH?)"
    except Exception as exc:
        reason = str(exc).splitlines()[0][:160] if str(exc) else type(exc).__name__
    if require_gpu:
        raise RuntimeError("%s could not run %s on the GPU (%s)" % (wanted[0], path.name, reason))
    print("warning: %s could not load %s (%s); using the CPU" % (wanted[0], path.name, reason), file=sys.stderr)
    return ort.InferenceSession(str(path), sess_options=options, providers=[CPU])

# -*- coding: utf-8 -*-
"""Run a verify script / the ask-adapter smoke with `tokenizers`, `transformers` and `torch` made un-importable,
proving the runtime does not need them.   python verify/run_blocked.py verify/verify.py [args...]"""
import runpy
import sys

for m in ("tokenizers", "transformers", "torch", "safetensors"):
    sys.modules[m] = None
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")

# -*- coding: utf-8 -*-
"""Run the gates in order and print a one-line summary each (numbers: see the README, "Verification").

    uv run verify/run_all.py --light              # only the gates that need this repository's dependencies and no model weights
    uv run verify/run_all.py                      # + the gates that load the model (needs models/onnx or QEV_MODEL_DIR; ~8 GB RAM)
    uv run verify/run_all.py --oracle --export    # + the gates that need QEV's own package / torch (see below)

Every gate runs with ``uv run`` in the dependency group it needs:
  * runtime gates (this repository's dependencies only): check_prompt.py (+ run_blocked.py), verify.py, check_padding.py
  * ``--oracle`` adds ``--group oracle``  (QEV's own package ``qev`` = veyra, transformers, tokenizers; Python 3.12): check_tokenizer_fuzz.py
    It also needs the pinned QEV checkpoint and Qwen3.5-2B backbone: QEV_HOME=models/qev_home ("Building the ONNX files" in the README).
  * ``--export`` adds ``--group export`` (torch, transformers; Python 3.12): check_vision_inputs.py
``oracle`` and ``export`` cannot be installed together (uv replaces one with the other) and each one re-syncs .venv with torch, so
they are never part of the default run and ``--light`` never installs them.
"""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

# (name, uv run arguments, needs the model weights, group flag that enables it or None)
GATES = [
    ("token ids: golden prompts (exact)", ["verify/check_prompt.py"], False, None),
    ("token ids again with tokenizers / transformers / torch un-importable", ["verify/run_blocked.py", "verify/check_prompt.py"], False, None),
    ("ONNX vs goldens (logits, probabilities, decisions, abstain)", ["verify/verify.py"], True, None),
    ("padding / graph length", ["verify/check_padding.py"], True, None),
    ("token ids: fuzz vs veyra + HF processor", ["--group", "oracle", "verify/check_tokenizer_fuzz.py", "1500"], False, "--oracle"),
    ("vision inputs vs processor / transformers", ["--group", "export", "verify/check_vision_inputs.py"], False, "--export"),
]


def main():
    flags = set(sys.argv[1:])
    light = "--light" in flags
    rc = 0
    for name, args, heavy, flag in GATES:
        if (heavy and light) or (flag and flag not in flags):
            continue
        p = subprocess.run(["uv", "run"] + args, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        lines = [line for line in (p.stdout + p.stderr).splitlines() if line.strip()]
        print("%-70s %s   %s" % (name, "PASS" if p.returncode == 0 else "FAIL", lines[-1] if lines else ""), flush=True)
        rc |= p.returncode
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

[[Japanese](README.md)/[English](README_EN.md)]

# QEV-ONNX-CPU

A minimal-dependency inference implementation of [`ken-jo/qev`](https://huggingface.co/ken-jo/qev) (QEV 0.2.1: typed decisions over text and one image) that runs on onnxruntime and numpy only. GPU inference works too (but it was not faster on the GPU I tested; see below).<br>
It does not use the torch and transformers that the official inference code (PyPI `qev`; the model code is the `veyra` package) requires.

# Features
- Minimal dependencies: the runtime needs only onnxruntime, numpy and pillow (pillow is imported only when a request carries an image)
- Two scripts are enough to run it: `download_model.py` (fetch the model) and `demo_inference_text.py` (inference). Images: `demo_inference_image.py`; camera: `demo_inference_webcam.py`
- Decisions match the reference implementation (QEV's own runtime, PyTorch fp32 on the CPU); see "Verification"
- One question is one forward pass (nothing is generated). Three question types: `noul` (yes/no), `choice` (one of 2 to 16 options), `score` (2 to 16 ordered levels). Every answer carries probabilities, a confidence and an `abstained` flag
- Image input (**one image per request at most**: QEV's rule)
- The preprocessing (Qwen byte-level BPE, prompt building, 3-axis M-RoPE, image patching and the vision-encoder inputs) is ported to pure Python, with cross-checks against the reference
- Runs on the CPU and on a GPU (CUDA or DirectML; `--device cuda`, `--gpu`), with `--require-gpu` so that a silent fall-back to the CPU cannot pass for a GPU run

# Purpose of This Repository
- How far can QEV (Qwen3.5-2B + LoRA + readout / condition / binding heads) be run with three runtime dependencies?
- Can the reference preprocessing be ported to pure Python and build the same inputs?
- Does an image decision with the ONNX vision encoder give the same answer as the original implementation?

# Requirements
```
Python 3.10 or later

numpy        1.26 or later
onnxruntime  1.20 or later
pillow       10.0 or later   (imported only when an image is used)
```
That is all that is needed to run; fetching the model uses the standard library only.<br>
**About 8 GB of free memory or more is needed** (the fp32 weights are 7.5 GB; an image adds the 1.3 GB vision encoder; peak RAM was not measured).<br>
Environments are built with [uv](https://docs.astral.sh/uv/). Optional groups:
- `test`: pytest for `tests/` (no model needed)
- `webcam`: OpenCV for `demo_inference_webcam.py`
- `oracle`: QEV's own package `qev` for the cross-checks against the reference (Python 3.12 only; pulls torch and transformers)
- `export`: torch, transformers etc. for building the ONNX files yourself (never used at run time; cannot share an environment with `oracle`)
- extra `gpu`: `onnxruntime-gpu` (see "Running on a GPU")

# Installation

```bash
git clone https://github.com/Kazuhito00/QEV-ONNX-CPU
cd QEV-ONNX-CPU

# runtime dependencies only (onnxruntime, numpy, pillow)
uv sync
```
`uv sync` also installs this checkout's `qev_decide` package; `uv pip install git+https://github.com/Kazuhito00/QEV-ONNX-CPU` gives `import qev_decide` too (the model files are fetched separately).<br>
**Never install the PyPI package `qev` into another repository's environment** (it has the same import name as OmniJev's `qev`). This implementation does not use `qev`; the `oracle` group is for verification only.

# Download Model
QEV has no published ONNX. The exported ONNX files are meant to be placed in the [Releases](https://github.com/Kazuhito00/QEV-ONNX-CPU/releases) (they are not in the repository; see `release/make_release.py` and "What is in the release" below. **No release has been created yet.**).<br>
This script fetches them into `models/onnx/`, over plain HTTPS from the standard library (no git, no git-lfs):
```bash
uv run download_model.py
```
- **Without arguments it fetches every file of the release (all six graphs L=256, 384, 512, 1024, 1280, 2048, the vision encoder, heads, tokenizer and licence texts: 25 assets, about 9.9 GB)**
- The source is the latest release of `Kazuhito00/QEV-ONNX-CPU` by default (`--repo OWNER/NAME` or the environment variable `QEV_ONNX_REPO` choose another one; priority: argument > environment > default)
- Every file is checked against the SHA-256 in the release's `manifest.json` (a mismatch deletes the file and stops)
- Files split to stay under GitHub's limit (under 2 GiB per file; the shared weights `torso_weights.data`, 7.5 GB, are in 4 parts) are joined after the download and checked (while joining, about 7.5 GB of extra disk is used for a moment)
- Nothing is done for files already present; an interrupted download resumes with the same command

```bash
uv run download_model.py --list                                          # list the files
uv run download_model.py --exclude "vision_encoder*"                     # text only (skips the vision encoder; about 1.3 GB less)
uv run download_model.py --seq-len 384                                   # only the default L=384 graph (the shared weights and vision encoder come along)
uv run download_model.py --tag v0.0.0                                    # a given tag (default: latest)
```

The release holds six graphs: L=256, 384 (default), 512, 1024, 1280 and 2048. **The weights (`torso_weights.data`, 7.5 GB) are shared by all six, so one more graph adds only about 0.16 to 0.18 GB.** `--seq-len 384` therefore still fetches 7.5 GB of weights (it saves only the five other graphs, about 0.85 GB). `manifest.json` lists, for each `.onnx`, the data files it needs, so `--seq-len` and `--exclude` never leave a graph without its weights.

To export the files yourself, see "Building the ONNX files" below (torch is needed; heavy in memory and time).

### What is in the release
`uv run release/make_release.py --dry-run` prints the list (it hashes nothing and writes nothing); `release/make_release.py` builds `dist/release/` from `models/` (it uploads nothing). Because of the limit of under 2 GiB per file, only the shared weights need splitting: 4 parts of at most 1.9 GiB.

| Asset | Size |
|---|---:|
| `onnx__torso_L256.onnx` / `_L384` / `_L512` / `_L1024` / `_L1280` / `_L2048` | 161.2 / 162.6 / 164.0 / 169.4 / 172.2 / 180.4 MB |
| `onnx__torso_weights.data.part000` to `part002` (the shared weights, 7,527,333,888 bytes, split) | 2,040,109,465 bytes each (1.9 GiB) |
| `onnx__torso_weights.data.part003` | 1,407,005,493 bytes |
| `onnx__vision_encoder.onnx` / `onnx__vision_encoder.onnx.data` (one asset, under 2 GiB) | 5.5 MB / 1,325,727,744 bytes |
| `onnx__head.npz` / `onnx__tokenizer.json` | 0.7 MB / 12.8 MB |
| `onnx__qev_config.json` / `onnx__preprocessor_config.json` / `onnx__graph_meta_L*.json` (6 files) | under 1 KB each |
| `onnx__LICENSE-QEV` / `onnx__NOTICE-QEV` / `onnx__LICENSE-QWEN` | 11 KB / 3.5 KB / 11 KB |
| `manifest.json` / `SHA256SUMS` | small |

The total is about 9.9 GB (25 assets + `manifest.json` + `SHA256SUMS`).

# Usage

### Running the demos (Python)
```bash
uv run demo_inference_text.py
uv run demo_inference_image.py
```
The text demo answers a built-in example (a customer-support message: is the customer angry, which team, how urgent). Pass your own input like this:
```bash
uv run demo_inference_text.py --state "My card was charged twice." \
    --choice "team=billing,tech" --noul "angry=The customer is angry" --score "urgency=low,medium,high"
```
The image demo uses `artifacts/images/bars.png` (a bar chart drawn by a script) by default. For your own picture give its path and the questions (**one image only**):
```bash
uv run demo_inference_image.py shot.png --choice "status=paid,pending,failed" --noul "paid=Is it paid?"
```

```python
from qev_decide import Decider, choice, noul, score

decider = Decider("models/onnx")          # default L=384 (change it with seq_len=512 etc.); the environment variable QEV_MODEL_DIR works too

decider.decide("My card was charged twice for the same order.", {
    "team": choice("Which team?", {"billing": "charges, refunds", "tech": "bugs"}),
    "angry": noul("The customer is angry."),
    "urgency": score("How urgent?", ["low", "medium", "high", "critical"]),
})
```
Each answer is QEV's own dict: `probabilities`, `confidence`, `abstained`, and `choice` (the chosen key) / `noul` (probability of yes) / `score` (expected level; `legend` describes the levels).

**One forward pass per question.** By QEV's design the options of a `choice` are **sorted by their descriptions** before the letters (A, B, ...) are assigned, so the letters the model sees are a permutation of the caller's order (the result is returned by key). Two options cannot share a description.

Mix an image into the same request (one picture):
```python
decider.decide("Look at the screenshot.", {...}, images=["shot.png"])
```
`images` takes a file path (up to 20 MiB), bytes or a Pillow image (PNG, JPEG, WebP, GIF; up to 40 MP). QEV's image processing works with 65,536 to 262,144 pixels in 32-px steps, so a small picture is scaled **up** (224x224 becomes 256x256, 64 visual tokens). A transparent PNG simply loses its alpha channel (it is not composited onto a background; same as QEV).

### Is there a person in the camera image?
Grabs frames with OpenCV, asks a yes/no question about the newest frame and draws the answer on the live image.
```bash
uv run --group webcam demo_inference_webcam.py                        # camera 0
uv run --group webcam demo_inference_webcam.py --camera 1 --mirror
uv run --group webcam demo_inference_webcam.py --question "Is the person wearing glasses?"
uv run --group webcam demo_inference_webcam.py --camera artifacts/images/bars.png   # an image or video file instead of a camera
```
- Decisions run in their own thread, so the live image never stalls; the banner at the top (green = yes, red = no) and the probability of yes change whenever a new decision is ready
- Change the question with `--question` (a yes/no question). The picture given to the model is shrunk to 224 px on the long side (`--size`)
- Press `q` or `Esc` to quit. OpenCV is a dependency of this demo only
- It has not been tried on a real camera (unconfirmed). What was checked in this repository is giving it an image file with `--no-window --max-decisions 1` and getting one decision

### Question types and limits
| Type | Built with | Answer |
|---|---|---|
| `choice` | `choice(instructions, {key: description})` (2 to 16 options) | one option (`choice`, `confidence`, `probabilities`) |
| `score` | `score(instructions, [lowest, ..., highest])` (2 to 16 levels) | expected level (`score`, `legend`, `confidence`); the first is the lowest |
| `noul` | `noul("statement")` | whether the statement holds (`noul` = probability of yes); the `true` / `false` descriptions can be replaced |

- A request holds 1 to 4 questions, **one image at most**, and at most **2,048 tokens** in total over all questions (more raises `ValueError`; shorten the input on the caller's side)
- `state` is a string (up to 100,000 characters); with an image it may be empty
- The probabilities go through QEV's calibration (a temperature and an abstain threshold per question type). The original `manifest.json` describes this calibration as "inherited parent policy; development only until refit" (see also "License")

### Running on a GPU (optional)
The default is the CPU. To use a GPU (CUDA) install `onnxruntime-gpu` instead of `onnxruntime`. **They write into the same `onnxruntime/` folder and cannot be installed together.** With the extra `gpu`, swap after installing:
```bash
uv sync --extra gpu                                              # also installs onnxruntime-gpu (next to the CPU build)
uv pip install --force-reinstall --no-deps onnxruntime-gpu       # overwrite the folder with the GPU build so they agree
uv run --no-sync demo_inference_text.py --device cuda            # from now on run with --no-sync (a plain uv run puts the CPU build back)
```
- **Requirements**: put the `bin` folders of CUDA 13 and cuDNN 9 on PATH (`onnxruntime-gpu` ships no CUDA DLLs and looks them up on PATH; open a new terminal afterwards). Windows example (adjust the versions):
  ```text
  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64
  C:\Program Files\NVIDIA\CUDNN\v9.20\bin\13.2\x64
  ```
  DirectML (Windows, no CUDA needed) works by installing `onnxruntime-directml` and using `--device auto` (not tried in this repository)
- **Choosing**: `--device cpu` (default) / `--device cuda` (CUDA, else CPU) / `--device auto` (CUDA or DirectML if the build has it, else CPU). `--gpu` is the same as `--device auto`. Available in `demo_inference_text.py`, `demo_inference_image.py`, `demo_inference_webcam.py` and `verify/bench.py`
- **No silent CPU fall-back**: the provider that was really used is printed (the last line of the demos, e.g. `provider: CUDAExecutionProvider`, what the most recently used graph's session actually got; and the output of `verify/bench.py`). If you asked for a GPU but a graph could not be loaded on it (missing DLLs, not enough VRAM, ...) a warning with the reason is printed and the CPU is used. **With `--require-gpu` it is an error instead of a warning** (a build without a CUDA provider stops before the model is read, for `--device cuda` as well as `--device auto`)
```python
from qev_decide import Decider, select_providers

decider = Decider("models/onnx", providers=select_providers("cuda"), require_gpu=True)
print(decider.provider)   # the provider really in use
```
- **VRAM**: the fp32 weights of the torso graph alone are about 7.5 GB, the vision encoder adds 1.3 GB. They do not fit a 4 GB GPU. This is an optional feature for GPUs with enough VRAM (speed on such a GPU is untested)
- Probabilities on the GPU do not match the CPU exactly (operation order; in the measurement below the largest difference was 3.0e-04). The verification against the reference was done on the CPU only

**It cannot be said that a GPU makes this faster.** My measurements:
- Environment: RTX 3050 Ti Laptop 4 GB, CUDA 13.4, cuDNN 9.20, onnxruntime-gpu 1.31, 224-px images, CPU with 8 threads (a measurement made during the work; `eval_common/gpu_bench`, median of 3 questions x up to 10 rounds)
- A CUDA session could be created (all graphs on `CUDAExecutionProvider`; peak VRAM stayed at 3,838 to 3,892 MiB), but one question took, for text: 5.4 s on CUDA against 4.5 s on the CPU; for an image: 5.9 s against 5.0 s. It is **slower than the CPU** (about 0.8x); there is no GPU benefit. The decisions agree with the CPU (3/3, probability difference at most 3.0e-04)
- The weights are about 7.5 GB and do not fit in 4 GB of VRAM, yet the CUDA session was created without an exception or a warning. The reason is untested (the guess during the work was that the Windows driver spills what exceeds dedicated GPU memory into shared system memory; it was not isolated). "The CUDA session was created" does not mean "it is faster"
- The README of the working source explains the slowness by the roughly 92,000 small nodes (launch-bound); this was not verified in this repository
- For comparison: on the same GPU, models such as JEVision and OneJev 0.8B were measured at more than 10x the CPU speed (they are unrelated to this repository's model), whereas 2B and 4B-class models whose weights exceed the VRAM were not faster

# The sequence length is fixed (important)
The model's linear attention loops over 64-token chunks in Python, which the exporter **unrolls into a fixed number of steps**. A graph with a dynamic sequence length cannot be made.

So the graphs are exported at a fixed length L (a multiple of 64) and the prompt is **right-padded** to L at run time. Every read position precedes the padding and the model is causal, so the result does not change (`verify/check_padding.py`). **Cost scales with L, not with the real token count.**

The release holds L=256, 384, 512, 1024, 1280 and 2048; **the default is L=384** (`Decider(..., seq_len=384)`; `--seq-len` in the demos and `verify/bench.py`; use 256 for very short prompts).

- The smallest graph with length `seq_len` or more is loaded first
- When a request does not fit, the runtime switches to **the smallest bucket that fits** (the old graph is released before the new one is loaded, so two 7.5 GB torso graphs are never in memory together). **Every switch costs a graph load (45 to 75 s in the records of the work)**, so sort requests of different lengths by length
- Above the largest L=2048 a `ValueError` is raised (QEV's own limit is also 2,048 tokens for the whole request)
- All six graphs **share one weights file** (`torso_weights.data`; they read byte-identical tensors from the same place, so the numbers do not change; the exporter compared every tensor byte for byte). **Do not delete `torso_weights.data`**
- An image lengthens the prompt (about 190 to 230 tokens for one 224-px image)

**Speed (only measurements that have a source):**

| Measurement | Value | Source |
|---|---|---|
| CPU (8 threads), median of one image question (Wikipedia image benchmark, Commons 1268 questions, calibrated blocks only) | 4.7 s | table "time per question" of `wikipedia-vision-decision-bench` (a measurement made before the move) |
| CPU (8 threads), median of one question (text / image, on the RTX 3050 Ti Laptop PC; the CPU side of the GPU comparison) | 4.5 s / 5.0 s | `eval_common/gpu_bench` (made during the work) |

Per-bucket times (by L) are not listed: there is no measurement yet on a quiet, calibrated machine. Peak RAM was not measured either. The first call, which reads the weights from disk, is especially slow (in this repository's demo the first question, including the graph load, took 96 s; this is not a speed measurement). To measure, stop other heavy work and use `uv run --no-sync verify/bench.py --seq-len N`.

### Accuracy
No new accuracy was measured in this repository. The following is quoted from another measurement: accuracy (%) of this ONNX implementation on the Wikipedia image benchmark (`wikipedia-vision-decision-bench`; 8 threads, 224 px):

| Commons en | Commons ja | synthetic en | synthetic ja | reasoning en | reasoning ja |
|---|---|---|---|---|---|
| 96.7 | 93.8 | 94.0 | 91.1 | 52.4 (212 questions) | 50.9 (212 questions) |

These are the image questions (Commons 1268, synthetic 382, reasoning 225) and a measurement made before the move. QEV takes one image only, so the 13 reasoning questions with two images are left out (212). The reasoning set includes formats with few options where a guess is often right: read it with the chance level in mind. The original model card ([`ken-jo/qev`](https://huggingface.co/ken-jo/qev)) also reports its own evaluations (values for the PyTorch implementation).

# Verification
```bash
uv run verify/run_all.py --light                         # only the gates that need this environment's dependencies and no model weights (check_prompt and run_blocked)
uv run verify/run_all.py                                 # + the gates that load the model (verify, check_padding; about 8 GB of RAM or more)
uv run verify/run_all.py --oracle --export               # + the gates that need QEV's own package / torch (heavy; see the table)
uv run --group test pytest                               # unit tests (no model needed)
```
`verify/run_all.py` runs each gate with `uv run` in the dependency group it needs. `oracle` and `export` are heavy groups that install torch and swap the environment, so they run only with `--oracle` / `--export`; `--light` never installs them (and they cannot be installed together).

| Gate | What it checks | Needs |
|---|---|---|
| check_prompt (and run_blocked) | prompt text, `input_ids` and the condition position equal the reference (QEV's own runtime) exactly | `models/onnx/tokenizer.json` |
| verify | decisions, probabilities, abstain flags and hidden states equal the reference (fp32), text and image | the model |
| check_padding | the amount of right padding does not move the hidden states at the read positions (checked on every graph) | the model |
| check_vision_inputs | image pixels and position ids equal QEV's image processing (transformers) | `--group export` (torch, transformers) |
| check_tokenizer_fuzz | fuzzes the prompt builder against veyra and the HF processor | `--group oracle` and `QEV_HOME` |
| check_bpe_fuzz / check_unicode_classes | BPE against transformers / tokenizers on 150k texts / every code point | `--group oracle` and `QEV_HOME` |
| run_blocked | runs a gate with `tokenizers`, `transformers` and `torch` made un-importable (the runtime does not need them) | none |

`artifacts/golden.json`, `golden_hidden.npz`, `golden_image.json` and `golden_image.npz` are ground truth made by actually running the reference (QEV's own runtime, PyTorch fp32, CPU). Both sides run the same weights, so any difference is a porting bug. The three pictures in `artifacts/images/` (`red_circle.png`, `green_triangle.png`, `bars.png`) are drawn by a script. **The golden data also contains six questions on real Wikipedia-derived photographs (`img-wiki*`); those pictures are not in this repository.** `verify/verify.py` skips the questions whose picture is missing (`skipped`).

Re-run in this repository with `uv run verify/verify.py --threads 6` (2026-10-11, CPU, fp32): on the **22 questions** whose pictures are present (19 text questions and the 3 questions on script-drawn pictures) there were **0 token-id mismatches, 0 decision flips and 0 abstain flips**. The largest differences were hidden states 1.14e-04, condition logits 2.92e-05, option logits 6.94e-05, probabilities 1.46e-06 and vision-encoder output 1.50e-04. `check_prompt` matches 28/28 exactly (the questions on real photographs can be checked too, since token ids are built from the text alone). In the records of the work (the original working tree, 28 questions including the 6 real-photo ones) decision and abstain flips were also 0, and the largest differences came from the real-photo questions (hidden states 1.54e-01, probability 4.84e-03, vision-encoder output 3.43e-01; the image preprocessing differs from QEV's torch version by 1 to 2 gray levels on some pixels). `check_padding`, `check_vision_inputs`, `check_tokenizer_fuzz`, `check_bpe_fuzz` and `check_unicode_classes` were not re-run in this repository (they need torch / transformers from the `oracle` / `export` groups and the original checkpoint; they passed in the original working tree). The unit tests in `tests/` pass in this repository.

# Building the ONNX files
How to export them yourself (torch is needed; heavy in memory and time. The original checkpoints are fetched by QEV's `qev download`, with pinned revisions and SHA-256 checks):
```bash
uv sync --group oracle                                          # QEV's own package (Python 3.12)
QEV_HOME=models/qev_home uv run --group oracle qev download     # the pinned checkpoint and Qwen3.5-2B (about 4.6 GB)
QEV_HOME=models/qev_home uv run --group oracle export/make_golden.py        # golden data (made with QEV's own runtime)
QEV_HOME=models/qev_home uv run --group oracle export/make_golden_image.py
bash export/export_all.sh 256 384 512 1024 1280 2048            # the fixed-length graphs (one at a time; --group export; heavy)
uv run --group export export/share_weights.py --apply           # make them read ONE weights file (byte-identical tensors only)
uv run --group export export/export_vision.py                   # the vision encoder
```
`export_onnx.py` also writes `models/onnx/head.npz`, `tokenizer.json`, `preprocessor_config.json` and `graph_meta_L<N>.json`. `oracle` and `export` cannot be installed together (uv swaps one for the other). The scripts in `export/` were not re-run after they were put into the uv-group layout used here (this is the work's procedure, tidied up).

# Limitations
- **One image per request** at most, 1 to 4 questions, 2 to 16 options for `choice` / `score`, and 2,048 tokens in total (QEV's rules)
- The sequence length is fixed. The default is L=384; a request that does not fit switches to a larger bucket, which takes about a minute to load
- fp32 only (no fp16 was built). No video
- It needs a lot of memory (7.5 GB for the weights alone; about 8 GB of free memory at the very least)
- Image preprocessing differs from QEV's (torch) image processing by 1 to 2 gray levels on some pixels (Pillow's bicubic). This is why the vision-encoder output and hidden states differ more on real photographs (the decisions agree)
- QEV's `manifest.json` describes the calibration as "inherited parent policy; development only until refit" (`release_allowed: false`). This implementation uses those values as they are
- On a GPU the probabilities do not match the CPU exactly, and a 4 GB GPU is not faster

# Project Structure

```text
README.md                # README (Japanese)
README_EN.md             # README (English)
LICENSE                  # Apache-2.0
pyproject.toml           # dependencies (runtime: onnxruntime, numpy, pillow; groups: test / webcam / oracle / export; extra: gpu)
uv.lock                  # pinned dependencies
download_model.py        # model download (GitHub Releases, SHA-256 check, joins split files)
demo_inference_text.py   # text inference demo
demo_inference_image.py  # image inference demo (one picture)
demo_inference_webcam.py # "is there a person?" on a camera image (OpenCV)
qev_decide/              # the inference implementation (light dependencies; never imports export/)
  runtime.py             #   ORT session + right padding + read-out + bucket switching
  _ort.py                #   session creation, CPU / GPU choice (--device / --gpu / --require-gpu)
  tokenizer.py           #   pure-Python Qwen byte-level BPE
  protocol.py            #   request validation, prompt building, heads, temperature, abstention
  image.py               #   Pillow decode + smart_resize + patching + vision-encoder inputs + 3-axis M-RoPE
  _regex_translate.py    #   rewrites Rust-dialect regexes for Python's re
  _uniprops.py           #   generated: Unicode range tables
export/                  # heavy dependencies (torch / transformers / onnx / qev)
  export_onnx.py         #   the torso (LoRA folded in) to fixed-length ONNX
  export_vision.py       #   the vision encoder to ONNX (vision_export_patch.py swaps in a single-image version)
  share_weights.py       #   share one weights file between buckets (byte-identical tensors only)
  export_all.sh          #   exports several lengths one after another
  make_golden.py         #   makes artifacts/golden.json (runs cases.py's questions through QEV's own runtime)
  make_golden_image.py   #   makes artifacts/golden_image.*
  gen_uniprops.py        #   makes _uniprops.py
release/
  make_release.py        # builds the release files from models/ (splitting, SHA-256, manifest.json); --dry-run only lists
  notes.md               # the release text
  licenses/              # licence texts shipped with the release (QEV, QEV's NOTICE, Qwen)
tests/                   # unit tests (no model; includes the provider-selection tests)
verify/                  # verification gates (all at once with run_all.py), bench.py (time of one question)
artifacts/               # golden* (ground truth), images/ (pictures for the demos and tests)
models/                  # ONNX, tokenizer, settings (gitignored)
```

# Reference
- [ken-jo/qev](https://huggingface.co/ken-jo/qev) (the original model: LoRA adapter and heads; Apache-2.0), [PyPI: qev 0.2.1](https://pypi.org/project/qev/) (the original inference code; the model code is the `veyra` package; Apache-2.0)
- [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) (the base of the original model; Alibaba Cloud; Apache-2.0; revision 15852e8c)

# License
QEV-ONNX-CPU is under the [Apache-2.0 license](LICENSE).<br>
The model weights follow the licence of the original model ([ken-jo/qev](https://huggingface.co/ken-jo/qev), Apache-2.0 on its model card; its base Qwen3.5-2B is Apache-2.0 too). The release ships QEV's licence text and NOTICE (as they came with the PyPI package `qev` 0.2.1) and Qwen's licence text unchanged (`release/licenses/`).<br>
The code in `qev_decide/` is a numpy port of QEV's inference code (Apache-2.0) and of Hugging Face Transformers' (Apache-2.0) Qwen2-VL image processing and Qwen3.5 vision-position computation.

**Please check before redistributing.** QEV's NOTICE lists the datasets it was trained on, among them:
- SNLI (CC-BY-SA-4.0): the NOTICE says that decision data derived from selected premise/hypothesis pairs "retains the applicable share-alike requirements". Whether that condition reaches the trained weights (and this converted form) is not judged in this repository
- CIFAR-10: used only as an image transfer evaluation guard; no images are redistributed. The NOTICE says the dataset card marks the licence as unknown and that this "does not establish permission for redistribution"
- BANKING77 (CC-BY-4.0), LocalLLaMA/typed-decisions (Apache-2.0), Beans and TrashNet (MIT), and others

The NOTICE shipped here refers to licence texts of those datasets under `data-licenses/` (they are in the original Hugging Face repository); they are not included in this release.

Also, QEV's checkpoint `manifest.json` says `"release_allowed": false` (the calibration is "development only until refit"), while the Hugging Face model card gives Apache-2.0 for code and adaptation (as of 2026-10-11). How these fit together (what `release_allowed` means) could not be confirmed.

## About the converted files
The ONNX files in the release are a conversion, made with this repository's `export/`, of [QEV](https://huggingface.co/ken-jo/qev) (Apache-2.0; a repository of a LoRA adapter and heads only), for which no ONNX is published. The modifications are:
- QEV's LoRA adapter (rank 8, all 24 layers; scale alpha / r = 2) is folded into the weights of the base model [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) (Alibaba Cloud; Apache-2.0) and the text decoder (torso) is exported as fp32 ONNX graphs. The original repository does not contain the base weights
- The graphs are split by sequence length and share one weights file
- The base model's vision encoder (Qwen3.5-2B's, weights unmodified) is exported as a single-image ONNX
- The readout, condition and binding heads (LayerNorm, projections and expert weights) are converted to `head.npz` (fp32) and computed in numpy. The temperatures and abstain thresholds are copied to `qev_config.json`

The weights are exported as fp32, and the decisions were checked against the original implementation (PyTorch fp32); see "Verification". The providers of the original model do not guarantee this converted form or its behaviour.

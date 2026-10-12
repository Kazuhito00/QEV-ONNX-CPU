ONNX files for [QEV-ONNX-CPU](https://github.com/Kazuhito00/QEV-ONNX-CPU): QEV 0.2.1 (`ken-jo/qev`, Apache-2.0; typed decisions over text and one image) on onnxruntime + numpy.

Fetch them with `uv run download_model.py` (SHA-256 checked against `manifest.json`; split files are joined). Do not download the assets by hand: the shared weights `onnx__torso_weights.data` are in 4 parts.

These are converted files (see "About the converted files" in the README): QEV's LoRA adapter (r=8, all 24 layers) is folded into the weights of Qwen/Qwen3.5-2B (Apache-2.0, revision 15852e8c) and the decoder is exported as fp32 ONNX graphs of six fixed sequence lengths (L=256, 384, 512, 1024, 1280, 2048; they share one weights file); the Qwen3.5 vision encoder is exported unchanged as a second graph; QEV's readout / condition / binding heads are `head.npz`. Decisions match QEV's own PyTorch runtime on the golden questions (see the README, "Verification").

Licences: `onnx__LICENSE-QEV` and `onnx__NOTICE-QEV` (QEV, Apache-2.0, from the PyPI package `qev` 0.2.1) and `onnx__LICENSE-QWEN` (Qwen3.5-2B, Apache-2.0), unchanged. The NOTICE lists the datasets QEV was trained on, among them SNLI (CC-BY-SA-4.0): read it before redistributing.

| Asset | Content |
|---|---|
| `onnx__torso_L{256,384,512,1024,1280,2048}.onnx` | decoder graphs, one per fixed sequence length (161-180 MB each) |
| `onnx__torso_weights.data.part000..003` | the shared fp32 weights (7,527,333,888 bytes), split to stay under 2 GiB |
| `onnx__vision_encoder.onnx`, `onnx__vision_encoder.onnx.data` | vision encoder (1.3 GB of weights) |
| `onnx__head.npz`, `onnx__qev_config.json`, `onnx__tokenizer.json`, `onnx__preprocessor_config.json`, `onnx__graph_meta_L*.json` | heads, calibration, tokenizer and settings |
| `onnx__LICENSE-QEV`, `onnx__NOTICE-QEV`, `onnx__LICENSE-QWEN` | licence texts |
| `manifest.json`, `SHA256SUMS` | file list with sizes and SHA-256, and how the parts are joined |

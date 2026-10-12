#!/bin/bash
# Export the torso graphs at several fixed lengths, one after the other (each export needs a lot of RAM).
#   bash export/export_all.sh 256 384 512 1024 1280 2048
# Afterwards make all graphs read ONE weights file (checked byte for byte; the per-graph .data files can then be deleted):
#   uv run --group export export/share_weights.py --apply
cd "$(dirname "$0")/.."
mkdir -p models
for L in "$@"; do
  PYTHONIOENCODING=utf-8 uv run --group export export/export_onnx.py --seq-len "$L" > "models/export_L$L.log" 2>&1
  echo "L=$L exit $?" >> models/export_all.log
done
echo "done" >> models/export_all.log

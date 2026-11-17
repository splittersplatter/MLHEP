#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(dirname "$PWD")/.linux-deps"
export MPLBACKEND=Agg
PY=/home/dhruv/Desktop/projects/NNQS-AIMD/.venv/bin/python
"$PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print("GPU:", torch.cuda.get_device_name(0))'
"$PY" -m unittest test_study
"$PY" phase1.py
"$PY" study.py --lambdas 0.003
"$PY" report_study.py
"$PY" tune_disco.py
"$PY" confirm_disco.py --check-only
"$PY" confirm_disco.py

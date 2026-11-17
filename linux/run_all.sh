#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(dirname "$PWD")/.linux-deps"
export MPLBACKEND=Agg
PY=/home/dhruv/Desktop/projects/NNQS-AIMD/.venv/bin/python
run() { printf '\n[%s] %s\n' "$(date -Is)" "$*"; "$@"; }
run "$PY" -c 'import torch; print("PyTorch", torch.__version__, "CUDA", torch.version.cuda, "GPU", torch.cuda.get_device_name(0)); assert torch.cuda.is_available()'
run "$PY" -m unittest test_study
run "$PY" phase1.py --events 2000 --epochs 2 --output results_phase1_smoke
run "$PY" study.py --events 2000 --epochs 2 --seeds 11 --fractions 0 0.005 0.01 --output results_study_smoke
run "$PY" report_study.py --output results_study_smoke
run "$PY" phase1.py
run "$PY" study.py
run "$PY" report_study.py
run "$PY" tune_disco.py
run "$PY" confirm_disco.py --check-only
run "$PY" confirm_disco.py
printf '\n[%s] All requested new experiment stages complete.\n' "$(date -Is)"

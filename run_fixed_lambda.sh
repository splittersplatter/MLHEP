#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export MPLBACKEND=Agg
"$PYTHON_BIN" -c 'import sys, numpy, torch, sklearn, scipy, h5py, hdf5plugin, fastjet, energyflow; print(sys.executable); print("torch", torch.__version__, "CUDA", torch.cuda.is_available())'
"$PYTHON_BIN" study.py "$@"
# Final evaluation is intentionally a separate, explicitly frozen command.

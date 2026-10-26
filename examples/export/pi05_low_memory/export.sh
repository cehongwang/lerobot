#!/usr/bin/env bash
# The low-memory pi0.5 export: three ExecuTorch + TensorRT programs (prefix, step_table, actions).
#
#   bash examples/export/pi05_low_memory/export.sh <policy dir> <weights dump> <output dir> [extra measure.py args]
#
# Run from the lerobot checkout. Needs PYTHONPATH to hold Torch-TensorRT's py/ from
# pytorch/TensorRT:cehongw/pi05-low-memory-repro (TORCH_TENSORRT_PY below), and the weights dump that
# DUMP_WEIGHTS writes (README.md). The policy folder needs only config.json and the processor JSONs.
set -euo pipefail
POLICY=$1 WEIGHTS=$2 OUT=$3
shift 3
HERE=examples/export/pi05_low_memory
TRT_PY=${TORCH_TENSORRT_PY:?set TORCH_TENSORRT_PY to the py/ folder of the Torch-TensorRT branch}
export PYTHONPATH=src:examples/export:$TRT_PY:$HERE PYTHONUNBUFFERED=1
# The prefix's 8 engines and the step's 1 are written here as they are built, then streamed into the .pte.
ENGINES=${ENGINES:-$OUT.engines}
TASK=${TASK:-pick up the red block}

# How each switch saves memory: docs/pi05_export_memory.md section 2.1 in cehongwang/lerobot_project.
ET_SHIM=0 NOCHECK=1 META_LOAD=1 SAVE_TRACE=1 STEP_TABLE=only FAKE_CACHE_EXAMPLE=1 CPU_COMPILE=1 \
MMAP_WEIGHTS=$WEIGHTS MMAP_IMPL=$TRT_PY/torch_tensorrt/dynamo/tools/weight_mapping.py \
FREE_WEIGHTS=1 OPT_LEVEL=${OPT_LEVEL:-3} PATCH_4783=1 ENGINE_CAP_GB=0.5 CAP_PREFIX=0.5 CAP_STEP_TABLE=0.6 TRIM=1 \
REF_CACHE=$HERE/pi05_ref.npz ENGINE_OUTPUT_DIR=$ENGINES TIMING_CACHE=$OUT.timing WRITE_PROGRAM_BINS=1 \
  python $HERE/measure.py --gpu -1 --tag pi05_low_memory --results $OUT.measure --timeline "$@" -- \
  python $HERE/run_anthony.py $HERE/anth_step_export.py --policy.path=$POLICY "--task=$TASK" --output_dir=$OUT
rm -rf "$ENGINES" "$OUT.timing"   # the programs hold the engines now

#!/bin/bash

pip show papermill >/dev/null 2>&1 || pip install papermill

# ------------------------------------------------------------
# Reduce noisy logs
# ------------------------------------------------------------
export PYTHONWARNINGS="ignore"
export TOKENIZERS_PARALLELISM=false

# install GNU parallel if missing
if ! command -v parallel >/dev/null 2>&1; then
    echo "GNU parallel not found. Installing..."
    sudo apt-get install -y parallel
fi

# ============================================================
# Multimodal runs: MVAE (MoE / PoE fusion) and MLP baseline
# Each job is "notebook:model"
# ============================================================
JOBS=("nb/mvae.ipynb:MoEVAE" "nb/mvae.ipynb:PoEVAE" "nb/multimodal_mlp.ipynb:MLP")
EXPERIMENTS=("SIMPLE")
TASKS=("AROUSAL" "VALENCE" "STIMULUS-LABEL")

run_multimodal_job() {
    NOTEBOOK="${1%%:*}"
    MODEL_NAME="${1##*:}"
    EXPERIMENT="$2"
    TASK="$3"
    MODALITY_NAME="$4"

    TMP_NOTEBOOK=$(mktemp /tmp/papermill_multimodal_XXXXXX.ipynb)

    echo "Running $NOTEBOOK | $EXPERIMENT | $TASK | $MODALITY_NAME | $MODEL_NAME"

    EXPERIMENT="$EXPERIMENT" TASK="$TASK" MODALITY="$MODALITY_NAME" MODEL_NAME="$MODEL_NAME" \
    papermill "$NOTEBOOK" "$TMP_NOTEBOOK" --log-output

    rm -f "$TMP_NOTEBOOK"

    echo "Finished $NOTEBOOK | $EXPERIMENT | $TASK | $MODALITY_NAME | $MODEL_NAME"
}

export -f run_multimodal_job

SKIP_EDA_PPG=true

if [ "$SKIP_EDA_PPG" = "false" ]; then
    parallel --will-cite --bar --jobs 1 --line-buffer --tag \
        run_multimodal_job ::: "${JOBS[@]}" ::: "${EXPERIMENTS[@]}" ::: "${TASKS[@]}" ::: "EDA+PPG"
fi

SKIP_EDA_TEXT=false

if [ "$SKIP_EDA_TEXT" = "false" ]; then
    parallel --will-cite --bar --jobs 1 --line-buffer --tag \
        run_multimodal_job ::: "${JOBS[@]}" ::: "${EXPERIMENTS[@]}" ::: "${TASKS[@]}" ::: "EDA+TEXT"
fi

SKIP_PPG_TEXT=false

if [ "$SKIP_PPG_TEXT" = "false" ]; then
    parallel --will-cite --bar --jobs 1 --line-buffer --tag \
        run_multimodal_job ::: "${JOBS[@]}" ::: "${EXPERIMENTS[@]}" ::: "${TASKS[@]}" ::: "PPG+TEXT"
fi
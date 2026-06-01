#!/usr/bin/env bash
# ask.sh — MobileBERT question answering wrapper
#
# Usage:
#   ./ask.sh                              interactive (paste a passage, then ask)
#   ./ask.sh "question" "context text"   single-shot inline
#   ./ask.sh "question" -f passage.txt   single-shot from file

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/venv39/bin/python3.9"
QA_SCRIPT="$SCRIPT_DIR/bert_qa.py"
MODEL="$SCRIPT_DIR/mobilebert_qa.tflite"
VOCAB="$SCRIPT_DIR/bert_vocab.txt"

# ── check deps ────────────────────────────────────────────────────────────────
if [[ ! -f "$MODEL" || ! -f "$VOCAB" ]]; then
    echo "Downloading MobileBERT model and vocab..."
    curl -fsSL --retry 3 -o "$MODEL" \
      "https://storage.googleapis.com/download.tensorflow.org/models/tflite/bert_qa/mobilebert_float_20191023.tflite"
    curl -fsSL --retry 3 -o "$VOCAB" \
      "https://storage.googleapis.com/download.tensorflow.org/models/tflite/bert_qa/vocab.txt"
    echo "Done."
fi

# ── single-shot mode ──────────────────────────────────────────────────────────
if [[ $# -ge 1 ]]; then
    QUESTION="$1"
    shift

    if [[ "$1" == "-f" ]]; then
        "$PYTHON" "$QA_SCRIPT" \
            --model "$MODEL" --vocab "$VOCAB" \
            --question "$QUESTION" --file "$2"
    else
        "$PYTHON" "$QA_SCRIPT" \
            --model "$MODEL" --vocab "$VOCAB" \
            --question "$QUESTION" --context "$*"
    fi
    exit $?
fi

# ── interactive mode ──────────────────────────────────────────────────────────
exec "$PYTHON" "$QA_SCRIPT" --model "$MODEL" --vocab "$VOCAB"

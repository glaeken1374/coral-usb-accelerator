#!/usr/bin/env bash
# pose.sh — MoveNet single-person pose estimation on Coral Edge TPU
#
# Usage:
#   ./pose.sh                        fully interactive
#   ./pose.sh image <path> [1|2]     image mode, optional model choice
#   ./pose.sh webcam [seconds] [1|2] webcam mode, optional duration + model

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/venv39/bin/python3.9"
MODEL_BASE="https://github.com/google-coral/test_data/raw/master"

BOLD='\033[1m'; CYAN='\033[0;36m'; GREEN='\033[0;32m'
RED='\033[0;31m'; RESET='\033[0m'

declare -a MODELS=(
    "MoveNet Lightning  192×192  (faster,  ~5 ms) |movenet_lightning_edgetpu.tflite|movenet_single_pose_lightning_ptq_edgetpu.tflite"
    "MoveNet Thunder    256×256  (accurate, ~9 ms) |movenet_thunder_edgetpu.tflite|movenet_single_pose_thunder_ptq_edgetpu.tflite"
)

download_if_missing() {
    local url="$1" dest="$2" name="$3"
    if [[ ! -f "$dest" ]]; then
        echo -e "${CYAN}  Downloading $name...${RESET}"
        curl -fsSL --retry 3 -o "$dest" "$url" \
            || { echo -e "${RED}  Download failed${RESET}"; exit 1; }
        echo -e "${GREEN}  Downloaded.${RESET}"
    fi
}

pick_model() {
    local preselect="${1:-}"
    echo -e "${BOLD}Choose a model:${RESET}\n"
    for i in "${!MODELS[@]}"; do
        IFS='|' read -r label _ _ <<< "${MODELS[$i]}"
        printf "  ${CYAN}%d)${RESET} %s\n" $((i + 1)) "$label"
    done
    echo ""
    if [[ -n "$preselect" ]]; then
        CHOICE="$preselect"
    else
        while true; do
            read -p "Model [1-${#MODELS[@]}]: " CHOICE
            [[ "$CHOICE" =~ ^[0-9]+$ ]] && (( CHOICE >= 1 && CHOICE <= ${#MODELS[@]} )) && break
            echo -e "${RED}  Enter 1 or 2${RESET}"
        done
    fi
    IFS='|' read -r LABEL LOCAL_NAME REMOTE_NAME <<< "${MODELS[$((CHOICE - 1))]}"
    MODEL_PATH="$SCRIPT_DIR/$LOCAL_NAME"
    echo -e "\n${GREEN}  ✓ $LABEL${RESET}\n"
    download_if_missing "$MODEL_BASE/$REMOTE_NAME" "$MODEL_PATH" "$LOCAL_NAME"
}

echo -e "\n${BOLD}━━ Coral MoveNet Pose Estimation ━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"

# ── mode selection ────────────────────────────────────────────────────────────
MODE="${1:-}"
if [[ -z "$MODE" ]]; then
    echo -e "${BOLD}Input source:${RESET}\n"
    echo -e "  ${CYAN}1)${RESET} Image file"
    echo -e "  ${CYAN}2)${RESET} Webcam (live)\n"
    while true; do
        read -p "Source [1-2]: " SRC
        case "$SRC" in
            1) MODE="image";  break ;;
            2) MODE="webcam"; break ;;
            *) echo -e "${RED}  Enter 1 or 2${RESET}" ;;
        esac
    done
    echo ""
fi

# ── image mode ────────────────────────────────────────────────────────────────
if [[ "$MODE" == "image" ]]; then
    if [[ $# -ge 2 ]]; then
        IMAGE_PATH="${2/#\~/$HOME}"
    else
        while true; do
            read -e -p "Image path: " IMAGE_PATH
            IMAGE_PATH="${IMAGE_PATH/#\~/$HOME}"
            [[ -f "$IMAGE_PATH" ]] && break
            echo -e "${RED}  File not found: $IMAGE_PATH${RESET}"
        done
    fi
    [[ -f "$IMAGE_PATH" ]] || { echo -e "${RED}File not found: $IMAGE_PATH${RESET}"; exit 1; }
    echo -e "${GREEN}  ✓ $IMAGE_PATH${RESET}\n"

    pick_model "${3:-}"

    BASE="${IMAGE_PATH%.*}"; EXT="${IMAGE_PATH##*.}"
    OUTPUT="${BASE}_pose.${EXT}"

    echo -e "${BOLD}Running pose estimation...${RESET}\n"
    "$PYTHON" "$SCRIPT_DIR/pose_estimate.py" "$IMAGE_PATH" "$MODEL_PATH" "$OUTPUT"
    echo -e "\n${GREEN}${BOLD}Done.${RESET} Saved to: $OUTPUT"

# ── webcam mode ───────────────────────────────────────────────────────────────
elif [[ "$MODE" == "webcam" ]]; then
    DURATION="${2:-5}"
    pick_model "${3:-}"

    OUTPUT="$SCRIPT_DIR/pose_live.mp4"

    echo -e "${BOLD}Recording ${DURATION}s from webcam...${RESET}\n"
    "$PYTHON" "$SCRIPT_DIR/pose_video.py" "$OUTPUT" "$DURATION" "0.25" "$MODEL_PATH"
    echo -e "\n${GREEN}${BOLD}Done.${RESET} Saved to: $OUTPUT"

else
    echo -e "${RED}Unknown mode: $MODE  (use 'image' or 'webcam')${RESET}"
    exit 1
fi

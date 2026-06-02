#!/usr/bin/env bash
# pose.sh — MoveNet single-person pose estimation on Coral Edge TPU
#
# Usage:
#   ./pose.sh                        interactive
#   ./pose.sh <image>                use default model (Lightning)
#   ./pose.sh <image> <1|2>          skip model prompt

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

echo -e "\n${BOLD}━━ Coral MoveNet Pose Estimation ━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"

# ── image path ────────────────────────────────────────────────────────────────
if [[ $# -ge 1 ]]; then
    IMAGE_PATH="${1/#\~/$HOME}"
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

# ── model choice ──────────────────────────────────────────────────────────────
echo -e "${BOLD}Choose a model:${RESET}\n"
for i in "${!MODELS[@]}"; do
    IFS='|' read -r label _ _ <<< "${MODELS[$i]}"
    printf "  ${CYAN}%d)${RESET} %s\n" $((i + 1)) "$label"
done
echo ""

if [[ $# -ge 2 ]]; then
    CHOICE="$2"
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

# ── download model if needed ──────────────────────────────────────────────────
download_if_missing "$MODEL_BASE/$REMOTE_NAME" "$MODEL_PATH" "$LOCAL_NAME"

# ── run inference ─────────────────────────────────────────────────────────────
BASE="${IMAGE_PATH%.*}"
EXT="${IMAGE_PATH##*.}"
OUTPUT="${BASE}_pose.${EXT}"

echo -e "${BOLD}Running pose estimation...${RESET}\n"
"$PYTHON" "$SCRIPT_DIR/pose_estimate.py" "$IMAGE_PATH" "$MODEL_PATH" "$OUTPUT"

echo -e "\n${GREEN}${BOLD}Done.${RESET} Annotated image saved to: $OUTPUT"

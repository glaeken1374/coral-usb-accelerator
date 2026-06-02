#!/usr/bin/env bash
# edet.sh — EfficientDet Lite object detection on Coral Edge TPU
#
# Usage:
#   ./edet.sh                              fully interactive
#   ./edet.sh <image> [model#] [thresh%]   non-interactive

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/venv39/bin/python3.9"
MODEL_BASE="https://github.com/google-coral/test_data/raw/master"

BOLD='\033[1m'; CYAN='\033[0;36m'; GREEN='\033[0;32m'
RED='\033[0;31m'; YELLOW='\033[1;33m'; RESET='\033[0m'

# name | local filename | remote filename | input size | approx latency
# Note: Lite3x (640×640) is omitted — exceeds USB transfer deadline on Coral USB Accelerator
declare -a MODELS=(
    "Lite0  320×320  fastest, good for real-time   |efficientdet_lite0_320_ptq_edgetpu.tflite|efficientdet_lite0_320_ptq_edgetpu.tflite|320|~105 ms"
    "Lite1  384×384  balanced speed and accuracy   |efficientdet_lite1_384_ptq_edgetpu.tflite|efficientdet_lite1_384_ptq_edgetpu.tflite|384|~145 ms"
    "Lite2  448×448  better for small objects      |efficientdet_lite2_448_ptq_edgetpu.tflite|efficientdet_lite2_448_ptq_edgetpu.tflite|448|~200 ms"
    "Lite3  512×512  high accuracy                 |efficientdet_lite3_512_ptq_edgetpu.tflite|efficientdet_lite3_512_ptq_edgetpu.tflite|512|~280 ms"
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

echo -e "\n${BOLD}━━ Coral EfficientDet Lite Object Detection ━━━━━━━━━━━${RESET}\n"

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

# ── model menu ────────────────────────────────────────────────────────────────
echo -e "${BOLD}Choose a model:${RESET}\n"
for i in "${!MODELS[@]}"; do
    IFS='|' read -r label _ _ _ latency <<< "${MODELS[$i]}"
    printf "  ${CYAN}%d)${RESET} EfficientDet %-45s ${YELLOW}%s${RESET}\n" \
        $((i + 1)) "$label" "$latency"
done
echo ""

if [[ $# -ge 2 ]]; then
    CHOICE="$2"
else
    while true; do
        read -p "Model [1-${#MODELS[@]}]: " CHOICE
        [[ "$CHOICE" =~ ^[0-9]+$ ]] && (( CHOICE >= 1 && CHOICE <= ${#MODELS[@]} )) && break
        echo -e "${RED}  Enter a number between 1 and ${#MODELS[@]}${RESET}"
    done
fi

IFS='|' read -r LABEL LOCAL_NAME REMOTE_NAME SIZE LATENCY <<< "${MODELS[$((CHOICE - 1))]}"
MODEL_PATH="$SCRIPT_DIR/$LOCAL_NAME"
echo -e "\n${GREEN}  ✓ EfficientDet $LABEL${RESET}"

if [[ "$CHOICE" == "4" ]]; then
    echo -e "${YELLOW}  ⚠ Lite3 (512px) is the largest model supported by the Coral USB Accelerator."
    echo -e "    If it aborts with a USB transfer error, use Lite0–Lite2 instead.${RESET}"
fi
echo ""

download_if_missing "$MODEL_BASE/$REMOTE_NAME" "$MODEL_PATH" "$LOCAL_NAME"

# ── confidence threshold ──────────────────────────────────────────────────────
if [[ $# -ge 3 ]]; then
    RAW_THRESH="$3"
else
    echo -e "${BOLD}Confidence threshold:${RESET}\n"
    echo -e "  ${CYAN}1)${RESET} 30%  — catch more objects (may include uncertain detections)"
    echo -e "  ${CYAN}2)${RESET} 40%  — balanced ${YELLOW}(recommended)${RESET}"
    echo -e "  ${CYAN}3)${RESET} 60%  — high confidence only"
    echo -e "  ${CYAN}4)${RESET} Custom\n"
    while true; do
        read -p "Threshold [1-4]: " TC
        case "$TC" in
            1) RAW_THRESH=30; break ;;
            2) RAW_THRESH=40; break ;;
            3) RAW_THRESH=60; break ;;
            4)
                read -p "  Enter threshold (1-99): " RAW_THRESH
                [[ "$RAW_THRESH" =~ ^[0-9]+$ ]] && (( RAW_THRESH >= 1 && RAW_THRESH <= 99 )) && break
                echo -e "${RED}  Invalid value${RESET}" ;;
            *) echo -e "${RED}  Enter 1-4${RESET}" ;;
        esac
    done
fi

THRESHOLD=$(echo "scale=2; $RAW_THRESH / 100" | bc)
echo -e "\n${GREEN}  ✓ Threshold: ${RAW_THRESH}%${RESET}\n"

# ── run inference ─────────────────────────────────────────────────────────────
BASE="${IMAGE_PATH%.*}"; EXT="${IMAGE_PATH##*.}"
OUTPUT="${BASE}_edet.${EXT}"

echo -e "${BOLD}Running detection...${RESET}\n"
if "$PYTHON" "$SCRIPT_DIR/efficientdet.py" \
    "$IMAGE_PATH" "$MODEL_PATH" "$OUTPUT" "$THRESHOLD"; then
    echo -e "\n${GREEN}${BOLD}Done.${RESET} Annotated image saved to: $OUTPUT"
else
    echo -e "\n${RED}Detection failed.${RESET}"
    echo -e "If you saw a USB transfer error, the model is too large for the Coral USB Accelerator."
    echo -e "Try ${CYAN}Lite0${RESET}, ${CYAN}Lite1${RESET}, or ${CYAN}Lite2${RESET} instead."
    exit 1
fi

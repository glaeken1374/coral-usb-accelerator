#!/usr/bin/env bash
# classify.sh — Interactive MobileNet v1/v2 image classification on Coral Edge TPU

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/venv39/bin/python3.9"
MODEL_BASE="https://github.com/google-coral/test_data/raw/master"

# ── colours ───────────────────────────────────────────────────────────────────
BOLD='\033[1m'; CYAN='\033[0;36m'; GREEN='\033[0;32m'
YELLOW='\033[1;33m'; RED='\033[0;31m'; RESET='\033[0m'

# ── model catalogue ───────────────────────────────────────────────────────────
# Each entry: "display name|filename|labels filename"
declare -a MODELS=(
  "MobileNet v1  1.0  224px  (most accurate v1)  |mobilenet_v1_1.0_224_quant_edgetpu.tflite|imagenet_labels.txt"
  "MobileNet v1  0.75 192px                       |mobilenet_v1_0.75_192_quant_edgetpu.tflite|imagenet_labels.txt"
  "MobileNet v1  0.5  160px                       |mobilenet_v1_0.5_160_quant_edgetpu.tflite|imagenet_labels.txt"
  "MobileNet v1  0.25 128px  (fastest)            |mobilenet_v1_0.25_128_quant_edgetpu.tflite|imagenet_labels.txt"
  "MobileNet v2  1.0  224px  (most accurate v2)   |mobilenet_v2_1.0_224_quant_edgetpu.tflite|imagenet_labels.txt"
  "MobileNet v2  iNat Birds  (964 bird species)   |mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite|inat_bird_labels.txt"
  "MobileNet v2  iNat Insects (1,021 species)     |mobilenet_v2_1.0_224_inat_insect_quant_edgetpu.tflite|inat_insect_labels.txt"
  "MobileNet v2  iNat Plants  (2,101 species)     |mobilenet_v2_1.0_224_inat_plant_quant_edgetpu.tflite|inat_plant_labels.txt"
)

# ── helpers ───────────────────────────────────────────────────────────────────
download_if_missing() {
    local url="$1" dest="$2" name="$3"
    if [[ ! -f "$dest" ]]; then
        echo -e "${CYAN}  Downloading $name...${RESET}"
        curl -fsSL --retry 3 -o "$dest" "$url" || {
            echo -e "${RED}  Failed to download $name${RESET}"
            exit 1
        }
        echo -e "${GREEN}  Downloaded.${RESET}"
    fi
}

# ── prompt for image path ─────────────────────────────────────────────────────
echo -e "\n${BOLD}━━ Coral MobileNet Image Classifier ━━━━━━━━━━━━━━━━━━━━${RESET}\n"

while true; do
    read -e -p "Image path: " IMAGE_PATH
    IMAGE_PATH="${IMAGE_PATH/#\~/$HOME}"   # expand ~
    if [[ -f "$IMAGE_PATH" ]]; then
        break
    fi
    echo -e "${RED}  File not found: $IMAGE_PATH${RESET}"
done
echo -e "${GREEN}  ✓ $IMAGE_PATH${RESET}\n"

# ── model menu ────────────────────────────────────────────────────────────────
echo -e "${BOLD}Choose a model:${RESET}\n"
for i in "${!MODELS[@]}"; do
    IFS='|' read -r label _ _ <<< "${MODELS[$i]}"
    printf "  ${CYAN}%d)${RESET} %s\n" $((i + 1)) "$label"
done
echo ""

while true; do
    read -p "Model [1-${#MODELS[@]}]: " CHOICE
    if [[ "$CHOICE" =~ ^[0-9]+$ ]] && (( CHOICE >= 1 && CHOICE <= ${#MODELS[@]} )); then
        break
    fi
    echo -e "${RED}  Enter a number between 1 and ${#MODELS[@]}${RESET}"
done

IFS='|' read -r LABEL MODEL_FILE LABELS_FILE <<< "${MODELS[$((CHOICE - 1))]}"
MODEL_PATH="$SCRIPT_DIR/$MODEL_FILE"
LABELS_PATH="$SCRIPT_DIR/$LABELS_FILE"

echo -e "\n${GREEN}  ✓ $LABEL${RESET}\n"

# ── download model + labels if needed ────────────────────────────────────────
download_if_missing "$MODEL_BASE/$MODEL_FILE"  "$MODEL_PATH"  "$MODEL_FILE"
download_if_missing "$MODEL_BASE/$LABELS_FILE" "$LABELS_PATH" "$LABELS_FILE"

# ── run inference ─────────────────────────────────────────────────────────────
echo -e "${BOLD}Running inference...${RESET}\n"
"$PYTHON" "$SCRIPT_DIR/run_inference.py" "$IMAGE_PATH" "$MODEL_PATH" "$LABELS_PATH"

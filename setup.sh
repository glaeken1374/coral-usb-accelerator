#!/usr/bin/env bash
# setup.sh — Install Google Coral USB Accelerator on Debian trixie (aarch64)
#
# What this does:
#   1. Adds the Google Coral apt repo and installs libedgetpu1-std
#   2. Reloads udev rules for USB access without root
#   3. Downloads pre-compiled Python 3.9 (no compilation needed)
#   4. Creates venv39 (pycoral 2.0 + tflite-runtime 2.5)
#      — works for both classification and object detection
#   5. Downloads default models and labels
#   6. Runs a quick smoke test to confirm the Edge TPU is working
#
# Usage:
#   bash setup.sh           # full install
#   bash setup.sh --verify  # smoke test only (skip install)

set -euo pipefail

# ── colours ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}▸ $*${RESET}"; }
success() { echo -e "${GREEN}✓ $*${RESET}"; }
warn()    { echo -e "${YELLOW}⚠ $*${RESET}"; }
fail()    { echo -e "${RED}✗ $*${RESET}"; exit 1; }
header()  { echo -e "\n${BOLD}━━ $* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/venv39/bin/python3.9"
PIP="$SCRIPT_DIR/venv39/bin/pip"

# ── constants ──────────────────────────────────────────────────────────────────
PY39_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20240107/cpython-3.9.18%2B20240107-aarch64-unknown-linux-gnu-install_only.tar.gz"

MODEL_BASE="https://github.com/google-coral/test_data/raw/master"
MODELS=(
    "mobilenet_v2_1.0_224_quant_edgetpu.tflite"
    "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
)
LABELS=(
    "imagenet_labels.txt"
    "coco_labels.txt"
)

# ── helpers ────────────────────────────────────────────────────────────────────
require_arch() {
    local arch; arch=$(uname -m)
    [[ "$arch" == "aarch64" ]] || fail "This script requires aarch64 (ARM64). Detected: $arch"
}

download_if_missing() {
    local url="$1" dest="$2" label="${3:-$(basename "$2")}"
    if [[ -f "$dest" ]]; then
        info "$label already exists, skipping download"
    else
        info "Downloading $label..."
        curl -fsSL --retry 3 -o "$dest" "$url" || fail "Failed to download $label"
        success "Downloaded $label"
    fi
}

# ── verify-only mode ───────────────────────────────────────────────────────────
run_verify() {
    header "Smoke Test"

    info "Checking USB device..."
    if lsusb | grep -q "18d1:9302"; then
        success "Coral USB Accelerator detected (runtime mode)"
    elif lsusb | grep -q "1a6e:089a"; then
        warn "Coral detected in DFU mode — unplug and replug, then re-run"
        exit 1
    else
        fail "Coral USB Accelerator not found. Is it plugged in?"
    fi

    info "Testing Edge TPU delegate..."
    "$PYTHON" - <<'PYEOF'
import tflite_runtime.interpreter as tflite
d = tflite.load_delegate('libedgetpu.so.1')
print("  delegate loaded:", d)
PYEOF
    success "Edge TPU delegate OK"

    info "Running classification inference on parrot.jpg..."
    "$PYTHON" "$SCRIPT_DIR/run_inference.py" \
        "$SCRIPT_DIR/parrot.jpg" \
        "$SCRIPT_DIR/mobilenet_v2_1.0_224_quant_edgetpu.tflite" \
        "$SCRIPT_DIR/imagenet_labels.txt"
    success "Classification smoke test passed"

    info "Testing detection model..."
    "$PYTHON" - <<PYEOF
import tflite_runtime.interpreter as tflite, numpy as np
delegate = tflite.load_delegate('libedgetpu.so.1')
interp   = tflite.Interpreter(
    '$SCRIPT_DIR/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite',
    experimental_delegates=[delegate])
interp.allocate_tensors()
inp = interp.get_input_details()[0]
interp.set_tensor(inp['index'], np.zeros(inp['shape'], dtype=np.uint8))
interp.invoke()
print("  detection model OK")
PYEOF
    success "Detection smoke test passed"

    echo -e "\n${GREEN}${BOLD}All checks passed — Coral USB Accelerator is ready.${RESET}"
}

[[ "${1:-}" == "--verify" ]] && { run_verify; exit 0; }

# ══════════════════════════════════════════════════════════════════════════════
#  FULL INSTALL
# ══════════════════════════════════════════════════════════════════════════════

require_arch

if ! sudo -n true 2>/dev/null; then
    warn "Some steps require sudo. You may be prompted for your password."
fi

echo -e "${BOLD}"
echo "  ╔══════════════════════════════════════════════╗"
echo "  ║   Google Coral USB Accelerator Setup         ║"
echo "  ║   Debian trixie · aarch64                    ║"
echo "  ╚══════════════════════════════════════════════╝"
echo -e "${RESET}"

# ── step 1: apt repo + libedgetpu ─────────────────────────────────────────────
header "Step 1/4 — Edge TPU Runtime"

KEYRING=/usr/share/keyrings/coral-edgetpu-archive-keyring.gpg
SOURCES=/etc/apt/sources.list.d/coral-edgetpu.list

if [[ ! -f "$KEYRING" ]]; then
    info "Adding Google Coral apt key..."
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | sudo gpg --dearmor -o "$KEYRING"
    success "Key added"
else
    info "Coral apt key already present"
fi

if [[ ! -f "$SOURCES" ]]; then
    info "Adding Coral apt source..."
    echo "deb [signed-by=$KEYRING] https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
        | sudo tee "$SOURCES" > /dev/null
    sudo apt-get update -qq
    success "Apt source added"
else
    info "Coral apt source already present"
fi

if dpkg -l libedgetpu1-std &>/dev/null; then
    info "libedgetpu1-std already installed"
else
    info "Installing libedgetpu1-std..."
    sudo apt-get install -y libedgetpu1-std
    success "libedgetpu1-std installed"
fi

info "Reloading udev rules..."
sudo udevadm control --reload-rules
sudo udevadm trigger
success "udev rules reloaded"

if groups "$USER" | grep -qw plugdev; then
    success "User '$USER' is in plugdev group"
else
    warn "User '$USER' is NOT in plugdev group. Adding now..."
    sudo usermod -aG plugdev "$USER"
    warn "Log out and back in (or reboot) for group membership to take effect."
fi

# ── step 2: python 3.9 ────────────────────────────────────────────────────────
header "Step 2/4 — Python 3.9"

if [[ -x "$SCRIPT_DIR/python39/bin/python3.9" ]]; then
    info "Python 3.9 already installed"
else
    download_if_missing "$PY39_URL" /tmp/cpython39.tar.gz "Python 3.9 standalone"
    info "Extracting Python 3.9..."
    mkdir -p "$SCRIPT_DIR/python39"
    tar -xzf /tmp/cpython39.tar.gz -C "$SCRIPT_DIR/python39" --strip-components=1
    success "Python $("$SCRIPT_DIR/python39/bin/python3.9" --version) installed"
fi

if [[ ! -d "$SCRIPT_DIR/venv39" ]]; then
    info "Creating venv39..."
    "$SCRIPT_DIR/python39/bin/python3.9" -m venv "$SCRIPT_DIR/venv39"
fi

info "Installing pycoral + dependencies into venv39..."
"$PIP" install -q --upgrade pip
"$PIP" install -q \
    --extra-index-url https://google-coral.github.io/py-repo/ \
    pycoral~=2.0 "numpy<2" Pillow opencv-python-headless
success "pycoral 2.0 + tflite-runtime 2.5 installed"

# ── step 3: models + labels ───────────────────────────────────────────────────
header "Step 3/4 — Models & Labels"

for model in "${MODELS[@]}"; do
    download_if_missing "$MODEL_BASE/$model" "$SCRIPT_DIR/$model" "$model"
done
for label in "${LABELS[@]}"; do
    download_if_missing "$MODEL_BASE/$label" "$SCRIPT_DIR/$label" "$label"
done
download_if_missing "$MODEL_BASE/parrot.jpg" "$SCRIPT_DIR/parrot.jpg" "parrot.jpg (test image)"

# ── step 4: smoke test ────────────────────────────────────────────────────────
header "Step 4/4 — Smoke Test"

if ! lsusb | grep -q "18d1:9302\|1a6e:089a"; then
    warn "Coral USB Accelerator not detected — plug it in and run: bash setup.sh --verify"
else
    run_verify
fi

# ── done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Setup complete. Quick reference:${RESET}"
echo ""
echo "  # Image classification (Edge TPU)"
echo "  $SCRIPT_DIR/venv39/bin/python3.9 $SCRIPT_DIR/run_inference.py <image.jpg>"
echo ""
echo "  # Live webcam detection (Edge TPU)"
echo "  $SCRIPT_DIR/venv39/bin/python3.9 $SCRIPT_DIR/detect_video.py [out.mp4] [seconds] [threshold_%]"
echo ""
echo "  # Verify device at any time"
echo "  bash $SCRIPT_DIR/setup.sh --verify"
echo ""

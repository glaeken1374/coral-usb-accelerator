# Google Coral USB Accelerator

Image classification and live object detection using the Google Coral USB
Accelerator on a Raspberry Pi (Debian trixie, aarch64).

## Requirements

- Raspberry Pi running Debian 13 (trixie), aarch64
- Google Coral USB Accelerator plugged in
- `sudo` access

## Setup

```bash
bash setup.sh
```

Takes ~2 minutes (mostly downloads — no compilation). Safe to re-run.

```bash
bash setup.sh --verify   # check device + run smoke test any time
```

## Usage

### Image classification

Identifies what's in a photo using MobileNet v2 (1000 ImageNet classes).

```bash
./venv39/bin/python3.9 run_inference.py <image.jpg>
```

Example output:
```
Inference: 6.1 ms  (avg over 10 runs, Edge TPU)
Top-5 predictions:
  1. German shepherd ...  score=227
  2. malinois          ...  score=28
```

### Live webcam detection

Detects objects in real time from a USB webcam, saves an annotated video.

```bash
./venv39/bin/python3.9 detect_video.py [output.mp4] [seconds] [threshold_%]
```

| Argument | Default | Description |
|---|---|---|
| `output.mp4` | `live_detection.mp4` | Output file |
| `seconds` | `5` | Recording duration |
| `threshold_%` | `40` | Minimum confidence to show a detection |

Examples:
```bash
# 5 seconds, 40% threshold (default)
./venv39/bin/python3.9 detect_video.py

# 10 seconds, stricter 60% threshold
./venv39/bin/python3.9 detect_video.py out.mp4 10 60
```

Example output:
```
Recording 5s from /dev/video0 → live_detection.mp4
Model: SSD MobileNet v2 COCO  |  threshold: 60%

  frame    2   16.9ms  → person 72%
  frame    3   14.5ms  → person 67%
  ...

66 frames  |  avg TPU: 16.6ms (60 fps)
```

## Models

| Model | Task | Inference |
|---|---|---|
| `mobilenet_v2_1.0_224_quant_edgetpu.tflite` | Classification (1000 classes) | ~6 ms |
| `ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite` | Detection (90 COCO classes) | ~17 ms |

Both are downloaded automatically by `setup.sh`. Source: [coral.ai/models](https://coral.ai/models).

## Python environment

Everything runs in `venv39` (Python 3.9 + pycoral 2.0 + tflite-runtime 2.5).
The system Python (3.13) is not compatible with the Edge TPU runtime.

```
venv39/          Python 3.9 virtualenv — use this for all Coral work
python39/        Python 3.9 standalone interpreter (used to build venv39)
```

## Notes

- **Threshold:** lower values catch more objects but increase false positives.
  40% is a good starting point; 60%+ for high-confidence-only results.
- **Detection model compatibility:** SSD models use split execution
  (backbone on Edge TPU, post-processing on CPU). This requires
  tflite-runtime 2.5 — newer TFLite versions segfault with libedgetpu 16.0.
- **USB device IDs:** `1a6e:089a` (DFU/startup), `18d1:9302` (runtime).
  The device switches to runtime mode on first use.

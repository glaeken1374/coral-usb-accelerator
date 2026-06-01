#!/usr/bin/env python3
"""Run image classification inference on the Google Coral USB Accelerator.

Usage:
  python3 run_inference.py <image> [model.tflite] [labels.txt]

Defaults:
  model  = mobilenet_v2_1.0_224_quant_edgetpu.tflite (in script directory)
  labels = imagenet_labels.txt (in script directory)
"""

import sys
import time
import pathlib
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

SCRIPT_DIR = pathlib.Path(__file__).parent

def load_labels(path):
    return pathlib.Path(path).read_text().splitlines()

def preprocess(image_path, height, width):
    img = Image.open(image_path).convert("RGB").resize((width, height))
    return np.expand_dims(np.array(img, dtype=np.uint8), axis=0)

def run(image_path,
        model_path=None,
        labels_path=None,
        top_k=5,
        num_warmup=1,
        num_runs=10):
    model_path  = model_path  or SCRIPT_DIR / "mobilenet_v2_1.0_224_quant_edgetpu.tflite"
    labels_path = labels_path or SCRIPT_DIR / "imagenet_labels.txt"

    delegate = tflite.load_delegate("libedgetpu.so.1")
    interp   = tflite.Interpreter(str(model_path), experimental_delegates=[delegate])
    interp.allocate_tensors()

    inp  = interp.get_input_details()[0]
    outp = interp.get_output_details()[0]
    h, w = inp["shape"][1], inp["shape"][2]

    data = preprocess(image_path, h, w)
    interp.set_tensor(inp["index"], data)

    for _ in range(num_warmup):
        interp.invoke()

    t0 = time.perf_counter()
    for _ in range(num_runs):
        interp.invoke()
    elapsed_ms = (time.perf_counter() - t0) / num_runs * 1000

    scores = interp.get_tensor(outp["index"])[0]
    top    = np.argsort(scores)[::-1][:top_k]
    labels = load_labels(labels_path)

    print(f"Inference: {elapsed_ms:.1f} ms  (avg over {num_runs} runs, Edge TPU)")
    print(f"Top-{top_k} predictions:")
    for rank, idx in enumerate(top, 1):
        label = labels[idx] if idx < len(labels) else f"class {idx}"
        print(f"  {rank}. {label:45s} score={scores[idx]}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    run(
        image_path  = sys.argv[1],
        model_path  = sys.argv[2] if len(sys.argv) > 2 else None,
        labels_path = sys.argv[3] if len(sys.argv) > 3 else None,
    )

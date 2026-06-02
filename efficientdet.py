#!/usr/bin/env python3
"""EfficientDet Lite object detection on the Coral Edge TPU.

Detects up to 25 objects per image across 90 COCO classes.

Usage:
  python3 efficientdet.py <image> [model.tflite] [output.jpg] [threshold]
"""

import sys
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LABELS    = '/home/glaeken/coral/coco_labels.txt'
MODEL     = '/home/glaeken/coral/efficientdet_lite0_320_ptq_edgetpu.tflite'
THRESHOLD = 0.4

PALETTE = [
    '#E6194B', '#3CB44B', '#4363D8', '#F58231', '#911EB4',
    '#42D4F4', '#F032E6', '#BFEF45', '#FABED4', '#469990',
    '#DCBEFF', '#9A6324', '#FFFAC8', '#800000', '#AAFFC3',
]

def run(image_path, model_path=None, output_path=None, threshold=None):
    from pycoral.utils.edgetpu import make_interpreter

    model_path  = model_path  or MODEL
    output_path = output_path or image_path.rsplit('.', 1)[0] + '_det.jpg'
    threshold   = threshold   if threshold is not None else THRESHOLD

    labels = open(LABELS).read().splitlines()

    interp = make_interpreter(model_path)
    interp.allocate_tensors()

    inp = interp.get_input_details()[0]
    h, w = inp['shape'][1], inp['shape'][2]

    orig = Image.open(image_path).convert('RGB')
    ow, oh = orig.size
    data = np.expand_dims(
        np.array(orig.resize((w, h)), dtype=np.uint8), axis=0)

    interp.set_tensor(inp['index'], data)
    interp.invoke()                     # warm-up
    t0 = time.perf_counter()
    for _ in range(5):
        interp.invoke()
    elapsed_ms = (time.perf_counter() - t0) / 5 * 1000

    # Sort by tensor index — output order is consistent across all Lite variants
    # even though the names differ (Lite0-2: :0–:3, Lite3: :31–:34)
    out_tensors = sorted(interp.get_output_details(), key=lambda t: t['index'])
    boxes   = interp.get_tensor(out_tensors[0]['index'])[0]   # [25, 4] ymin xmin ymax xmax
    classes = interp.get_tensor(out_tensors[1]['index'])[0]   # [25]
    scores  = interp.get_tensor(out_tensors[2]['index'])[0]   # [25]
    count   = int(interp.get_tensor(out_tensors[3]['index'])[0])

    draw = ImageDraw.Draw(orig)
    detections = []

    for i in range(count):
        if scores[i] < threshold:
            continue
        ymin, xmin, ymax, xmax = boxes[i]
        x1, y1 = int(xmin * ow), int(ymin * oh)
        x2, y2 = int(xmax * ow), int(ymax * oh)
        cls_id = int(classes[i])
        label  = labels[cls_id] if cls_id < len(labels) else str(cls_id)
        colour = PALETTE[cls_id % len(PALETTE)]
        conf   = scores[i] * 100
        detections.append((label, conf, x1, y1, x2, y2))

        draw.rectangle([x1, y1, x2, y2], outline=colour, width=3)
        text = f'{label} {conf:.0f}%'
        tw   = len(text) * 7
        draw.rectangle([x1, y1 - 20, x1 + tw + 4, y1], fill=colour)
        draw.text((x1 + 3, y1 - 19), text, fill='white')

    orig.save(output_path)

    print(f'Model     : {model_path.split("/")[-1]}')
    print(f'Inference : {elapsed_ms:.1f} ms avg (5 runs, Edge TPU)')
    print(f'Threshold : {threshold*100:.0f}%')
    print(f'Saved     : {output_path}\n')

    if detections:
        print(f'{"Object":<22} {"Conf":>5}   {"Box (x1,y1,x2,y2)"}')
        print('─' * 55)
        for label, conf, x1, y1, x2, y2 in detections:
            print(f'{label:<22} {conf:>4.0f}%   ({x1},{y1}) → ({x2},{y2})')
    else:
        print(f'No detections above {threshold*100:.0f}% confidence.')

    return detections


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    run(
        image_path  = sys.argv[1],
        model_path  = sys.argv[2] if len(sys.argv) > 2 else None,
        output_path = sys.argv[3] if len(sys.argv) > 3 else None,
        threshold   = float(sys.argv[4]) if len(sys.argv) > 4 else None,
    )

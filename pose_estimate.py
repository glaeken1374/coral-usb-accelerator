#!/usr/bin/env python3
"""Single-person pose estimation using MoveNet on the Coral Edge TPU.

Detects 17 body keypoints and draws a skeleton over the image.

Usage:
  python3 pose_estimate.py <image> [model.tflite] [output.jpg]
"""

import sys
import time
import numpy as np
from PIL import Image, ImageDraw

KEYPOINTS = [
    'nose', 'left eye', 'right eye', 'left ear', 'right ear',
    'left shoulder', 'right shoulder', 'left elbow', 'right elbow',
    'left wrist', 'right wrist', 'left hip', 'right hip',
    'left knee', 'right knee', 'left ankle', 'right ankle',
]

# (keypoint_a, keypoint_b, colour)
SKELETON = [
    (0,  1,  '#FFD700'), (0,  2,  '#FFD700'),   # nose → eyes
    (1,  3,  '#FFD700'), (2,  4,  '#FFD700'),   # eyes → ears
    (5,  6,  '#FF6B6B'),                         # shoulder bar
    (5,  7,  '#4ECDC4'), (7,  9,  '#4ECDC4'),   # left arm
    (6,  8,  '#45B7D1'), (8,  10, '#45B7D1'),   # right arm
    (5,  11, '#FF6B6B'), (6,  12, '#FF6B6B'),   # torso sides
    (11, 12, '#FF6B6B'),                         # hip bar
    (11, 13, '#96CEB4'), (13, 15, '#96CEB4'),   # left leg
    (12, 14, '#FFEAA7'), (14, 16, '#FFEAA7'),   # right leg
]

MODEL   = '/home/glaeken/coral/movenet_lightning_edgetpu.tflite'
CONF    = 0.25   # minimum keypoint confidence to draw


def run(image_path, model_path=None, output_path=None):
    import tflite_runtime.interpreter as tflite
    from pycoral.utils.edgetpu import make_interpreter

    model_path  = model_path  or MODEL
    output_path = output_path or image_path.rsplit('.', 1)[0] + '_pose.jpg'

    interp = make_interpreter(model_path)
    interp.allocate_tensors()

    inp = interp.get_input_details()[0]
    h, w = inp['shape'][1], inp['shape'][2]

    orig = Image.open(image_path).convert('RGB')
    ow, oh = orig.size
    img  = orig.resize((w, h))
    data = np.expand_dims(np.array(img, dtype=np.uint8), axis=0)

    interp.set_tensor(inp['index'], data)
    interp.invoke()                     # warm-up
    t0 = time.perf_counter()
    for _ in range(10):
        interp.invoke()
    elapsed_ms = (time.perf_counter() - t0) / 10 * 1000

    # output: [1, 1, 17, 3]  →  [y, x, score] normalised
    kps = interp.get_output_details()[0]
    keypoints = interp.get_tensor(kps['index'])[0][0]   # shape: (17, 3)

    # scale to original image size
    coords = []
    for y_n, x_n, score in keypoints:
        coords.append((int(x_n * ow), int(y_n * oh), float(score)))

    # draw skeleton
    draw = ImageDraw.Draw(orig)
    for a, b, colour in SKELETON:
        xa, ya, sa = coords[a]
        xb, yb, sb = coords[b]
        if sa >= CONF and sb >= CONF:
            draw.line([(xa, ya), (xb, yb)], fill=colour, width=3)

    # draw keypoints
    r = max(4, min(ow, oh) // 100)
    for i, (x, y, score) in enumerate(coords):
        if score < CONF:
            continue
        draw.ellipse([x - r, y - r, x + r, y + r], fill='white', outline='black', width=2)

    orig.save(output_path)

    # print results
    print(f'Inference : {elapsed_ms:.1f} ms avg (10 runs, Edge TPU)')
    print(f'Saved     : {output_path}\n')
    print(f'{"Keypoint":<18} {"x":>5} {"y":>5} {"conf":>6}')
    print('─' * 38)
    for i, (x, y, score) in enumerate(coords):
        flag = '' if score >= CONF else '  (low confidence)'
        print(f'{KEYPOINTS[i]:<18} {x:>5} {y:>5} {score:>6.2f}{flag}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    run(
        image_path  = sys.argv[1],
        model_path  = sys.argv[2] if len(sys.argv) > 2 else None,
        output_path = sys.argv[3] if len(sys.argv) > 3 else None,
    )

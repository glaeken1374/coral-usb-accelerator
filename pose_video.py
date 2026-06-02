#!/usr/bin/env python3
"""Live single-person pose estimation from webcam using MoveNet on Coral Edge TPU.

Usage:
  python3 pose_video.py [output.mp4] [duration_seconds] [confidence_threshold]
"""

import sys
import time
import numpy as np
import cv2
from pycoral.utils.edgetpu import make_interpreter

MODEL   = '/home/glaeken/coral/movenet_lightning_edgetpu.tflite'
OUTPUT  = sys.argv[1] if len(sys.argv) > 1 else '/home/glaeken/coral/pose_live.mp4'
DURATION = int(sys.argv[2])   if len(sys.argv) > 2 else 5
CONF     = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
MODEL    = sys.argv[4]        if len(sys.argv) > 4 else MODEL
CAMERA   = 0

KEYPOINTS = [
    'nose', 'left eye', 'right eye', 'left ear', 'right ear',
    'left shoulder', 'right shoulder', 'left elbow', 'right elbow',
    'left wrist', 'right wrist', 'left hip', 'right hip',
    'left knee', 'right knee', 'left ankle', 'right ankle',
]

# (a, b, BGR colour)
SKELETON = [
    (0,  1,  (0, 215, 255)), (0,  2,  (0, 215, 255)),   # nose → eyes
    (1,  3,  (0, 215, 255)), (2,  4,  (0, 215, 255)),   # eyes → ears
    (5,  6,  (82, 107, 255)),                            # shoulder bar
    (5,  7,  (204, 209, 72)), (7,  9,  (204, 209, 72)), # left arm
    (6,  8,  (209, 183, 69)), (8,  10, (209, 183, 69)), # right arm
    (5,  11, (82, 107, 255)), (6,  12, (82, 107, 255)), # torso
    (11, 12, (82, 107, 255)),                            # hip bar
    (11, 13, (150, 206, 180)), (13, 15, (150, 206, 180)), # left leg
    (12, 14, (100, 230, 255)), (14, 16, (100, 230, 255)), # right leg
]

def draw_pose(frame, coords, conf_threshold):
    h, w = frame.shape[:2]
    for a, b, colour in SKELETON:
        xa, ya, sa = coords[a]
        xb, yb, sb = coords[b]
        if sa >= conf_threshold and sb >= conf_threshold:
            cv2.line(frame, (xa, ya), (xb, yb), colour, 2, cv2.LINE_AA)
    r = max(4, min(w, h) // 120)
    for x, y, score in coords:
        if score >= conf_threshold:
            cv2.circle(frame, (x, y), r, (255, 255, 255), -1)
            cv2.circle(frame, (x, y), r, (0, 0, 0), 1)

def main():
    interp = make_interpreter(MODEL)
    interp.allocate_tensors()

    inp = interp.get_input_details()[0]
    ih, iw = inp['shape'][1], inp['shape'][2]

    cap = cv2.VideoCapture(CAMERA)
    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(OUTPUT, cv2.VideoWriter_fourcc(*'mp4v'), 20, (fw, fh))

    print(f'Recording {DURATION}s from /dev/video{CAMERA} → {OUTPUT}')
    print(f'Model: {MODEL.split("/")[-1]}  |  confidence threshold: {CONF}\n')

    frame_count = 0
    tpu_times   = []
    deadline    = time.time() + DURATION

    while time.time() < deadline:
        ret, frame = cap.read()
        if not ret:
            break

        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        data = np.expand_dims(
            cv2.resize(rgb, (iw, ih)).astype(np.uint8), axis=0)

        interp.set_tensor(inp['index'], data)
        t0 = time.perf_counter()
        interp.invoke()
        tpu_ms = (time.perf_counter() - t0) * 1000
        tpu_times.append(tpu_ms)

        # keypoints: [y, x, score] normalised → pixel coords
        kp_raw = interp.get_output_details()[0]
        keypoints = interp.get_tensor(kp_raw['index'])[0][0]
        coords = [
            (int(x * fw), int(y * fh), float(s))
            for y, x, s in keypoints
        ]

        draw_pose(frame, coords, CONF)

        # HUD
        remaining = max(0, deadline - time.time())
        visible   = [KEYPOINTS[i] for i, (_, _, s) in enumerate(coords) if s >= CONF]
        cv2.putText(frame,
            f'MoveNet  {tpu_ms:.1f}ms  [{remaining:.1f}s left]',
            (8, fh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        writer.write(frame)
        frame_count += 1

        detected = ', '.join(visible) if visible else '(nothing above threshold)'
        print(f'  frame {frame_count:4d}  {tpu_ms:5.1f}ms  → {detected}')

    cap.release()
    writer.release()

    avg_ms = sum(tpu_times) / len(tpu_times) if tpu_times else 0
    print(f'\n{frame_count} frames  |  avg TPU: {avg_ms:.1f}ms ({1000/avg_ms:.0f} fps)')
    print(f'Saved: {OUTPUT}')

if __name__ == '__main__':
    main()

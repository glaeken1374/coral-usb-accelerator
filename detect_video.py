#!/usr/bin/env python3
"""Live object detection on webcam using Coral Edge TPU.

Runs for DURATION seconds, prints detections to console, saves annotated video.
Usage: python3 detect_video.py [output.mp4] [duration_seconds] [threshold_pct]

Examples:
  python3 detect_video.py                          # defaults
  python3 detect_video.py out.mp4 10 60            # 10s, 60% threshold
"""

import sys
import time
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite

MODEL     = '/home/glaeken/coral/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite'
LABELS    = '/home/glaeken/coral/coco_labels.txt'
OUTPUT    = sys.argv[1] if len(sys.argv) > 1 else '/home/glaeken/coral/live_detection.mp4'
DURATION  = int(sys.argv[2])   if len(sys.argv) > 2 else 5
THRESHOLD = int(sys.argv[3])/100 if len(sys.argv) > 3 else 0.4
CAMERA    = 0

COLORS = [
    (57, 255, 20), (255, 50, 50), (50, 50, 255),
    (255, 200, 0), (0, 200, 255), (200, 0, 255),
]

def load_labels(path):
    return open(path).read().splitlines()

def draw_box(frame, label, score, x1, y1, x2, y2, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text  = f'{label} {score*100:.0f}%'
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

def main():
    labels   = load_labels(LABELS)
    delegate = tflite.load_delegate('libedgetpu.so.1')
    interp   = tflite.Interpreter(MODEL, experimental_delegates=[delegate])
    interp.allocate_tensors()

    inp   = interp.get_input_details()[0]
    outs  = interp.get_output_details()
    ih, iw = inp['shape'][1], inp['shape'][2]

    cap = cv2.VideoCapture(CAMERA)
    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        OUTPUT, cv2.VideoWriter_fourcc(*'mp4v'), 20, (fw, fh)
    )

    print(f'Recording {DURATION}s from /dev/video{CAMERA} → {OUTPUT}')
    print(f'Model: SSD MobileNet v2 COCO  |  threshold: {THRESHOLD*100:.0f}%')
    print(f'Usage: python3 detect_video.py [output.mp4] [seconds] [threshold_pct]\n')

    frame_count = 0
    tpu_times   = []
    deadline    = time.time() + DURATION

    while time.time() < deadline:
        ret, frame = cap.read()
        if not ret:
            break

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (iw, ih))
        data   = np.expand_dims(resized, axis=0)

        interp.set_tensor(inp['index'], data)
        t0 = time.perf_counter()
        interp.invoke()
        tpu_ms = (time.perf_counter() - t0) * 1000
        tpu_times.append(tpu_ms)

        boxes   = interp.get_tensor(outs[0]['index'])[0]
        classes = interp.get_tensor(outs[1]['index'])[0]
        scores  = interp.get_tensor(outs[2]['index'])[0]
        count   = int(interp.get_tensor(outs[3]['index'])[0])

        detections = []
        for i in range(count):
            if scores[i] < THRESHOLD:
                continue
            ymin, xmin, ymax, xmax = boxes[i]
            x1 = int(xmin * fw); y1 = int(ymin * fh)
            x2 = int(xmax * fw); y2 = int(ymax * fh)
            cls_id = int(classes[i])
            label  = labels[cls_id] if cls_id < len(labels) else str(cls_id)
            color  = COLORS[cls_id % len(COLORS)]
            detections.append((label, scores[i], x1, y1, x2, y2, color))
            draw_box(frame, label, scores[i], x1, y1, x2, y2, color)

        remaining = max(0, deadline - time.time())
        cv2.putText(frame, f'Edge TPU  {tpu_ms:.1f}ms  [{remaining:.1f}s left]',
                    (8, fh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        writer.write(frame)
        frame_count += 1

        if detections:
            det_str = ', '.join(f'{l} {s*100:.0f}%' for l, s, *_ in detections)
            print(f'  frame {frame_count:4d}  {tpu_ms:5.1f}ms  → {det_str}')

    cap.release()
    writer.release()

    avg_ms  = sum(tpu_times) / len(tpu_times) if tpu_times else 0
    avg_fps = 1000 / avg_ms if avg_ms else 0
    print(f'\n{frame_count} frames  |  avg TPU: {avg_ms:.1f}ms ({avg_fps:.0f} fps)')
    print(f'Saved: {OUTPUT}')

if __name__ == '__main__':
    main()

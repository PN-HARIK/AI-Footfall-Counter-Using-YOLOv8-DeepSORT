
"""
Footfall Counter using YOLOv8 + DeepSORT
----------------------------------------
Author: Harikrishna PN
Description:
This script detects and tracks people entering and exiting a defined region (ROI line)
in a video stream. It uses YOLOv8 for person detection and DeepSORT for tracking.

Features:
- Auto-adapts to any video resolution (ROI line set by RELATIVE_POS)
- Counts entries and exits
- Displays real-time bounding boxes, counts, and FPS
- Logs events (timestamp, frame, ID, direction) into CSV

Usage:
    python footfall_counter.py
"""

import os
import cv2
import csv
import math
import time
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH   = "footfall_yolov8x.pt"  # your weights
VIDEO_PATH   = "input.mp4"             # file path or 0 for webcam
OUT_VIDEO    = "output.mp4"
EVENT_LOG    = "footfall_events.csv"
PERSON_CLS   = 0          # YOLO person class
CONF_THRESH  = 0.5
LINE_X       = 935    # if None, auto at mid-frame on first read
LINE_COLOR   = (0, 170, 255)
TEXT_COLOR   = (0, 255, 255)
BOX_COLOR    = (0, 255, 0)
HYSTERESIS   = 30         # px distance away from line required to "re-arm"
MAX_AGE      = 30         # DeepSORT max_age frames
DISPLAY_WIN  = True       # set False if you run headless

# -------------------------
# HELPERS
# -------------------------
def side_of_line(x, line_x):
    return "L" if x < line_x else "R"

def crosses(prev_x, curr_x, line_x):
    # strict cross check
    return (prev_x < line_x <= curr_x) or (prev_x > line_x >= curr_x)

def distance_to_line(x, line_x):
    return abs(x - line_x)

def safe_fps(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps if fps and not math.isnan(fps) and fps > 1e-3 else 30.0

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# -------------------------
# INIT
# -------------------------
model   = YOLO(MODEL_PATH)
tracker = DeepSort(max_age=MAX_AGE)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video source: {VIDEO_PATH}")

fps    = safe_fps(cap)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if LINE_X is None:
    LINE_X = width // 2  # auto center if not provided

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out    = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (width, height))

# counts and per-track state
entry_count = 0
exit_count  = 0
track_state = {}  # tid -> {"last_x": int, "side": "L"/"R", "armed": bool}

# prepare CSV log
new_file = not os.path.exists(EVENT_LOG)
with open(EVENT_LOG, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    if new_file:
        writer.writerow(["timestamp", "frame", "track_id", "direction"])  # header

frame_idx = 0
t0 = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # YOLO inference
        results = model(frame, verbose=False)

        detections = []  # [ [x,y,w,h], conf, cls_name(optional) ]
        for r in results:
            # r.boxes may be empty; guard
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls = int(box.cls.item())
                conf = float(box.conf.item())
                if cls == PERSON_CLS and conf >= CONF_THRESH:
                    x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().flatten())
                    w, h = x2 - x1, y2 - y1
                    # DeepSortRealtime expects xywh
                    detections.append(([x1, y1, w, h], conf, "person"))

        # Track update
        tracks = tracker.update_tracks(detections, frame=frame)

        # draw line
        cv2.line(frame, (LINE_X, 0), (LINE_X, height - 1), LINE_COLOR, 2)

        # iterate tracks
        for tr in tracks:
            # Only confirmed, recently updated tracks
            if hasattr(tr, "is_confirmed") and not tr.is_confirmed():
                continue
            if hasattr(tr, "time_since_update") and tr.time_since_update > 0:
                continue

            tid  = tr.track_id
            ltrb = tr.to_ltrb()  # (l, t, r, b)
            x1, y1, x2, y2 = map(int, ltrb)
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width - 1))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height - 1))

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # draw box & id
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
            cv2.putText(frame, f"ID:{tid}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, BOX_COLOR, 2)

            # init state if new
            if tid not in track_state:
                track_state[tid] = {
                    "last_x": cx,
                    "side": side_of_line(cx, LINE_X),
                    "armed": distance_to_line(cx, LINE_X) > HYSTERESIS,
                }

            prev_x   = track_state[tid]["last_x"]
            prev_side= track_state[tid]["side"]
            armed    = track_state[tid]["armed"]

            curr_side = side_of_line(cx, LINE_X)
            track_state[tid]["last_x"] = cx
            track_state[tid]["side"]   = curr_side

            # (re)arm when far enough from the line
            if distance_to_line(cx, LINE_X) > HYSTERESIS:
                track_state[tid]["armed"] = True

            # count exactly once per crossing while armed
            if armed and curr_side != prev_side and crosses(prev_x, cx, LINE_X):
                direction = "entry" if prev_x < LINE_X <= cx else "exit"
                if direction == "entry":
                    entry_count += 1
                else:
                    exit_count += 1

                # disarm until the track moves away again
                track_state[tid]["armed"] = False

                # log event
                with open(EVENT_LOG, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([now_str(), frame_idx, tid, direction])

                # flash marker on line at crossing point
                cv2.circle(frame, (LINE_X, cy), 8, (255, 255, 255), -1)

        # HUD
        elapsed = max(1e-6, time.time() - t0)
        proc_fps = frame_idx / elapsed
        cv2.putText(frame, f"Entries: {entry_count}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2)
        cv2.putText(frame, f"Exits:   {exit_count}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2)
        cv2.putText(frame, f"FPS: {proc_fps:.1f}", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2)

        out.write(frame)
        if DISPLAY_WIN:
            cv2.imshow("Footfall Counter (YOLOv8 + DeepSORT)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    cap.release()
    out.release()
    if DISPLAY_WIN:
        cv2.destroyAllWindows()

print(f"Done. Entries={entry_count} Exits={exit_count}")
print(f"Saved video -> {OUT_VIDEO}")
print(f"Event log   -> {EVENT_LOG}")

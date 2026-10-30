"""Conservative practice-activity detector; pose motion, not match scoring.

The near-player preset retains probable swings and shadow swings. It does not
verify ball contact. All thresholds are configuration, not sample timestamps.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
import sys
import time

import av
import numpy as np
from monitor import progress

VERSION = "practice-pose-v1"
LOG = logging.getLogger("tennis")


def merge_intervals(intervals, duration, before=2.0, after=1.5, gap=0.5):
    merged = []
    for a, b in sorted(intervals):
        if not np.isfinite([a, b]).all() or b < a:
            raise ValueError("Invalid detector interval")
        a, b = max(0.0, a - before), min(duration, b + after)
        if b <= a:
            continue
        if merged and a <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def score_poses(records, cfg, width, height):
    """Score wrist motion relative to the torso, filtering out distant people."""
    prev = None
    trace = []
    roi = cfg.get("target_region", [0, 0, 1, 1])
    visible = 0
    for r in records:
        t = r["t"]
        b = np.asarray(r["boxes"])
        k, c = np.asarray(r["k"]), np.asarray(r["kc"])
        candidates = [j for j, box in enumerate(b) if
                      roi[0] <= (box[0] + box[2]) / (2 * width) <= roi[2]
                      and roi[1] <= (box[1] + box[3]) / (2 * height) <= roi[3]
                      and box[3] - box[1] >= height * cfg.get("min_player_height", 0.12)]
        if not candidates:
            trace.append([t, 0, 0, 0])
            prev = None
            continue
        # Prefer the nearest large player; the region can exclude neighboring courts.
        j = max(candidates, key=lambda j: b[j, 3] - b[j, 1])
        kp, kc = k[j], c[j]
        if min(kc[[5, 6, 11, 12]]) < 0.25:
            trace.append([t, 0, 0, 0])
            prev = None
            continue
        visible += 1
        shoulder = (kp[5] + kp[6]) / 2
        hip = (kp[11] + kp[12]) / 2
        torso = max(height * 0.023, float(np.linalg.norm(hip - shoulder)))
        wrists = (kp[[9, 10]] - shoulder) / torso
        valid = kc[[9, 10]] > 0.35
        speed = 0.0
        if prev and 0 < t - prev[0] < 0.4 and np.linalg.norm(shoulder - prev[1]) < (b[j, 3] - b[j, 1]) * 0.5:
            good = valid & prev[3]
            if good.any():
                speed = float(np.linalg.norm(wrists - prev[2], axis=1)[good].max() / (t - prev[0]))
        reach = float(np.linalg.norm(wrists, axis=1)[valid].max()) if valid.any() else 0.0
        upright = float((hip[1] - shoulder[1]) / torso)
        trace.append([t, speed, reach, upright])
        prev = (t, shoulder, wrists, valid)
    a = np.asarray(trace)
    events = []
    for i in range(1, len(a) - 1):
        speed = float(np.median(a[i-1:i+2, 1]))
        reach = float(a[i-1:i+2, 2].max())
        if speed >= cfg.get("swing_speed", 2.0) and reach >= cfg.get("arm_reach", 0.85) and a[i, 3] >= 0.65:
            t = float(a[i, 0])
            if events and t - events[-1][1] <= 1.0:
                events[-1][1] = t
            else:
                events.append([t, t])
    return events, visible / max(1, len(records))


def detect(video, cfg):
    source = Path(cfg["rallyclip_path"])
    sys.path.insert(0, str(source / "src"))
    from extraction.pose_extractor import PoseExtractor
    # PoseExtractor handles CoreML availability and CPU fallback.
    pose = PoseExtractor(model_path=str(source / "models/rallyclip_v0.6.0/yolov8n-pose-544x960-static.onnx"), device=cfg.get("pose_device", "coreml"))
    records = []
    last_log = time.monotonic()
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        duration = float(stream.duration * stream.time_base)
        width, height = stream.width, stream.height
        for frame in container.decode(stream):
            result = pose.model.predict(frame.to_ndarray(format="bgr24"), conf=0.30, imgsz=960)[0]
            records.append({"t": float(frame.time), "boxes": result.boxes.xyxy.numpy(), "k": result.keypoints.xy.numpy(), "kc": result.keypoints.conf.numpy()})
            if time.monotonic() - last_log > 30:
                LOG.info("Analyzing %.0f / %.0f seconds", frame.time, duration)
                progress(cfg, "Analyzing player movement", round(float(frame.time), 1), round(duration, 1), "video seconds")
                last_log = time.monotonic()
    events, visibility = score_poses(records, cfg, width, height)
    intervals = merge_intervals(events, duration, cfg.get("padding_before", 2.0), cfg.get("padding_after", 1.5), cfg.get("merge_gap", 0.5))
    return intervals, {"detector": VERSION, "target_visible_fraction": round(visibility, 3), "motion_events": len(events), "event_intervals": events,
                       "note": "Candidate practice activity based on near-player pose motion. May include shadow swings or walking and miss obscured/slow strokes. Original is preserved."}

"""Bounded worker heuristic diagnostics on the original-recording time axis.

Pose/person presence, track loss, and motion heuristics are non-authoritative
operator review aids. They never verify tennis play, swing truth, player
identity, or benchmark/training labels.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re

import numpy as np

SCHEMA_VERSION = 1
CACHE_SCHEMA_VERSION = 1
GENERATOR_KIND = "worker_heuristic_v1"
TRUTH_STATUS = "heuristic_only_not_play_or_swing_truth"
TRAINING_LABEL_STATUS = "not_released"
TARGET_SELECTION_POLICY = "largest_candidate_in_roi_v1"
TARGET_SELECTION_POLICY_VERSION = "primary-track-v1"
PROXY_TARGET_FPS = 8.0
PROXY_SCALE_FILTER = "scale=w='min(960,iw)':h=-2,fps=8,format=yuv420p"
MAX_HINTS_PER_ITEM = 5
MAX_DIAGNOSTICS_BYTES = 512 * 1024

ALLOWED_HEURISTIC_STATUSES = {
    "no_detector_warnings",
    "check_boundary_activity",
    "extended_low_motion",
    "person_track_interrupted",
    "ambiguous_multiple_persons",
    "unknown",
}

ALLOWED_IDENTITY_STATUSES = {
    "unverified_single_primary_track",
    "ambiguous_multiple_persons",
    "no_person_detected",
    "unknown",
}

ALLOWED_HINT_CODES = {
    "person_track_lost_near_start",
    "person_track_lost_near_end",
    "low_motion_near_start",
    "low_motion_near_end",
    "extended_low_motion",
    "person_track_interrupted",
    "ambiguous_multiple_persons",
    "detector_visibility_degraded",
    "sustained_roi_motion_in_gap",
    "boundary_adjacent_roi_motion_in_gap",
}


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def compact_json_sha256(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def format_ms(ms: int) -> str:
    ms = max(0, int(round(ms)))
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"
    return f"{minutes:02d}:{seconds:02d}.{millis:03d}"


def clip_id_for_bounds(source_sha256: str, start_seconds: float, end_seconds: float) -> str:
    return hashlib.sha256(f"{source_sha256}:{start_seconds:.6f}:{end_seconds:.6f}".encode("utf-8")).hexdigest()[:24]


def _default_proxy_dimensions(meta: dict | None) -> tuple[int, int]:
    width = int((meta or {}).get("width") or 960)
    height = int((meta or {}).get("height") or 540)
    if width <= 0 or height <= 0:
        return 960, 540
    proxy_w = min(960, width)
    proxy_h = max(2, int(round(height * (proxy_w / width) / 2.0)) * 2)
    return proxy_w, proxy_h


def compute_extractor_code_digest(root: Path | None = None) -> tuple[str, dict[str, str]]:
    base = Path(root) if root is not None else Path(__file__).resolve().parent
    module_digests: dict[str, str] = {}
    combined = hashlib.sha256()
    for name in ("pipeline.py", "detector.py", "diagnostics.py"):
        path = base / name
        data = path.read_bytes() if path.is_file() else b""
        module_digests[name] = hashlib.sha256(data).hexdigest()
        combined.update(name.encode("utf-8") + b"\0" + data + b"\0")
    return combined.hexdigest(), module_digests


def resolve_model_sha256(cfg: dict | None) -> str | None:
    cfg = cfg or {}
    if isinstance(cfg.get("model_sha256"), str) and re.fullmatch(r"[a-f0-9]{64}", cfg["model_sha256"]):
        return cfg["model_sha256"]
    rallyclip = cfg.get("rallyclip_path")
    if not rallyclip:
        return None
    model = Path(rallyclip) / "models/rallyclip_v0.6.0/yolov8n-pose-544x960-static.onnx"
    if not model.is_file():
        return None
    h = hashlib.sha256()
    with model.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_extraction_provenance(
    source_sha256: str,
    meta: dict | None,
    cfg: dict | None,
    proxy_meta: dict | None = None,
    model_sha256: str | None = None,
    extractor_code_digest: str | None = None,
    cache_reuse: dict | None = None,
) -> tuple[str, dict]:
    if not isinstance(source_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", source_sha256):
        raise ValueError("Invalid source_sha256 for diagnostics provenance")
    meta = dict(meta or {})
    cfg = dict(cfg or {})
    proxy_meta = dict(proxy_meta or {})
    default_w, default_h = _default_proxy_dimensions(meta)
    proxy_w = int(proxy_meta.get("width") or cfg.get("proxy_width") or default_w)
    proxy_h = int(proxy_meta.get("height") or cfg.get("proxy_height") or default_h)
    target_fps = float(proxy_meta.get("target_fps") or cfg.get("proxy_target_fps") or PROXY_TARGET_FPS)
    actual_fps = float(proxy_meta.get("actual_fps") or target_fps)
    pts_timebase = str(proxy_meta.get("pts_timebase") or cfg.get("proxy_pts_timebase") or "1/1000")
    orig_meta = meta.get("original_metadata") if isinstance(meta.get("original_metadata"), dict) else {}
    rotation = orig_meta.get("rotation_degrees")
    rotation_degrees = int(rotation) if _is_number(rotation) else 0
    transfer = int(meta.get("transfer") or 1)
    color_transfer = str(
        proxy_meta.get("color_transfer")
        or ("tonemapped_bt709" if transfer in (16, 18) else "sdr_bt709")
    )
    crop_box = cfg.get("crop_box")
    if crop_box is not None:
        crop_box = [float(v) for v in crop_box]

    roi = [float(v) for v in cfg.get("target_region", [0, 0, 1, 1])]
    if len(roi) != 4 or not all(_is_number(v) for v in roi):
        raise ValueError("target_region must be four finite numbers")
    min_player_height = float(cfg.get("min_player_height", 0.12))
    swing_speed = float(cfg.get("swing_speed", 2.0))
    arm_reach = float(cfg.get("arm_reach", 0.85))
    policy_version = str(cfg.get("target_selection_policy_version", TARGET_SELECTION_POLICY_VERSION))
    player_assignment_revision = cfg.get("player_assignment_revision")
    if player_assignment_revision is not None:
        player_assignment_revision = str(player_assignment_revision)

    tracking_params = {
        "target_region": roi,
        "min_player_height": min_player_height,
        "swing_speed": swing_speed,
        "arm_reach": arm_reach,
        "upright_min": 0.65,
        "torso_keypoint_min_conf": 0.25,
        "wrist_keypoint_min_conf": 0.35,
        "active_wrist_speed": float(cfg.get("active_wrist_speed", 0.65)),
        "active_body_speed": float(cfg.get("active_body_speed", 0.25)),
        "boundary_warning_min_ms": 3000,
        "extended_low_motion_min_ms": 8000,
        "track_interrupt_min_ms": 3000,
        "ambiguous_multi_person_min_ms": 1500,
        "gap_sustained_motion_min_ms": 1500,
        "competing_height_ratio": 0.70,
        "target_selection_policy": TARGET_SELECTION_POLICY,
        "target_selection_policy_version": policy_version,
        "player_assignment_revision": player_assignment_revision,
    }
    tracking_config_digest = str(cfg.get("tracking_config_digest") or compact_json_sha256(tracking_params))

    computed_code_digest, module_digests = compute_extractor_code_digest()
    code_digest = str(extractor_code_digest or cfg.get("extractor_code_digest") or computed_code_digest)
    resolved_model_sha = model_sha256 if model_sha256 is not None else resolve_model_sha256(cfg)

    proxy_spec = {
        "width": proxy_w,
        "height": proxy_h,
        "target_fps": target_fps,
        "actual_fps": actual_fps,
        "rotation_degrees": rotation_degrees,
        "crop_box": crop_box,
        "color_transfer": color_transfer,
        "pixel_format": "yuv420p",
        "scale_filter": str(proxy_meta.get("scale_filter") or PROXY_SCALE_FILTER),
        "pts_timebase": pts_timebase,
    }
    detector_spec = {
        "detector_version": "practice-pose-v1",
        "model_name": "yolov8n-pose-544x960-static.onnx",
        "model_sha256": resolved_model_sha,
        "input_tensor_shape": [1, 3, 544, 960],
        "imgsz": 960,
        "backend": str(cfg.get("pose_device", "coreml")),
        "predict_conf": 0.30,
        "person_min_score": 0.25,
        "wrist_min_score": 0.35,
    }
    roi_and_filter_config = {
        "target_region": roi,
        "min_player_height": min_player_height,
        "target_selection_policy": TARGET_SELECTION_POLICY,
        "target_selection_policy_version": policy_version,
        "player_assignment_revision": player_assignment_revision,
        "tracking_config_digest": tracking_config_digest,
    }

    cache_tuple = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "source_sha256": source_sha256,
        "proxy_spec": proxy_spec,
        "detector_spec": detector_spec,
        "extractor_code_digest": code_digest,
        "roi_and_filter_config": roi_and_filter_config,
    }
    extraction_cache_key = compact_json_sha256(cache_tuple)

    duration_seconds = float(meta.get("duration") or 0.0)
    duration_ms = max(1, int(round(duration_seconds * 1000))) if duration_seconds > 0 else 0
    provenance = {
        **cache_tuple,
        "extraction_cache_key": extraction_cache_key,
        "timebase": {
            "axis": "original_recording_time",
            "unit": "milliseconds",
            "boundary_convention": "start_inclusive_end_exclusive",
            "pts_timebase": pts_timebase,
            "proxy_target_fps": target_fps,
            "proxy_actual_fps": actual_fps,
            "frame_timestamp_source": "av_frame_time_seconds",
            "source_duration_ms": duration_ms,
            "source_duration_seconds": duration_seconds,
        },
        "software": {
            "pipeline_version": "2.0.0",
            "detector_version": "practice-pose-v1",
            "generator_kind": GENERATOR_KIND,
            "extractor_code_digest": code_digest,
            "module_digests": module_digests,
        },
        "cut_assembly_config": {
            "padding_before": float(cfg.get("padding_before", 2.0)),
            "padding_after": float(cfg.get("padding_after", 1.5)),
            "merge_gap": float(cfg.get("merge_gap", 0.5)),
        },
        "cache_reuse": dict(
            cache_reuse
            or {
                "reused": False,
                "scope": "none",
                "note": "Task stage checkpoints are task-local recovery only, not cross-rerun observation cache reuse.",
            }
        ),
    }
    return extraction_cache_key, provenance


def extract_frame_observations(
    records: list[dict],
    cfg: dict | None,
    width: int,
    height: int,
    events: list[list[float]] | None = None,
) -> list[dict]:
    """Derive bounded per-frame heuristic signals from existing pose records."""
    cfg = cfg or {}
    roi = cfg.get("target_region", [0, 0, 1, 1])
    min_height_px = height * float(cfg.get("min_player_height", 0.12))
    active_wrist_threshold = float(cfg.get("active_wrist_speed", 0.65))
    active_body_threshold = float(cfg.get("active_body_speed", 0.25))
    events = events or []

    prev = None
    raw_frames: list[dict] = []
    for r in records:
        t = float(r["t"])
        boxes_raw = r.get("boxes", [])
        b = np.asarray(boxes_raw, dtype=float).reshape((-1, 4)) if len(boxes_raw) else np.zeros((0, 4), dtype=float)
        k = np.asarray(r.get("k", []), dtype=float)
        c = np.asarray(r.get("kc", []), dtype=float)
        candidates = [
            j
            for j, box in enumerate(b)
            if roi[0] <= (box[0] + box[2]) / (2 * width) <= roi[2]
            and roi[1] <= (box[1] + box[3]) / (2 * height) <= roi[3]
            and (box[3] - box[1]) >= min_height_px
        ]
        if not candidates:
            raw_frames.append(
                {
                    "t": round(t, 6),
                    "person_present": False,
                    "active_motion": False,
                    "ambiguous_multiple": False,
                    "low_confidence_candidate": False,
                    "roi_candidates": 0,
                    "wrist_speed": 0.0,
                    "body_speed": 0.0,
                    "reach": 0.0,
                    "upright": 0.0,
                }
            )
            prev = None
            continue

        valid_candidates = [
            idx
            for idx in candidates
            if idx < len(c) and len(c[idx]) > 12 and float(np.min(c[idx][[5, 6, 11, 12]])) >= 0.25
        ]
        j = max(candidates, key=lambda idx: b[idx, 3] - b[idx, 1])
        primary_height = float(b[j, 3] - b[j, 1])
        primary_center = np.asarray([(b[j, 0] + b[j, 2]) / 2.0, (b[j, 1] + b[j, 3]) / 2.0], dtype=float)

        competing = [
            idx
            for idx in valid_candidates
            if float(b[idx, 3] - b[idx, 1]) >= 0.70 * primary_height
        ]
        ambiguous_multiple = len(competing) >= 2

        if j >= len(c) or len(c[j]) <= 12 or float(np.min(c[j][[5, 6, 11, 12]])) < 0.25:
            raw_frames.append(
                {
                    "t": round(t, 6),
                    "person_present": False,
                    "active_motion": False,
                    "ambiguous_multiple": ambiguous_multiple,
                    "low_confidence_candidate": True,
                    "roi_candidates": len(candidates),
                    "wrist_speed": 0.0,
                    "body_speed": 0.0,
                    "reach": 0.0,
                    "upright": 0.0,
                }
            )
            prev = None
            continue

        kp, kc = k[j], c[j]
        shoulder = (kp[5] + kp[6]) / 2.0
        hip = (kp[11] + kp[12]) / 2.0
        torso = max(height * 0.023, float(np.linalg.norm(hip - shoulder)))
        wrists = (kp[[9, 10]] - shoulder) / torso
        valid = kc[[9, 10]] > 0.35
        wrist_speed = 0.0
        body_speed = 0.0
        if prev is not None and 0 < t - prev[0] < 0.4:
            dt = t - prev[0]
            shoulder_disp = float(np.linalg.norm(shoulder - prev[1]))
            if shoulder_disp < primary_height * 0.5:
                body_speed = shoulder_disp / (max(1.0, primary_height) * dt)
                good = valid & prev[3]
                if good.any():
                    wrist_speed = float(np.linalg.norm(wrists - prev[2], axis=1)[good].max() / dt)
            elif len(valid_candidates) >= 2:
                ambiguous_multiple = True
        reach = float(np.linalg.norm(wrists, axis=1)[valid].max()) if valid.any() else 0.0
        upright = float((hip[1] - shoulder[1]) / torso)
        raw_frames.append(
            {
                "t": round(t, 6),
                "person_present": True,
                "active_motion": False,
                "ambiguous_multiple": ambiguous_multiple,
                "low_confidence_candidate": False,
                "roi_candidates": len(candidates),
                "wrist_speed": round(wrist_speed, 4),
                "body_speed": round(body_speed, 4),
                "reach": round(reach, 4),
                "upright": round(upright, 4),
            }
        )
        prev = (t, shoulder, wrists, valid, primary_center)

    n = len(raw_frames)
    speeds = [f["wrist_speed"] for f in raw_frames]
    for i, frame in enumerate(raw_frames):
        if not frame["person_present"]:
            continue
        window = speeds[max(0, i - 1) : min(n, i + 2)]
        median_wrist = float(np.median(window)) if window else 0.0
        t = frame["t"]
        in_event = any((ev[0] - 0.25) <= t <= (ev[1] + 0.25) for ev in events)
        frame["active_motion"] = bool(
            in_event
            or frame["wrist_speed"] >= active_wrist_threshold
            or median_wrist >= (active_wrist_threshold * 0.85)
            or frame["body_speed"] >= active_body_threshold
        )
    return raw_frames


def save_local_observations(path: Path, observations: list[dict], proxy_meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "proxy_meta": dict(proxy_meta or {}),
        "frame_observations": observations,
    }
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    os.replace(tmp, path)


def load_local_observations(path: Path) -> tuple[list[dict] | None, dict]:
    path = Path(path)
    if not path.is_file():
        return None, {}
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict) and isinstance(data.get("frame_observations"), list):
            return data["frame_observations"], dict(data.get("proxy_meta") or {})
    except (OSError, ValueError, TypeError):
        return None, {}
    return None, {}


def save_shared_observation_cache(
    cache_dir: Path,
    extraction_cache_key: str,
    observations: list[dict],
    proxy_meta: dict | None = None,
    events: list[list[float]] | None = None,
    visibility: float | None = None,
) -> Path:
    if not re.fullmatch(r"[a-f0-9]{64}", extraction_cache_key):
        raise ValueError("Invalid extraction_cache_key")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    entry = cache_dir / f"{extraction_cache_key}.json"
    marker = cache_dir / f"{extraction_cache_key}.complete"
    payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "extraction_cache_key": extraction_cache_key,
        "proxy_meta": dict(proxy_meta or {}),
        "events": list(events or []),
        "target_visible_fraction": float(visibility) if _is_number(visibility) else 0.0,
        "frame_observations": list(observations),
    }
    serialized = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
    digest = hashlib.sha256(serialized).hexdigest()
    tmp = cache_dir / f".{extraction_cache_key}.{os.getpid()}.tmp"
    tmp_marker = cache_dir / f".{extraction_cache_key}.{os.getpid()}.complete.tmp"
    tmp.write_bytes(serialized)
    os.replace(tmp, entry)
    tmp_marker.write_text(
        json.dumps({"extraction_cache_key": extraction_cache_key, "sha256": digest, "bytes": len(serialized)}) + "\n"
    )
    os.replace(tmp_marker, marker)
    return entry


def load_shared_observation_cache(cache_dir: Path, extraction_cache_key: str) -> dict | None:
    if not re.fullmatch(r"[a-f0-9]{64}", extraction_cache_key):
        return None
    cache_dir = Path(cache_dir)
    entry = cache_dir / f"{extraction_cache_key}.json"
    marker = cache_dir / f"{extraction_cache_key}.complete"
    if not entry.is_file() or not marker.is_file():
        return None
    try:
        marker_data = json.loads(marker.read_text())
        raw = entry.read_bytes()
        if (
            marker_data.get("extraction_cache_key") != extraction_cache_key
            or marker_data.get("bytes") != len(raw)
            or hashlib.sha256(raw).hexdigest() != marker_data.get("sha256")
        ):
            raise ValueError("cache_marker_mismatch")
        data = json.loads(raw.decode("utf-8"))
        if (
            not isinstance(data, dict)
            or data.get("cache_schema_version") != CACHE_SCHEMA_VERSION
            or data.get("extraction_cache_key") != extraction_cache_key
            or not isinstance(data.get("frame_observations"), list)
        ):
            raise ValueError("invalid_cache_payload")
        return data
    except (OSError, ValueError, TypeError, KeyError):
        for p in (marker, entry):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        return None


def _frame_step_ms(observations: list[dict]) -> int:
    if len(observations) < 2:
        return int(round(1000.0 / PROXY_TARGET_FPS))
    diffs = [
        round((float(observations[i + 1]["t"]) - float(observations[i]["t"])) * 1000)
        for i in range(len(observations) - 1)
        if 0 < float(observations[i + 1]["t"]) - float(observations[i]["t"]) <= 2.0
    ]
    if not diffs:
        return int(round(1000.0 / PROXY_TARGET_FPS))
    return max(1, int(round(float(np.median(diffs)))))


def _frames_in_span(observations: list[dict], start_ms: int, end_ms: int, step_ms: int) -> list[tuple[int, int, dict]]:
    """Return clamped (frame_start_ms, frame_end_ms, frame) slices inside [start_ms, end_ms]."""
    if not observations or end_ms <= start_ms:
        return []
    slices: list[tuple[int, int, dict]] = []
    n = len(observations)
    for i, frame in enumerate(observations):
        t_ms = int(round(float(frame["t"]) * 1000))
        if i + 1 < n:
            next_ms = int(round(float(observations[i + 1]["t"]) * 1000))
            f_end = next_ms if 0 < (next_ms - t_ms) <= max(1000, step_ms * 4) else (t_ms + step_ms)
        else:
            f_end = t_ms + step_ms
        s = max(start_ms, t_ms)
        e = min(end_ms, max(t_ms + 1, f_end))
        if e > s and start_ms <= t_ms < end_ms:
            slices.append((s, e, frame))
    if not slices:
        return []
    snap_tol = max(250, step_ms * 2)
    first_s, first_e, first_f = slices[0]
    if first_s - start_ms <= snap_tol:
        slices[0] = (start_ms, max(start_ms + 1, first_e), first_f)
    last_s, last_e, last_f = slices[-1]
    if end_ms - last_e <= snap_tol:
        slices[-1] = (min(last_s, end_ms - 1), end_ms, last_f)
    return slices


def _contiguous_spans(
    slices: list[tuple[int, int, dict]],
    predicate,
    parent_start_ms: int,
    parent_end_ms: int,
    max_bridge_ms: int = 250,
) -> list[tuple[int, int]]:
    raw: list[list[int]] = []
    for s, e, frame in slices:
        if predicate(frame):
            if raw and (s - raw[-1][1]) <= max_bridge_ms:
                raw[-1][1] = max(raw[-1][1], e)
            else:
                raw.append([s, e])
    result: list[tuple[int, int]] = []
    for s, e in raw:
        cs = max(parent_start_ms, int(s))
        ce = min(parent_end_ms, int(e))
        if cs < ce:
            result.append((cs, ce))
    return result


def _cap_and_rank_hints(candidates: list[dict], limit: int = MAX_HINTS_PER_ITEM) -> tuple[list[dict], int, int]:
    total = len(candidates)
    ordered = sorted(
        candidates,
        key=lambda item: (
            int(item.get("_priority", 50)),
            -(int(item["original_end_ms"]) - int(item["original_start_ms"])),
            int(item["original_start_ms"]),
            str(item["code"]),
        ),
    )
    selected = []
    for item in ordered[:limit]:
        clean = {k: v for k, v in item.items() if not k.startswith("_")}
        selected.append(clean)
    selected.sort(key=lambda item: (int(item["original_start_ms"]), int(item["original_end_ms"]), str(item["hint_id"])))
    omitted = max(0, total - len(selected))
    return selected, total, omitted


def _summarize_clip(
    clip_index: int,
    clip_id: str,
    file_name: str,
    start_sec: float,
    end_sec: float,
    start_ms: int,
    end_ms: int,
    observations: list[dict] | None,
    step_ms: int,
) -> dict:
    if observations is None:
        return {
            "clip_id": clip_id,
            "clip_index": clip_index,
            "file": file_name,
            "original_start_ms": start_ms,
            "original_end_ms": end_ms,
            "original_start_seconds": round(float(start_sec), 6),
            "original_end_seconds": round(float(end_sec), 6),
            "heuristic_status": "unknown",
            "identity_status": "unknown",
            "person_presence_ratio": None,
            "active_motion_ratio": None,
            "ambiguous_multi_person_ratio": None,
            "observed_frame_count": 0,
            "hints": [],
            "total_candidate_hints": 0,
            "omitted_hints_count": 0,
            "hints_truncated": False,
        }

    slices = _frames_in_span(observations, start_ms, end_ms, step_ms)
    if not slices:
        return {
            "clip_id": clip_id,
            "clip_index": clip_index,
            "file": file_name,
            "original_start_ms": start_ms,
            "original_end_ms": end_ms,
            "original_start_seconds": round(float(start_sec), 6),
            "original_end_seconds": round(float(end_sec), 6),
            "heuristic_status": "unknown",
            "identity_status": "unknown",
            "person_presence_ratio": None,
            "active_motion_ratio": None,
            "ambiguous_multi_person_ratio": None,
            "observed_frame_count": 0,
            "hints": [],
            "total_candidate_hints": 0,
            "omitted_hints_count": 0,
            "hints_truncated": False,
        }

    frame_count = len(slices)
    present_count = sum(1 for _, _, f in slices if f.get("person_present"))
    active_count = sum(1 for _, _, f in slices if f.get("active_motion"))
    ambiguous_count = sum(1 for _, _, f in slices if f.get("ambiguous_multiple"))
    low_conf_count = sum(1 for _, _, f in slices if f.get("low_confidence_candidate"))

    presence_ratio = round(present_count / frame_count, 4)
    motion_ratio = round(active_count / frame_count, 4)
    ambiguous_ratio = round(ambiguous_count / frame_count, 4)

    candidate_hints: list[dict] = []
    boundary_edge_tol_ms = max(500, step_ms * 3)
    boundary_min_ms = 3000
    mid_low_motion_min_ms = 8000
    interrupt_min_ms = 3000
    ambiguous_min_ms = 1500

    lost_spans = _contiguous_spans(slices, lambda f: not f.get("person_present"), start_ms, end_ms, max_bridge_ms=step_ms * 2)
    low_motion_spans = _contiguous_spans(slices, lambda f: not f.get("active_motion"), start_ms, end_ms, max_bridge_ms=step_ms * 2)
    ambiguous_spans = _contiguous_spans(slices, lambda f: bool(f.get("ambiguous_multiple")), start_ms, end_ms, max_bridge_ms=step_ms * 2)
    degraded_spans = _contiguous_spans(slices, lambda f: bool(f.get("low_confidence_candidate")), start_ms, end_ms, max_bridge_ms=step_ms * 2)

    boundary_lost_intervals: list[tuple[int, int]] = []
    mid_lost_seq = 0
    for s, e in lost_spans:
        dur_ms = e - s
        if dur_ms < interrupt_min_ms:
            continue
        near_start = (s - start_ms) <= boundary_edge_tol_ms
        near_end = (end_ms - e) <= boundary_edge_tol_ms
        if near_start and e < end_ms:
            hs, he = start_ms, e
            dur_s = (he - hs) / 1000.0
            hint = {
                "_priority": 10,
                "hint_id": f"h_clip{clip_index:02d}_head",
                "code": "person_track_lost_near_start",
                "original_start_ms": hs,
                "original_end_ms": he,
                "candidate_start_ms": he,
                "summary": f"Primary person track absent or outside ROI for initial {dur_s:.1f}s ({format_ms(hs)}–{format_ms(he)})",
                "mapped_reasons": ["start_early", "lost_player"],
            }
            candidate_hints.append(hint)
            boundary_lost_intervals.append((hs, he))
        elif near_end and s > start_ms:
            hs, he = s, end_ms
            dur_s = (he - hs) / 1000.0
            hint = {
                "_priority": 10,
                "hint_id": f"h_clip{clip_index:02d}_tail",
                "code": "person_track_lost_near_end",
                "original_start_ms": hs,
                "original_end_ms": he,
                "candidate_end_ms": hs,
                "summary": f"Primary person track disappeared or exited ROI for final {dur_s:.1f}s ({format_ms(hs)}–{format_ms(he)})",
                "mapped_reasons": ["end_late", "lost_player"],
            }
            candidate_hints.append(hint)
            boundary_lost_intervals.append((hs, he))
        else:
            mid_lost_seq += 1
            dur_s = dur_ms / 1000.0
            suffix = "" if mid_lost_seq == 1 else f"_{mid_lost_seq:02d}"
            candidate_hints.append(
                {
                    "_priority": 25,
                    "hint_id": f"h_clip{clip_index:02d}_track_loss{suffix}",
                    "code": "person_track_interrupted",
                    "original_start_ms": s,
                    "original_end_ms": e,
                    "summary": f"Primary person track interrupted for {dur_s:.1f}s ({format_ms(s)}–{format_ms(e)})",
                    "mapped_reasons": ["lost_player"],
                }
            )

    mid_low_seq = 0
    for s, e in low_motion_spans:
        dur_ms = e - s
        near_start = (s - start_ms) <= boundary_edge_tol_ms
        near_end = (end_ms - e) <= boundary_edge_tol_ms
        overlaps_boundary_lost = any(not (e <= bs or s >= be) for bs, be in boundary_lost_intervals)
        if near_start and dur_ms >= boundary_min_ms and e < end_ms and not overlaps_boundary_lost:
            hs, he = start_ms, e
            dur_s = (he - hs) / 1000.0
            candidate_hints.append(
                {
                    "_priority": 15,
                    "hint_id": f"h_clip{clip_index:02d}_head_low_motion",
                    "code": "low_motion_near_start",
                    "original_start_ms": hs,
                    "original_end_ms": he,
                    "candidate_start_ms": he,
                    "summary": f"Extended low-motion span for initial {dur_s:.1f}s ({format_ms(hs)}–{format_ms(he)})",
                    "mapped_reasons": ["start_early", "downtime"],
                }
            )
        elif near_end and dur_ms >= boundary_min_ms and s > start_ms and not overlaps_boundary_lost:
            hs, he = s, end_ms
            dur_s = (he - hs) / 1000.0
            candidate_hints.append(
                {
                    "_priority": 15,
                    "hint_id": f"h_clip{clip_index:02d}_tail_low_motion",
                    "code": "low_motion_near_end",
                    "original_start_ms": hs,
                    "original_end_ms": he,
                    "candidate_end_ms": hs,
                    "summary": f"Extended low-motion span for final {dur_s:.1f}s ({format_ms(hs)}–{format_ms(he)})",
                    "mapped_reasons": ["end_late", "downtime"],
                }
            )
        elif dur_ms >= mid_low_motion_min_ms:
            mid_low_seq += 1
            dur_s = dur_ms / 1000.0
            suffix = "" if mid_low_seq == 1 else f"_{mid_low_seq:02d}"
            candidate_hints.append(
                {
                    "_priority": 20,
                    "hint_id": f"h_clip{clip_index:02d}_low_motion{suffix}",
                    "code": "extended_low_motion",
                    "original_start_ms": s,
                    "original_end_ms": e,
                    "summary": f"Extended low-motion span ({dur_s:.1f}s) at {format_ms(s)}–{format_ms(e)}",
                    "mapped_reasons": ["downtime", "split_clip"],
                }
            )

    amb_seq = 0
    for s, e in ambiguous_spans:
        dur_ms = e - s
        if dur_ms >= ambiguous_min_ms:
            amb_seq += 1
            dur_s = dur_ms / 1000.0
            suffix = "" if amb_seq == 1 else f"_{amb_seq:02d}"
            candidate_hints.append(
                {
                    "_priority": 30,
                    "hint_id": f"h_clip{clip_index:02d}_ambiguous{suffix}",
                    "code": "ambiguous_multiple_persons",
                    "original_start_ms": s,
                    "original_end_ms": e,
                    "summary": f"Multiple similar-scale person tracks in ROI ({dur_s:.1f}s) at {format_ms(s)}–{format_ms(e)} without verified identity lock",
                    "mapped_reasons": ["wrong_player", "multiple_players"],
                }
            )

    is_degraded_unknown = present_count == 0 and low_conf_count >= max(1, frame_count // 2)
    if is_degraded_unknown:
        for idx_d, (s, e) in enumerate(degraded_spans, 1):
            dur_s = (e - s) / 1000.0
            suffix = "" if idx_d == 1 else f"_{idx_d:02d}"
            candidate_hints.append(
                {
                    "_priority": 5,
                    "hint_id": f"h_clip{clip_index:02d}_degraded{suffix}",
                    "code": "detector_visibility_degraded",
                    "original_start_ms": s,
                    "original_end_ms": e,
                    "summary": f"Degraded pose keypoint confidence ({dur_s:.1f}s) at {format_ms(s)}–{format_ms(e)}; camera visibility or occlusion uncertain",
                    "mapped_reasons": ["occluded_or_unclear"],
                }
            )

    hints, total_candidates, omitted_count = _cap_and_rank_hints(candidate_hints, MAX_HINTS_PER_ITEM)

    if is_degraded_unknown:
        identity_status = "unknown"
    elif amb_seq > 0 or ambiguous_ratio >= 0.25:
        identity_status = "ambiguous_multiple_persons"
    elif present_count == 0:
        identity_status = "no_person_detected"
    else:
        identity_status = "unverified_single_primary_track"

    codes = {h["code"] for h in candidate_hints}
    if is_degraded_unknown:
        heuristic_status = "unknown"
    elif codes & {
        "person_track_lost_near_start",
        "person_track_lost_near_end",
        "low_motion_near_start",
        "low_motion_near_end",
    }:
        heuristic_status = "check_boundary_activity"
    elif "extended_low_motion" in codes:
        heuristic_status = "extended_low_motion"
    elif "person_track_interrupted" in codes:
        heuristic_status = "person_track_interrupted"
    elif "ambiguous_multiple_persons" in codes or identity_status == "ambiguous_multiple_persons":
        heuristic_status = "ambiguous_multiple_persons"
    else:
        heuristic_status = "no_detector_warnings"

    return {
        "clip_id": clip_id,
        "clip_index": clip_index,
        "file": file_name,
        "original_start_ms": start_ms,
        "original_end_ms": end_ms,
        "original_start_seconds": round(float(start_sec), 6),
        "original_end_seconds": round(float(end_sec), 6),
        "heuristic_status": heuristic_status,
        "identity_status": identity_status,
        "person_presence_ratio": presence_ratio,
        "active_motion_ratio": motion_ratio,
        "ambiguous_multi_person_ratio": ambiguous_ratio,
        "observed_frame_count": frame_count,
        "hints": hints,
        "total_candidate_hints": total_candidates,
        "omitted_hints_count": omitted_count,
        "hints_truncated": omitted_count > 0,
    }


def _summarize_gap(
    gap_index: int,
    start_ms: int,
    end_ms: int,
    observations: list[dict] | None,
    step_ms: int,
) -> dict:
    gap_id = f"gap-{gap_index:03d}-{start_ms}-{end_ms}"
    start_sec = round(start_ms / 1000.0, 6)
    end_sec = round(end_ms / 1000.0, 6)
    if observations is None:
        return {
            "gap_id": gap_id,
            "gap_index": gap_index,
            "original_start_ms": start_ms,
            "original_end_ms": end_ms,
            "original_start_seconds": start_sec,
            "original_end_seconds": end_sec,
            "heuristic_status": "unknown",
            "identity_status": "unknown",
            "person_presence_ratio": None,
            "active_motion_ratio": None,
            "ambiguous_multi_person_ratio": None,
            "observed_frame_count": 0,
            "hints": [],
            "total_candidate_hints": 0,
            "omitted_hints_count": 0,
            "hints_truncated": False,
        }

    slices = _frames_in_span(observations, start_ms, end_ms, step_ms)
    if not slices:
        return {
            "gap_id": gap_id,
            "gap_index": gap_index,
            "original_start_ms": start_ms,
            "original_end_ms": end_ms,
            "original_start_seconds": start_sec,
            "original_end_seconds": end_sec,
            "heuristic_status": "unknown",
            "identity_status": "unknown",
            "person_presence_ratio": None,
            "active_motion_ratio": None,
            "ambiguous_multi_person_ratio": None,
            "observed_frame_count": 0,
            "hints": [],
            "total_candidate_hints": 0,
            "omitted_hints_count": 0,
            "hints_truncated": False,
        }

    frame_count = len(slices)
    present_count = sum(1 for _, _, f in slices if f.get("person_present"))
    active_count = sum(1 for _, _, f in slices if f.get("active_motion"))
    ambiguous_count = sum(1 for _, _, f in slices if f.get("ambiguous_multiple"))
    low_conf_count = sum(1 for _, _, f in slices if f.get("low_confidence_candidate"))

    presence_ratio = round(present_count / frame_count, 4)
    motion_ratio = round(active_count / frame_count, 4)
    ambiguous_ratio = round(ambiguous_count / frame_count, 4)

    candidate_hints: list[dict] = []
    motion_spans = _contiguous_spans(slices, lambda f: bool(f.get("active_motion")), start_ms, end_ms, max_bridge_ms=max(500, step_ms * 4))
    ambiguous_spans = _contiguous_spans(slices, lambda f: bool(f.get("ambiguous_multiple")), start_ms, end_ms, max_bridge_ms=step_ms * 2)
    degraded_spans = _contiguous_spans(slices, lambda f: bool(f.get("low_confidence_candidate")), start_ms, end_ms, max_bridge_ms=step_ms * 2)

    motion_seq = 0
    for s, e in motion_spans:
        dur_ms = e - s
        if dur_ms >= 1500:
            motion_seq += 1
            dur_s = dur_ms / 1000.0
            suffix = "" if motion_seq == 1 else f"_{motion_seq:02d}"
            candidate_hints.append(
                {
                    "_priority": 10,
                    "hint_id": f"h_gap{gap_index:02d}_motion{suffix}",
                    "code": "sustained_roi_motion_in_gap",
                    "original_start_ms": s,
                    "original_end_ms": e,
                    "summary": f"Sustained court-ROI person motion ({dur_s:.1f}s) at {format_ms(s)}–{format_ms(e)} inside gap {format_ms(start_ms)}–{format_ms(end_ms)}",
                    "mapped_reasons": ["missed_play"],
                }
            )
        elif dur_ms >= 750 and ((s - start_ms) <= 1500 or (end_ms - e) <= 1500):
            motion_seq += 1
            dur_s = dur_ms / 1000.0
            suffix = "" if motion_seq == 1 else f"_{motion_seq:02d}"
            candidate_hints.append(
                {
                    "_priority": 15,
                    "hint_id": f"h_gap{gap_index:02d}_boundary_motion{suffix}",
                    "code": "boundary_adjacent_roi_motion_in_gap",
                    "original_start_ms": s,
                    "original_end_ms": e,
                    "summary": f"Boundary-adjacent court-ROI person motion ({dur_s:.1f}s) at {format_ms(s)}–{format_ms(e)} inside gap {format_ms(start_ms)}–{format_ms(end_ms)}",
                    "mapped_reasons": ["missed_play", "too_short"],
                }
            )

    amb_seq = 0
    for s, e in ambiguous_spans:
        dur_ms = e - s
        if dur_ms >= 1500:
            amb_seq += 1
            dur_s = dur_ms / 1000.0
            suffix = "" if amb_seq == 1 else f"_{amb_seq:02d}"
            candidate_hints.append(
                {
                    "_priority": 30,
                    "hint_id": f"h_gap{gap_index:02d}_ambiguous{suffix}",
                    "code": "ambiguous_multiple_persons",
                    "original_start_ms": s,
                    "original_end_ms": e,
                    "summary": f"Multiple similar-scale person tracks in removed gap ({dur_s:.1f}s) at {format_ms(s)}–{format_ms(e)}",
                    "mapped_reasons": ["wrong_player", "multiple_players"],
                }
            )

    is_degraded_unknown = present_count == 0 and low_conf_count >= max(1, frame_count // 2)
    if is_degraded_unknown:
        for idx_d, (s, e) in enumerate(degraded_spans, 1):
            dur_s = (e - s) / 1000.0
            suffix = "" if idx_d == 1 else f"_{idx_d:02d}"
            candidate_hints.append(
                {
                    "_priority": 5,
                    "hint_id": f"h_gap{gap_index:02d}_degraded{suffix}",
                    "code": "detector_visibility_degraded",
                    "original_start_ms": s,
                    "original_end_ms": e,
                    "summary": f"Degraded pose keypoint confidence ({dur_s:.1f}s) at {format_ms(s)}–{format_ms(e)} inside removed gap",
                    "mapped_reasons": ["occluded_or_unclear"],
                }
            )

    hints, total_candidates, omitted_count = _cap_and_rank_hints(candidate_hints, MAX_HINTS_PER_ITEM)

    if is_degraded_unknown:
        identity_status = "unknown"
    elif amb_seq > 0 or ambiguous_ratio >= 0.25:
        identity_status = "ambiguous_multiple_persons"
    elif present_count == 0:
        identity_status = "no_person_detected"
    else:
        identity_status = "unverified_single_primary_track"

    codes = {h["code"] for h in candidate_hints}
    if is_degraded_unknown:
        heuristic_status = "unknown"
    elif codes & {"sustained_roi_motion_in_gap", "boundary_adjacent_roi_motion_in_gap"}:
        heuristic_status = "check_boundary_activity"
    elif "ambiguous_multiple_persons" in codes:
        heuristic_status = "ambiguous_multiple_persons"
    else:
        heuristic_status = "no_detector_warnings"

    return {
        "gap_id": gap_id,
        "gap_index": gap_index,
        "original_start_ms": start_ms,
        "original_end_ms": end_ms,
        "original_start_seconds": start_sec,
        "original_end_seconds": end_sec,
        "heuristic_status": heuristic_status,
        "identity_status": identity_status,
        "person_presence_ratio": presence_ratio,
        "active_motion_ratio": motion_ratio,
        "ambiguous_multi_person_ratio": ambiguous_ratio,
        "observed_frame_count": frame_count,
        "hints": hints,
        "total_candidate_hints": total_candidates,
        "omitted_hints_count": omitted_count,
        "hints_truncated": omitted_count > 0,
    }


def compute_video_version_id(public: dict, attempt_id: str | None = None) -> tuple[str, str]:
    if not isinstance(public, dict) or not re.fullmatch(r"[a-f0-9]{64}", str(public.get("id", ""))):
        raise ValueError("Frozen public_session with valid 64-hex id is required")
    published_at = public.get("published_at")
    if not _is_number(published_at):
        raise ValueError("Frozen public_session must include numeric published_at")
    canonical = {
        "schema_version": 1,
        "source_sha256": public["id"],
        "media_directory": public.get("media_directory") or public["id"],
        "published_at": float(published_at),
        "algorithm": str(public.get("algorithm", "unknown")),
        "chapter_alignment": str(public.get("chapter_alignment", "")),
        "attempt_id": attempt_id,
        "clips": [
            {
                "id": str(c["id"]),
                "index": int(c["index"]),
                "file": str(c["file"]),
                "original_start_seconds": float(c["original_start_seconds"]),
                "original_end_seconds": float(c["original_end_seconds"]),
                "duration_seconds": float(c["duration_seconds"]),
                "playback_start_seconds": float(c["playback_start_seconds"]),
                "playback_end_seconds": float(c["playback_end_seconds"]),
            }
            for c in public.get("clips", [])
        ],
    }
    digest = compact_json_sha256(canonical)
    # Shared website/server identity: compact JSON of source SHA + frozen token.
    identity = hashlib.sha256(json.dumps([public["id"], float(published_at)], separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return f"v_{identity[:32]}", digest


def build_publication_linkage(
    public: dict,
    task_id: str | None = None,
    attempt_id: str | None = None,
    video_version_id: str | None = None,
) -> dict:
    computed_version_id, payload_sha256 = compute_video_version_id(public, attempt_id=attempt_id)
    version_id = str(video_version_id or computed_version_id)
    clip_ids = [str(c["id"]) for c in public.get("clips", [])]
    video_info = public.get("video") if isinstance(public.get("video"), dict) else {}
    return {
        "session_id": public["id"],
        "published_at": float(public["published_at"]),
        "media_directory": str(public.get("media_directory") or public["id"]),
        "algorithm": str(public.get("algorithm", "unknown")),
        "chapter_alignment": str(public.get("chapter_alignment", "")),
        "combined_duration_seconds": float(video_info.get("duration_seconds", 0.0)),
        "clip_count": len(clip_ids),
        "clip_ids": clip_ids,
        "task_id": task_id,
        "attempt_id": attempt_id,
        "video_version_id": version_id,
        "version_payload_sha256": payload_sha256,
    }


def build_heuristic_diagnostics(
    source_sha256: str,
    duration_seconds: float,
    intervals: list[list[float]],
    cfg: dict | None = None,
    meta: dict | None = None,
    analysis: dict | None = None,
    frame_observations: list[dict] | None = None,
    public: dict | None = None,
    task_id: str | None = None,
    attempt_id: str | None = None,
    video_version_id: str | None = None,
    model_sha256: str | None = None,
    extractor_code_digest: str | None = None,
    cache_reuse: dict | None = None,
) -> dict:
    if not isinstance(source_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", source_sha256):
        raise ValueError("Invalid source_sha256")
    if not _is_number(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive and finite")
    cfg = dict(cfg or {})
    meta = dict(meta or {"duration": duration_seconds})
    meta.setdefault("duration", duration_seconds)
    analysis = dict(analysis or {})

    proxy_meta = dict(analysis.get("proxy_observation_meta") or {})
    if frame_observations is None:
        if isinstance(analysis.get("_frame_observations"), list):
            frame_observations = analysis["_frame_observations"]
        elif isinstance(cfg.get("_frame_observations"), list):
            frame_observations = cfg["_frame_observations"]
        elif isinstance(analysis.get("pose_records"), list) or isinstance(cfg.get("pose_records"), list):
            records = analysis.get("pose_records") if isinstance(analysis.get("pose_records"), list) else cfg.get("pose_records")
            pw, ph = _default_proxy_dimensions(meta)
            pw = int(proxy_meta.get("width") or pw)
            ph = int(proxy_meta.get("height") or ph)
            frame_observations = extract_frame_observations(
                records,
                cfg,
                pw,
                ph,
                events=analysis.get("event_intervals"),
            )

    extraction_cache_key, provenance = build_extraction_provenance(
        source_sha256=source_sha256,
        meta=meta,
        cfg=cfg,
        proxy_meta=proxy_meta,
        model_sha256=model_sha256,
        extractor_code_digest=extractor_code_digest,
        cache_reuse=cache_reuse,
    )

    source_duration_ms = max(1, int(round(float(duration_seconds) * 1000)))
    step_ms = _frame_step_ms(frame_observations) if frame_observations else int(round(1000.0 / PROXY_TARGET_FPS))

    clips_summary: list[dict] = []
    gaps_summary: list[dict] = []
    cursor_ms = 0
    cursor_sec = 0.0
    public_clips = public.get("clips") if isinstance(public, dict) and isinstance(public.get("clips"), list) else None

    for idx, pair in enumerate(intervals, 1):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("Invalid clip interval")
        start_sec, end_sec = float(pair[0]), float(pair[1])
        if not (_is_number(start_sec) and _is_number(end_sec)) or not (cursor_sec <= start_sec < end_sec <= duration_seconds + 1e-6):
            raise ValueError("Kept intervals must be ordered, non-overlapping and inside source duration")
        start_ms = max(cursor_ms, int(round(start_sec * 1000)))
        end_ms = min(source_duration_ms, int(round(end_sec * 1000)))
        if end_ms <= start_ms:
            raise ValueError("Kept interval collapses to zero milliseconds")

        if start_ms > cursor_ms:
            gap_idx = len(gaps_summary) + 1
            gaps_summary.append(_summarize_gap(gap_idx, cursor_ms, start_ms, frame_observations, step_ms))

        if public_clips and idx - 1 < len(public_clips):
            p_clip = public_clips[idx - 1]
            c_id = str(p_clip["id"])
            c_file = str(p_clip.get("file", f"Clips/clip-{idx:03d}.mp4"))
        else:
            c_id = clip_id_for_bounds(source_sha256, start_sec, end_sec)
            c_file = f"Clips/clip-{idx:03d}.mp4"

        clips_summary.append(
            _summarize_clip(
                clip_index=idx,
                clip_id=c_id,
                file_name=c_file,
                start_sec=start_sec,
                end_sec=end_sec,
                start_ms=start_ms,
                end_ms=end_ms,
                observations=frame_observations,
                step_ms=step_ms,
            )
        )
        cursor_ms = end_ms
        cursor_sec = end_sec

    if cursor_ms < source_duration_ms:
        gap_idx = len(gaps_summary) + 1
        gaps_summary.append(_summarize_gap(gap_idx, cursor_ms, source_duration_ms, frame_observations, step_ms))

    omitted_clip_hints = sum(int(c["omitted_hints_count"]) for c in clips_summary)
    omitted_gap_hints = sum(int(g["omitted_hints_count"]) for g in gaps_summary)
    total_omitted = omitted_clip_hints + omitted_gap_hints

    publication_linkage = None
    resolved_version_id = video_version_id
    published_at = None
    if public is not None:
        publication_linkage = build_publication_linkage(
            public,
            task_id=task_id,
            attempt_id=attempt_id,
            video_version_id=video_version_id,
        )
        resolved_version_id = publication_linkage["video_version_id"]
        published_at = publication_linkage["published_at"]

    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_sha256": source_sha256,
        "source_duration_ms": source_duration_ms,
        "source_duration_seconds": round(float(duration_seconds), 6),
        "task_id": task_id,
        "attempt_id": attempt_id,
        "video_version_id": resolved_version_id,
        "published_at": published_at,
        "extraction_cache_key": extraction_cache_key,
        "truth_status": TRUTH_STATUS,
        "training_label_status": TRAINING_LABEL_STATUS,
        "generator": {
            "kind": GENERATOR_KIND,
            "truth_status": TRUTH_STATUS,
            "training_label_status": TRAINING_LABEL_STATUS,
            "benchmark_truth": False,
            "verified_play_or_swing": False,
            "verified_player_identity": False,
        },
        "provenance": provenance,
        "publication": publication_linkage,
        "limits": {
            "max_hints_per_clip": MAX_HINTS_PER_ITEM,
            "max_hints_per_gap": MAX_HINTS_PER_ITEM,
            "max_artifact_bytes": MAX_DIAGNOSTICS_BYTES,
            "omitted_clip_hints_count": omitted_clip_hints,
            "omitted_gap_hints_count": omitted_gap_hints,
            "omitted_hints_count": total_omitted,
            "hints_truncated": total_omitted > 0,
        },
        "omitted_hints_count": total_omitted,
        "clips": clips_summary,
        "removed_gaps": gaps_summary,
    }
    validate_heuristic_diagnostics(payload, expected_source_sha256=source_sha256, expected_duration_ms=source_duration_ms)
    return payload


def bind_publication_diagnostics(
    path_or_payload: Path | dict,
    public: dict,
    task_id: str | None = None,
    attempt_id: str | None = None,
    video_version_id: str | None = None,
) -> dict:
    """Bind frozen public_session and attempt linkage into heuristic-diagnostics.json."""
    path: Path | None = None
    if isinstance(path_or_payload, Path):
        path = path_or_payload
        payload = json.loads(path.read_text())
    elif isinstance(path_or_payload, str):
        path = Path(path_or_payload)
        payload = json.loads(path.read_text())
    elif isinstance(path_or_payload, dict):
        payload = json.loads(json.dumps(path_or_payload, allow_nan=False))
    else:
        raise TypeError("Expected path or dict for bind_publication_diagnostics")

    eff_task_id = task_id if task_id is not None else payload.get("task_id")
    eff_attempt_id = attempt_id if attempt_id is not None else payload.get("attempt_id")
    linkage = build_publication_linkage(
        public,
        task_id=eff_task_id,
        attempt_id=eff_attempt_id,
        video_version_id=video_version_id,
    )
    if payload.get("source_sha256") != public["id"]:
        raise ValueError("Diagnostics source_sha256 does not match frozen public_session id")
    public_clips = public.get("clips", [])
    diag_clips = payload.get("clips", [])
    if len(public_clips) != len(diag_clips):
        raise ValueError("Diagnostics clip count does not match frozen public_session clips")
    for p_clip, d_clip in zip(public_clips, diag_clips):
        if d_clip.get("clip_id") != p_clip.get("id") or d_clip.get("clip_index") != p_clip.get("index"):
            raise ValueError("Diagnostics clip identity mismatch against frozen public_session")

    payload["task_id"] = eff_task_id
    payload["attempt_id"] = eff_attempt_id
    payload["video_version_id"] = linkage["video_version_id"]
    payload["published_at"] = linkage["published_at"]
    payload["publication"] = linkage

    validate_heuristic_diagnostics(payload, expected_source_sha256=public["id"])
    if path is not None:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        os.replace(tmp, path)
    return payload


def _validate_hint(hint: dict, parent_start_ms: int, parent_end_ms: int, source_duration_ms: int, is_clip: bool, seen_ids: set[str]) -> None:
    if not isinstance(hint, dict):
        raise ValueError("Hint must be an object")
    hint_id = hint.get("hint_id")
    if not isinstance(hint_id, str) or not hint_id or hint_id in seen_ids:
        raise ValueError(f"Invalid or duplicate hint_id: {hint_id!r}")
    seen_ids.add(hint_id)
    code = hint.get("code")
    if code not in ALLOWED_HINT_CODES:
        raise ValueError(f"Unsupported hint code: {code!r}")
    hs = hint.get("original_start_ms")
    he = hint.get("original_end_ms")
    if not (_is_int(hs) and _is_int(he)):
        raise ValueError("Hint original_start_ms and original_end_ms must be integers")
    if not (parent_start_ms <= hs < he <= parent_end_ms <= source_duration_ms):
        raise ValueError(
            f"Hint [{hs}, {he}] violates parent bounds [{parent_start_ms}, {parent_end_ms}] or source_duration_ms {source_duration_ms}"
        )
    if "candidate_start_ms" in hint and hint["candidate_start_ms"] is not None:
        cs = hint["candidate_start_ms"]
        if not is_clip or not _is_int(cs) or not (parent_start_ms < cs < parent_end_ms):
            raise ValueError("Invalid candidate_start_ms on hint")
    if "candidate_end_ms" in hint and hint["candidate_end_ms"] is not None:
        ce = hint["candidate_end_ms"]
        if not is_clip or not _is_int(ce) or not (parent_start_ms < ce < parent_end_ms):
            raise ValueError("Invalid candidate_end_ms on hint")
    summary = hint.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("Hint summary must be a non-empty string")
    reasons = hint.get("mapped_reasons")
    if not isinstance(reasons, list) or not reasons or not all(isinstance(r, str) and r for r in reasons):
        raise ValueError("Hint mapped_reasons must be a non-empty list of strings")


def validate_heuristic_diagnostics(
    payload: dict,
    expected_source_sha256: str | None = None,
    expected_duration_ms: int | None = None,
) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Heuristic diagnostics payload must be a JSON object")
    serialized = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(serialized) > MAX_DIAGNOSTICS_BYTES:
        raise ValueError(f"Heuristic diagnostics exceeds {MAX_DIAGNOSTICS_BYTES} bytes")

    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported heuristic diagnostics schema_version: {payload.get('schema_version')!r}")
    source_sha256 = payload.get("source_sha256")
    if not isinstance(source_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", source_sha256):
        raise ValueError("Invalid source_sha256 in heuristic diagnostics")
    if expected_source_sha256 and source_sha256 != expected_source_sha256:
        raise ValueError("Heuristic diagnostics source_sha256 mismatch")

    source_duration_ms = payload.get("source_duration_ms")
    if not _is_int(source_duration_ms) or source_duration_ms <= 0:
        raise ValueError("source_duration_ms must be a positive integer")
    if expected_duration_ms is not None and abs(source_duration_ms - int(expected_duration_ms)) > 2:
        raise ValueError("source_duration_ms mismatch against source metadata")

    cache_key = payload.get("extraction_cache_key")
    if not isinstance(cache_key, str) or not re.fullmatch(r"[a-f0-9]{64}", cache_key):
        raise ValueError("Invalid extraction_cache_key")

    generator = payload.get("generator")
    if not isinstance(generator, dict):
        raise ValueError("Missing generator descriptor")
    if generator.get("kind") != GENERATOR_KIND:
        raise ValueError(f"Invalid generator.kind: {generator.get('kind')!r}")
    if generator.get("truth_status") != TRUTH_STATUS or payload.get("truth_status") != TRUTH_STATUS:
        raise ValueError("Invalid truth_status: heuristics must never claim play or swing truth")
    if generator.get("training_label_status") != TRAINING_LABEL_STATUS or payload.get("training_label_status") != TRAINING_LABEL_STATUS:
        raise ValueError("Invalid training_label_status")
    if generator.get("benchmark_truth") is not False or generator.get("verified_play_or_swing") is not False:
        raise ValueError("Heuristics must never claim benchmark or play/swing truth")

    provenance = payload.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("source_sha256") != source_sha256:
        raise ValueError("Missing or mismatched provenance in heuristic diagnostics")
    if provenance.get("extraction_cache_key") != cache_key:
        raise ValueError("provenance.extraction_cache_key mismatch")
    proxy_spec = provenance.get("proxy_spec")
    if not isinstance(proxy_spec, dict) or not _is_number(proxy_spec.get("target_fps")) or proxy_spec["target_fps"] <= 0:
        raise ValueError("Invalid provenance.proxy_spec")
    detector_spec = provenance.get("detector_spec")
    if not isinstance(detector_spec, dict) or not isinstance(detector_spec.get("backend"), str):
        raise ValueError("Invalid provenance.detector_spec")
    roi_cfg = provenance.get("roi_and_filter_config")
    if not isinstance(roi_cfg, dict) or not isinstance(roi_cfg.get("target_region"), list) or len(roi_cfg["target_region"]) != 4:
        raise ValueError("Invalid provenance.roi_and_filter_config")

    publication = payload.get("publication")
    if publication is not None:
        if not isinstance(publication, dict) or publication.get("session_id") != source_sha256:
            raise ValueError("Invalid publication linkage in heuristic diagnostics")
        if not _is_number(publication.get("published_at")) or payload.get("published_at") != publication.get("published_at"):
            raise ValueError("Mismatched published_at in publication linkage")
        version_id = publication.get("video_version_id")
        if not isinstance(version_id, str) or not re.fullmatch(r"v_[a-f0-9]{32}", version_id) or payload.get("video_version_id") != version_id:
            raise ValueError("Invalid video_version_id in publication linkage")

    clips = payload.get("clips")
    gaps = payload.get("removed_gaps")
    if not isinstance(clips, list) or not isinstance(gaps, list):
        raise ValueError("clips and removed_gaps must be lists")

    seen_clip_ids: set[str] = set()
    seen_clip_indices: set[int] = set()
    seen_hint_ids: set[str] = set()
    prev_end_ms = 0
    computed_omitted_clips = 0

    for clip in clips:
        if not isinstance(clip, dict):
            raise ValueError("Clip entry must be an object")
        c_id = clip.get("clip_id")
        c_idx = clip.get("clip_index")
        if not isinstance(c_id, str) or not re.fullmatch(r"[a-f0-9]{24}", c_id) or c_id in seen_clip_ids:
            raise ValueError(f"Invalid or duplicate clip_id: {c_id!r}")
        if not _is_int(c_idx) or c_idx < 1 or c_idx in seen_clip_indices:
            raise ValueError(f"Invalid or duplicate clip_index: {c_idx!r}")
        seen_clip_ids.add(c_id)
        seen_clip_indices.add(c_idx)

        cs = clip.get("original_start_ms")
        ce = clip.get("original_end_ms")
        if not (_is_int(cs) and _is_int(ce)) or not (prev_end_ms <= cs < ce <= source_duration_ms):
            raise ValueError(f"Invalid clip bounds [{cs}, {ce}] within source_duration_ms {source_duration_ms}")
        prev_end_ms = ce

        status = clip.get("heuristic_status")
        if status not in ALLOWED_HEURISTIC_STATUSES:
            raise ValueError(f"Invalid clip heuristic_status: {status!r}")
        id_status = clip.get("identity_status")
        if id_status not in ALLOWED_IDENTITY_STATUSES:
            raise ValueError(f"Invalid clip identity_status: {id_status!r}")

        for ratio_key in ("person_presence_ratio", "active_motion_ratio", "ambiguous_multi_person_ratio"):
            val = clip.get(ratio_key)
            if val is None:
                if status != "unknown":
                    raise ValueError(f"{ratio_key} may be null only when heuristic_status is 'unknown'")
            elif not _is_number(val) or not (0.0 <= float(val) <= 1.0):
                raise ValueError(f"Invalid {ratio_key}: {val!r}")

        hints = clip.get("hints")
        if not isinstance(hints, list) or len(hints) > MAX_HINTS_PER_ITEM:
            raise ValueError("Clip hints must be a list of at most 5 items")
        for hint in hints:
            _validate_hint(hint, cs, ce, source_duration_ms, is_clip=True, seen_ids=seen_hint_ids)

        omitted = clip.get("omitted_hints_count", 0)
        total_cand = clip.get("total_candidate_hints", len(hints))
        if not _is_int(omitted) or omitted < 0 or not _is_int(total_cand) or total_cand != len(hints) + omitted:
            raise ValueError("Invalid clip omitted_hints_count / total_candidate_hints")
        if clip.get("hints_truncated") != (omitted > 0):
            raise ValueError("Invalid clip hints_truncated flag")
        computed_omitted_clips += omitted

    seen_gap_indices: set[int] = set()
    prev_gap_end_ms = 0
    computed_omitted_gaps = 0

    for gap in gaps:
        if not isinstance(gap, dict):
            raise ValueError("Removed gap entry must be an object")
        g_idx = gap.get("gap_index")
        if not _is_int(g_idx) or g_idx < 1 or g_idx in seen_gap_indices:
            raise ValueError(f"Invalid or duplicate gap_index: {g_idx!r}")
        seen_gap_indices.add(g_idx)

        gs = gap.get("original_start_ms")
        ge = gap.get("original_end_ms")
        if not (_is_int(gs) and _is_int(ge)) or not (prev_gap_end_ms <= gs < ge <= source_duration_ms):
            raise ValueError(f"Invalid gap bounds [{gs}, {ge}] within source_duration_ms {source_duration_ms}")
        for clip in clips:
            if not (ge <= clip["original_start_ms"] or gs >= clip["original_end_ms"]):
                raise ValueError(f"Removed gap [{gs}, {ge}] overlaps kept clip [{clip['original_start_ms']}, {clip['original_end_ms']}]")
        prev_gap_end_ms = ge

        status = gap.get("heuristic_status")
        if status not in ALLOWED_HEURISTIC_STATUSES:
            raise ValueError(f"Invalid gap heuristic_status: {status!r}")
        id_status = gap.get("identity_status")
        if id_status is not None and id_status not in ALLOWED_IDENTITY_STATUSES:
            raise ValueError(f"Invalid gap identity_status: {id_status!r}")

        hints = gap.get("hints")
        if not isinstance(hints, list) or len(hints) > MAX_HINTS_PER_ITEM:
            raise ValueError("Gap hints must be a list of at most 5 items")
        for hint in hints:
            _validate_hint(hint, gs, ge, source_duration_ms, is_clip=False, seen_ids=seen_hint_ids)

        omitted = gap.get("omitted_hints_count", 0)
        total_cand = gap.get("total_candidate_hints", len(hints))
        if not _is_int(omitted) or omitted < 0 or not _is_int(total_cand) or total_cand != len(hints) + omitted:
            raise ValueError("Invalid gap omitted_hints_count / total_candidate_hints")
        if gap.get("hints_truncated") != (omitted > 0):
            raise ValueError("Invalid gap hints_truncated flag")
        computed_omitted_gaps += omitted

    total_omitted = computed_omitted_clips + computed_omitted_gaps
    if payload.get("omitted_hints_count") != total_omitted:
        raise ValueError("Top-level omitted_hints_count mismatch")
    limits = payload.get("limits")
    if not isinstance(limits, dict) or limits.get("omitted_hints_count") != total_omitted:
        raise ValueError("limits.omitted_hints_count mismatch")
    return payload

"""Hybrid AI + local pose video filtering for tennis practice sessions."""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

import av
import numpy as np
from PIL import Image, ImageDraw

LOG = logging.getLogger("tennis")

VERSION = "hybrid-ai-vision-v1"

NUMERIC_PARAMETERS = {
    "target_region",
    "min_player_height",
    "swing_speed",
    "arm_reach",
    "padding_before",
    "padding_after",
    "merge_gap",
}
AI_PARAMETERS = {"ai_model", "allow_model_fallback", "ai_tuning"}
PARAMETERS = NUMERIC_PARAMETERS | {"processing_strategy"} | AI_PARAMETERS

PROCESSING_STRATEGIES = ("vision_only", "hybrid_ai_vision", "ai_only")
AI_AGGRESSIVENESS_LEVELS = ("conservative", "balanced", "aggressive")
AI_ACCURACY_TIERS = ("fast", "standard", "thorough")
AI_PROVIDERS = ("antigravity", "gemini", "codex")
AI_MODEL_RE = re.compile(r"^(antigravity|gemini|codex):[A-Za-z0-9._-]{1,64}$")
MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
AGY_LINE_RE = re.compile(r"^([a-z0-9][a-z0-9._-]{1,63})\s{2,}(.+)$")
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

ALLOWED_ACTIONS = ("keep", "trim", "drop")
ALLOWED_CATEGORIES = (
    "rally",
    "serve",
    "drill",
    "warmup",
    "shadow_swing",
    "ball_pickup",
    "walking",
    "idle",
    "uncertain",
)

MAX_CANDIDATE_SECONDS = 45.0
MIN_VALLEY_GAP_SECONDS = 2.0
FRAME_GRID_SECONDS = 0.125
MAX_POSE_CACHE_ENTRIES = 16
MAX_POSE_CACHE_BYTES = 64 * 1024 * 1024
MAX_PROVIDER_OUTPUT_BYTES = 4 * 1024 * 1024
MAX_AI_REFINEMENT_FINDINGS_BYTES = 32768
GEMINI_ENDPOINT_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

TIER_CONFIG = {
    "fast": {"frames": 6, "cols": 3, "rows": 2, "tile_w": 320, "batch_size": 10, "effort": "low"},
    "standard": {
        "frames": 12,
        "cols": 4,
        "rows": 3,
        "long_span_s": 24.0,
        "long_frames": 16,
        "long_cols": 4,
        "long_rows": 4,
        "tile_w": 384,
        "batch_size": 6,
        "effort": "low",
    },
    "thorough": {
        "frames": 16,
        "cols": 4,
        "rows": 4,
        "long_span_s": 30.0,
        "long_frames": 20,
        "long_cols": 5,
        "long_rows": 4,
        "tile_w": 480,
        "batch_size": 4,
        "effort": "medium",
    },
}

DECISION_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decisions"],
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["candidate_id", "action", "category", "confidence", "reason"],
                "properties": {
                    "candidate_id": {"type": "string"},
                    "action": {"type": "string", "enum": list(ALLOWED_ACTIONS)},
                    "category": {"type": "string", "enum": list(ALLOWED_CATEGORIES)},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "reason": {"type": "string"},
                    "trimmed_start_s": {"type": "number"},
                    "trimmed_end_s": {"type": "number"},
                },
            },
        }
    },
}


class ProcessingTaskError(RuntimeError):
    """Structured non-crash worker task failure carrying server error_code and sanitized text."""

    def __init__(self, error_code: str, message: str = "", *, retryable: bool = False):
        self.error_code = str(error_code or "processing_failed")
        self.retryable = bool(retryable)
        clean = sanitize_error_message(message or self.error_code)
        super().__init__(clean)
        self.message = clean


class ProviderCallError(RuntimeError):
    """Retryable/fallback-eligible AI provider call or response schema error."""

    def __init__(self, error_code: str, message: str = ""):
        self.error_code = str(error_code or "ai_provider_failed")
        clean = sanitize_error_message(message or self.error_code)
        super().__init__(clean)
        self.message = clean


def sanitize_error_message(text: object, max_len: int = 400) -> str:
    """Strip credentials, tokens, and filesystem paths before reporting errors to the server."""
    raw = str(text or "").replace("\r", " ").replace("\n", " ")
    raw = ANSI_ESCAPE_RE.sub("", raw)
    raw = re.sub(r"AIza[A-Za-z0-9_-]{8,}", "[REDACTED_KEY]", raw)
    raw = re.sub(r"sk-[A-Za-z0-9_-]{8,}", "[REDACTED_KEY]", raw)
    raw = re.sub(r"(?i)bearer\s+[A-Za-z0-9._~+/-]{8,}=*", "Bearer [REDACTED]", raw)
    raw = re.sub(r"(?i)([?&](?:key|token|api_key|secret)=)[^&\s\"']+", r"\1[REDACTED]", raw)
    raw = re.sub(r"(?i)(?:api[_-]?key|token|secret)\s*[:=]\s*[^\s,;\"']+", "[REDACTED]", raw)
    raw = re.sub(r"(?<![A-Za-z0-9:])/(?:[A-Za-z0-9._@-]+/)*[A-Za-z0-9._@-]+", "[path]", raw)
    raw = re.sub(r"[A-Za-z]:\\(?:[^\\\s]+\\)*[^\\\s]+", "[path]", raw)
    clean = " ".join(raw.split())
    if len(clean) > max_len:
        clean = clean[: max_len - 3].rstrip() + "..."
    return clean


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def validate_task_parameters(parameters: object) -> None:
    """Validate task parameters using the same rules and error codes as the server."""
    if not isinstance(parameters, dict) or set(parameters) - PARAMETERS:
        raise ValueError("invalid_parameter_overrides")
    strategy = parameters.get("processing_strategy", "vision_only")
    if not isinstance(strategy, str) or strategy not in PROCESSING_STRATEGIES:
        raise ValueError("invalid_processing_strategy")
    if strategy == "ai_only":
        raise ProcessingTaskError("unsupported_processing_strategy", "Strategy ai_only is not yet supported.")
    if strategy == "vision_only" and any(k in parameters for k in AI_PARAMETERS):
        raise ValueError("ai_parameters_require_ai_strategy")
    for key, value in parameters.items():
        if key == "target_region":
            if (
                not isinstance(value, list)
                or len(value) != 4
                or any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1 for v in value)
                or not value[0] < value[2]
                or not value[1] < value[3]
            ):
                raise ValueError("invalid_parameter_overrides")
        elif key == "processing_strategy":
            continue
        elif key == "ai_model":
            if not isinstance(value, str) or (value != "auto" and not AI_MODEL_RE.fullmatch(value)):
                raise ValueError("invalid_ai_model")
        elif key == "allow_model_fallback":
            if type(value) is not bool:
                raise ValueError("invalid_allow_model_fallback")
        elif key == "ai_tuning":
            if not isinstance(value, dict) or set(value) - {"aggressiveness", "accuracy_tier"}:
                raise ValueError("invalid_ai_tuning")
            if "aggressiveness" in value and (
                not isinstance(value["aggressiveness"], str) or value["aggressiveness"] not in AI_AGGRESSIVENESS_LEVELS
            ):
                raise ValueError("invalid_ai_tuning")
            if "accuracy_tier" in value and (
                not isinstance(value["accuracy_tier"], str) or value["accuracy_tier"] not in AI_ACCURACY_TIERS
            ):
                raise ValueError("invalid_ai_tuning")
        elif type(value) not in (int, float) or not math.isfinite(value) or value < 0 or value > 10000:
            raise ValueError("invalid_parameter_overrides")


def normalize_task_parameters(parameters: object) -> dict:
    """Validate and return normalized parameters with hybrid_ai_vision defaults applied."""
    validate_task_parameters(parameters)
    out = copy.deepcopy(parameters)
    if out.get("processing_strategy") == "hybrid_ai_vision":
        out.setdefault("ai_model", "auto")
        if "allow_model_fallback" not in out:
            out["allow_model_fallback"] = True
        tuning = dict(out.get("ai_tuning") or {})
        out["ai_tuning"] = {
            "aggressiveness": tuning.get("aggressiveness", "balanced"),
            "accuracy_tier": tuning.get("accuracy_tier", "standard"),
        }
    return out


def _resolve_binary(explicit: object, candidates: list[str]) -> str | None:
    if isinstance(explicit, str) and explicit.strip():
        p = Path(explicit.strip()).expanduser()
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
        resolved = shutil.which(explicit.strip())
        if resolved:
            return resolved
        return None
    for cand in candidates:
        if cand.startswith("/"):
            if Path(cand).is_file() and os.access(cand, os.X_OK):
                return cand
        else:
            resolved = shutil.which(cand)
            if resolved:
                return resolved
    return None


def _resolve_gemini_api_key(ai_cfg: dict) -> str:
    direct = ai_cfg.get("gemini_api_key")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    env_name = ai_cfg.get("gemini_api_key_env") or "GEMINI_API_KEY"
    for name in (env_name, "GOOGLE_API_KEY"):
        if isinstance(name, str) and name:
            val = os.environ.get(name, "").strip()
            if val:
                return val
    return ""


def _parse_agy_models_output(output: str) -> list[dict]:
    models = []
    seen = set()
    cleaned = ANSI_ESCAPE_RE.sub("", output or "").replace("\r", "")
    for line in cleaned.splitlines():
        line = line.strip()
        m = AGY_LINE_RE.match(line)
        if not m:
            continue
        model_id = m.group(1)
        desc = m.group(2).strip()
        if "/" in model_id or "\\" in model_id or not MODEL_NAME_RE.fullmatch(model_id):
            continue
        if model_id in seen:
            continue
        seen.add(model_id)
        label_name = re.split(r"\s{2,}", desc)[0].strip() or model_id
        models.append({
            "id": f"antigravity:{model_id}",
            "provider": "antigravity",
            "model": model_id,
            "label": f"Antigravity · {label_name}"[:100],
            "is_default": False,
        })
    return models


def discover_worker_ai_profile(config: dict) -> dict:
    """Discover local AI providers/models advertised by this worker when AI is enabled."""
    empty = {
        "capabilities_extra": [],
        "ai_providers": [],
        "ai_models": [],
        "default_ai_model": None,
        "fallback_models": [],
        "provider_runtime": {},
    }
    ai_cfg = config.get("ai")
    if not isinstance(ai_cfg, dict) or ai_cfg.get("enabled") is not True:
        return empty

    providers_req = ai_cfg.get("providers", ["antigravity", "gemini", "codex"])
    providers_req = [p for p in providers_req if isinstance(p, str) and p in AI_PROVIDERS]

    discovered_providers: list[str] = []
    models: list[dict] = []
    provider_runtime: dict = {}

    if "antigravity" in providers_req:
        agy_bin = _resolve_binary(ai_cfg.get("antigravity_bin"), ["agy", "/opt/homebrew/bin/agy", "/usr/local/bin/agy"])
        if agy_bin:
            try:
                proc = subprocess.run([agy_bin, "models"], capture_output=True, text=True, timeout=8, check=False)
                if proc.returncode == 0:
                    agy_models = _parse_agy_models_output(proc.stdout)
                    if agy_models:
                        discovered_providers.append("antigravity")
                        models.extend(agy_models)
                        provider_runtime["antigravity_bin"] = agy_bin
            except (OSError, subprocess.TimeoutExpired):
                pass

    if "gemini" in providers_req:
        api_key = _resolve_gemini_api_key(ai_cfg)
        raw_gemini = ai_cfg.get("gemini_models", ["gemini-2.5-flash", "gemini-2.5-flash-lite"])
        gem_models = []
        for item in raw_gemini or []:
            if isinstance(item, str) and MODEL_NAME_RE.fullmatch(item):
                gem_models.append({
                    "id": f"gemini:{item}",
                    "provider": "gemini",
                    "model": item,
                    "label": f"Gemini · {item}"[:100],
                    "is_default": False,
                })
        if api_key and gem_models:
            discovered_providers.append("gemini")
            models.extend(gem_models)
            provider_runtime["gemini_api_key_configured"] = True

    if "codex" in providers_req:
        codex_bin = _resolve_binary(
            ai_cfg.get("codex_bin"),
            ["codex", "/Applications/ChatGPT.app/Contents/Resources/codex-cli/CodexCLI.app/Contents/MacOS/codex"],
        )
        if codex_bin:
            try:
                proc = subprocess.run([codex_bin, "login", "status"], capture_output=True, text=True, timeout=8, check=False)
                status_out = f"{proc.stdout}\n{proc.stderr}"
                if proc.returncode == 0 and "ChatGPT" in status_out:
                    raw_codex = ai_cfg.get("codex_models", ["default"])
                    cx_models = []
                    for item in raw_codex or []:
                        if isinstance(item, str) and MODEL_NAME_RE.fullmatch(item):
                            cx_models.append({
                                "id": f"codex:{item}",
                                "provider": "codex",
                                "model": item,
                                "label": f"Codex · {item}"[:100],
                                "is_default": False,
                            })
                    if cx_models:
                        discovered_providers.append("codex")
                        models.extend(cx_models)
                        provider_runtime["codex_bin"] = codex_bin
            except (OSError, subprocess.TimeoutExpired):
                pass

    dedup: list[dict] = []
    seen_ids: set[str] = set()
    for item in models:
        if item["id"] not in seen_ids:
            seen_ids.add(item["id"])
            dedup.append(item)
        if len(dedup) >= 50:
            break

    if not dedup:
        return empty

    configured_default = ai_cfg.get("default_model")
    default_id = configured_default if isinstance(configured_default, str) and configured_default in seen_ids else None
    if default_id is None:
        preferred = [m["id"] for m in dedup if "flash" in m["model"] or "hydrogen" in m["model"] or m["model"].endswith("-low")]
        default_id = preferred[0] if preferred else dedup[0]["id"]
    for item in dedup:
        item["is_default"] = item["id"] == default_id

    fallback_models = [
        m for m in (ai_cfg.get("fallback_models") or []) if isinstance(m, str) and m in seen_ids
    ]
    return {
        "capabilities_extra": [VERSION],
        "ai_providers": discovered_providers,
        "ai_models": dedup,
        "default_ai_model": default_id,
        "fallback_models": fallback_models,
        "provider_runtime": provider_runtime,
    }


def effective_detector_config(base_cfg: dict, ai_tuning: dict, explicit_keys: set[str] | None = None) -> tuple[dict, dict]:
    """Scale detector proposal and assembly settings by aggressiveness unless explicitly overridden."""
    explicit = set(explicit_keys or set())
    agg = str((ai_tuning or {}).get("aggressiveness", "balanced"))
    tier = str((ai_tuning or {}).get("accuracy_tier", "standard"))
    if agg not in AI_AGGRESSIVENESS_LEVELS:
        raise ValueError("invalid_ai_tuning")
    if tier not in AI_ACCURACY_TIERS:
        raise ValueError("invalid_ai_tuning")

    out = dict(base_cfg)
    swing = float(base_cfg.get("swing_speed", 2.0))
    reach = float(base_cfg.get("arm_reach", 0.85))
    height = float(base_cfg.get("min_player_height", 0.12))
    pad_before = float(base_cfg.get("padding_before", 2.0))
    pad_after = float(base_cfg.get("padding_after", 1.5))
    merge_gap = float(base_cfg.get("merge_gap", 0.5))

    if agg == "conservative":
        if "swing_speed" not in explicit:
            out["swing_speed"] = round(swing * 0.80, 6)
        if "arm_reach" not in explicit:
            out["arm_reach"] = round(reach * 0.92, 6)
        if "min_player_height" not in explicit:
            out["min_player_height"] = round(height * 0.85, 6)
        if "merge_gap" not in explicit:
            out["merge_gap"] = max(merge_gap, 1.0)
        if "padding_before" not in explicit:
            out["padding_before"] = pad_before
        if "padding_after" not in explicit:
            out["padding_after"] = pad_after
    elif agg == "aggressive":
        if "padding_before" not in explicit:
            out["padding_before"] = round(pad_before * 0.80, 6)
        if "padding_after" not in explicit:
            out["padding_after"] = round(pad_after * 0.80, 6)

    return out, {"aggressiveness": agg, "accuracy_tier": tier}


def _pose_extractor_digest() -> str:
    root = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for name in ("detector.py", "pipeline.py"):
        p = root / name
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()


def _prune_pose_cache(cache_dir: Path, keep: int = MAX_POSE_CACHE_ENTRIES) -> None:
    if not cache_dir.is_dir():
        return
    entries = []
    for p in cache_dir.glob("*.json"):
        try:
            entries.append((p.stat().st_mtime, p))
        except OSError:
            continue
    entries.sort(key=lambda item: item[0], reverse=True)
    for _, old in entries[keep:]:
        try:
            old.unlink()
        except OSError:
            pass


def _try_load_pose_cache(path: Path, expected_key: dict) -> tuple[list[dict], dict, float] | None:
    if not path.is_file() or path.is_symlink():
        return None
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(envelope, dict) or envelope.get("key") != expected_key:
            return None
        raw_records = envelope.get("records")
        proxy_meta = envelope.get("proxy_meta")
        duration = envelope.get("duration")
        if (
            not isinstance(raw_records, list)
            or not isinstance(proxy_meta, dict)
            or isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
        ):
            return None
        if _canonical_sha256(raw_records) != envelope.get("payload_sha256"):
            return None
        import detector
        records = detector.records_from_jsonable(raw_records)
        return records, proxy_meta, float(duration)
    except Exception:
        return None


def load_or_extract_pose_records(proxy: Path, meta: dict, cfg: dict) -> tuple[list[dict], dict, float, bool]:
    """Load verified raw pose records from task/shared cache or run raw CoreML extraction."""
    import detector

    proxy = Path(proxy)
    scratch = proxy.parent
    runtime = cfg.get("_runtime") if isinstance(cfg.get("_runtime"), dict) else {}
    cache_key_obj = {
        "cache_schema": "pose_records_v1",
        "source_sha256": cfg.get("_source_sha256"),
        "proxy_sha256": _file_sha256(proxy) if proxy.is_file() else None,
        "model_sha256": runtime.get("model_sha256"),
        "backend": str(cfg.get("pose_device", "coreml")),
        "extractor_code_digest": _pose_extractor_digest(),
    }
    cache_key_hex = _canonical_sha256(cache_key_obj)

    local_cache = scratch / "pose-records-cache.json"
    state_dir = cfg.get("_state_dir")
    shared_cache = Path(state_dir) / "cache" / "pose" / f"{cache_key_hex}.json" if state_dir else None

    loaded = _try_load_pose_cache(local_cache, cache_key_obj)
    if loaded is not None:
        return loaded[0], loaded[1], loaded[2], True

    if shared_cache is not None:
        loaded = _try_load_pose_cache(shared_cache, cache_key_obj)
        if loaded is not None:
            try:
                os.utime(shared_cache, None)
            except OSError:
                pass
            try:
                shutil.copyfile(shared_cache, local_cache)
            except OSError:
                pass
            return loaded[0], loaded[1], loaded[2], True

    records, proxy_meta, duration = detector.extract_pose_records(proxy, cfg)
    raw_records = detector.records_to_jsonable(records)
    envelope = {
        "key": cache_key_obj,
        "payload_sha256": _canonical_sha256(raw_records),
        "proxy_meta": proxy_meta,
        "duration": float(duration),
        "records": raw_records,
    }
    _atomic_json(local_cache, envelope)
    if shared_cache is not None and len(_canonical_json(envelope).encode("utf-8")) <= MAX_POSE_CACHE_BYTES:
        _atomic_json(shared_cache, envelope)
        _prune_pose_cache(shared_cache.parent, keep=MAX_POSE_CACHE_ENTRIES)
    return records, proxy_meta, float(duration), False


def _snap_grid(val: float, low: float, high: float) -> float:
    snapped = round(val / FRAME_GRID_SECONDS) * FRAME_GRID_SECONDS
    return max(low, min(high, round(snapped, 6)))


def _wrist_speed_series(records: list[dict], win_start: float, win_end: float) -> list[tuple[float, float]]:
    """Compute per-frame wrist speed values inside [win_start - 1.5, win_end + 1.5]."""
    if win_end <= win_start:
        return []
    pts: list[tuple[float, float]] = []
    prev = None
    for r in records:
        t = float(r["t"])
        if t < win_start - 1.5 or t > win_end + 1.5:
            continue
        b = np.asarray(r["boxes"])
        k, kc = np.asarray(r["k"]), np.asarray(r["kc"])
        if len(b) == 0:
            pts.append((t, 0.0))
            prev = None
            continue
        j = int(np.argmax(b[:, 3] - b[:, 1]))
        kp, conf = k[j], kc[j]
        if len(conf) < 13 or float(np.min(conf[[5, 6, 11, 12]])) < 0.25:
            pts.append((t, 0.0))
            prev = None
            continue
        shoulder = (kp[5] + kp[6]) / 2.0
        hip = (kp[11] + kp[12]) / 2.0
        torso = max(1.0, float(np.linalg.norm(hip - shoulder)))
        wrists = (kp[[9, 10]] - shoulder) / torso
        valid = conf[[9, 10]] > 0.35
        spd = 0.0
        if prev is not None and 0 < t - prev[0] < 0.4:
            good = valid & prev[2]
            if good.any():
                spd = float(np.linalg.norm(wrists - prev[1], axis=1)[good].max() / (t - prev[0]))
        pts.append((t, spd))
        prev = (t, wrists, valid)
    return pts


def _rolling_wrist_valley(records: list[dict], win_start: float, win_end: float) -> float | None:
    """Find the lowest 1-second rolling wrist displacement speed inside [win_start, win_end]."""
    pts = _wrist_speed_series(records, win_start, win_end)
    best_t = None
    best_score = float("inf")
    for t, _ in pts:
        if t < win_start or t > win_end:
            continue
        window_vals = [s for pt_t, s in pts if t - 0.5 <= pt_t <= t + 0.5]
        avg_spd = sum(window_vals) / max(1, len(window_vals))
        if avg_spd < best_score:
            best_score = avg_spd
            best_t = t
    return best_t


def _candidate_stats(start_s: float, end_s: float, events: list[list[float]], records: list[dict]) -> dict:
    in_events = [e for e in events if e[1] >= start_s and e[0] <= end_s]
    total_frames = 0
    person_frames = 0
    for r in records:
        t = float(r["t"])
        if start_s <= t <= end_s:
            total_frames += 1
            if len(r["boxes"]) > 0:
                person_frames += 1
    pts = _wrist_speed_series(records, start_s, end_s)
    in_window_speeds = [s for t, s in pts if start_s <= t <= end_s]
    peak_speed = max(in_window_speeds) if in_window_speeds else 0.0
    valley_t = _rolling_wrist_valley(records, start_s, end_s)
    return {
        "event_count": len(in_events),
        "active_ratio": round(person_frames / max(1, total_frames), 3),
        "peak_wrist_speed": round(float(peak_speed), 3),
        "min_motion_t": round(valley_t, 3) if valley_t is not None else None,
    }


def _split_span_pass1(start_s: float, end_s: float, events: list[list[float]], max_seconds: float) -> list[tuple[float, float]]:
    """Split >45s spans cleanly along internal >= 2.0s low-activity event gaps."""
    if end_s - start_s <= max_seconds:
        return [(start_s, end_s)]
    inside = sorted(
        [max(start_s, float(e[0])), min(end_s, float(e[1]))]
        for e in events
        if float(e[1]) >= start_s and float(e[0]) <= end_s and min(end_s, float(e[1])) >= max(start_s, float(e[0]))
    )
    target = (start_s + end_s) / 2.0
    best_mid = None
    best_dist = float("inf")
    for idx in range(len(inside) - 1):
        gap = inside[idx + 1][0] - inside[idx][1]
        if gap >= MIN_VALLEY_GAP_SECONDS:
            mid = _snap_grid((inside[idx][1] + inside[idx + 1][0]) / 2.0, start_s + 1.0, end_s - 1.0)
            if start_s + 1.0 <= mid <= end_s - 1.0:
                dist = abs(mid - target)
                if dist < best_dist:
                    best_dist = dist
                    best_mid = mid
    if best_mid is None:
        return [(start_s, end_s)]
    return _split_span_pass1(start_s, best_mid, events, max_seconds) + _split_span_pass1(best_mid, end_s, events, max_seconds)


def _split_span_pass2(start_s: float, end_s: float, records: list[dict], max_seconds: float) -> list[tuple[float, float]]:
    """Hard-bound any remaining >45s span using rolling wrist-speed valleys or uniform cuts."""
    out: list[tuple[float, float]] = []
    cur = start_s
    while end_s - cur > max_seconds:
        win_start = cur + 25.0
        win_end = min(end_s - 10.0, cur + max_seconds)
        cut = None
        if win_end > win_start:
            valley = _rolling_wrist_valley(records, win_start, win_end)
            if valley is not None:
                cand_cut = _snap_grid(valley, cur + 1.0, min(cur + max_seconds, end_s - 1.0))
                if cur < cand_cut <= cur + max_seconds and cand_cut < end_s:
                    cut = cand_cut
        if cut is None:
            remaining = end_s - cur
            # Balance last two halves cleanly when <= 90s so tail isn't tiny.
            raw_cut = cur + (remaining / 2.0 if remaining <= max_seconds * 1.6 else max_seconds)
            cut = _snap_grid(raw_cut, cur + 1.0, min(cur + max_seconds, end_s - 0.5))
            if cut - cur > max_seconds or cut <= cur or cut >= end_s:
                cut = min(cur + max_seconds, round((cur + end_s) / 2.0, 6))
        out.append((round(cur, 6), round(cut, 6)))
        cur = round(cut, 6)
    if end_s > cur:
        out.append((round(cur, 6), round(end_s, 6)))
    return out


def build_bounded_candidates(
    intervals: list[list[float]],
    events: list[list[float]],
    records: list[dict],
    duration: float,
    max_seconds: float = MAX_CANDIDATE_SECONDS,
) -> list[dict]:
    """Produce ordered non-overlapping candidates strictly satisfying 0 < duration_s <= max_seconds."""
    spans: list[tuple[float, float]] = []
    for raw_a, raw_b in sorted(intervals):
        a = max(0.0, float(raw_a))
        b = min(float(duration), float(raw_b))
        if b <= a:
            continue
        for p1_a, p1_b in _split_span_pass1(a, b, events, max_seconds):
            spans.extend(_split_span_pass2(p1_a, p1_b, records, max_seconds))

    candidates = []
    for idx, (a, b) in enumerate(spans, 1):
        stats = _candidate_stats(a, b, events, records)
        candidates.append({
            "candidate_id": f"C{idx:03d}",
            "start_s": round(a, 6),
            "end_s": round(b, 6),
            "duration_s": round(b - a, 6),
            "event_count": stats["event_count"],
            "peak_wrist_speed": stats["peak_wrist_speed"],
            "active_ratio": stats["active_ratio"],
        })
    return candidates


def _sampling_grid_for_candidate(duration_s: float, tier: str) -> tuple[int, int, int, int]:
    spec = TIER_CONFIG[tier]
    if "long_span_s" in spec and duration_s > spec["long_span_s"]:
        return spec["long_frames"], spec["long_cols"], spec["long_rows"], spec["tile_w"]
    return spec["frames"], spec["cols"], spec["rows"], spec["tile_w"]


def render_contact_sheets(
    proxy: Path,
    candidates: list[dict],
    work_dir: Path,
    accuracy_tier: str,
    guard=None,
) -> list[dict]:
    """Decode sampled proxy frames in a single pass and render EXIF-free JPEG contact sheets."""
    work_dir.mkdir(parents=True, exist_ok=True)
    if not candidates:
        return []

    sample_targets: list[tuple[float, int, int]] = []
    cand_layouts: dict[int, dict] = {}
    for c_idx, cand in enumerate(candidates):
        n_frames, cols, rows, tile_w = _sampling_grid_for_candidate(cand["duration_s"], accuracy_tier)
        start_s, end_s = cand["start_s"], cand["end_s"]
        if n_frames <= 1:
            ts_list = [(start_s + end_s) / 2.0]
        else:
            step = (end_s - start_s) / n_frames
            ts_list = [round(start_s + (i + 0.5) * step, 4) for i in range(n_frames)]
        cand_layouts[c_idx] = {
            "cols": cols,
            "rows": rows,
            "tile_w": tile_w,
            "timestamps": ts_list,
            "frames": [None] * len(ts_list),
            "dists": [float("inf")] * len(ts_list),
        }
        for t_idx, target_t in enumerate(ts_list):
            sample_targets.append((target_t, c_idx, t_idx))

    sample_targets.sort(key=lambda item: item[0])
    aspect_h = None

    with av.open(str(proxy)) as container:
        stream = container.streams.video[0]
        aspect_h = max(2, int(round(stream.height * (cand_layouts[0]["tile_w"] / max(1, stream.width)) / 2.0)) * 2)
        active_start = 0
        for frame in container.decode(stream):
            if guard:
                guard()
            ft = float(frame.time if frame.time is not None else 0.0)
            while active_start < len(sample_targets) and sample_targets[active_start][0] < ft - 2.0:
                active_start += 1
            if active_start >= len(sample_targets):
                break
            img = None
            scan_idx = active_start
            while scan_idx < len(sample_targets) and sample_targets[scan_idx][0] <= ft + 2.0:
                target_t, c_idx, t_idx = sample_targets[scan_idx]
                dist = abs(ft - target_t)
                layout = cand_layouts[c_idx]
                if dist < layout["dists"][t_idx]:
                    if img is None:
                        img = frame.to_image().convert("RGB")
                    layout["dists"][t_idx] = dist
                    layout["frames"][t_idx] = img
                scan_idx += 1

    rendered: list[dict] = []
    for c_idx, cand in enumerate(candidates):
        if guard:
            guard()
        layout = cand_layouts[c_idx]
        cols, rows, tile_w = layout["cols"], layout["rows"], layout["tile_w"]
        tile_h = aspect_h or int(round(tile_w * 9 / 16))
        banner_h = 28
        sheet = Image.new("RGB", (cols * tile_w, banner_h + rows * tile_h), (18, 22, 24))
        draw = ImageDraw.Draw(sheet)
        header = (
            f"{cand['candidate_id']}  {cand['start_s']:.1f}s-{cand['end_s']:.1f}s "
            f"({cand['duration_s']:.1f}s)  ev={cand['event_count']} act={cand['active_ratio']:.0%}"
        )
        draw.text((8, 7), header, fill=(235, 242, 238))
        fallback_tile = Image.new("RGB", (tile_w, tile_h), (32, 36, 40))
        for t_idx, ts in enumerate(layout["timestamps"]):
            col = t_idx % cols
            row = t_idx // cols
            x = col * tile_w
            y = banner_h + row * tile_h
            tile_src = layout["frames"][t_idx]
            tile = tile_src.resize((tile_w, tile_h), Image.Resampling.BILINEAR) if tile_src is not None else fallback_tile
            sheet.paste(tile, (x, y))
            draw.rectangle([x + 4, y + 4, x + 76, y + 20], fill=(0, 0, 0))
            draw.text((x + 8, y + 6), f"t={ts:.1f}s", fill=(255, 240, 140))

        out_path = work_dir / f"{cand['candidate_id']}.jpg"
        sheet.save(out_path, format="JPEG", quality=82)
        rendered.append({
            **cand,
            "sheet_path": str(out_path),
            "sheet_sha256": _file_sha256(out_path),
            "tile_count": len(layout["timestamps"]),
        })
    return rendered


def _terminate_process_group(pgid: int, process: subprocess.Popen, *, force_abort: bool) -> None:
    """Synchronously terminate any remaining child/descendant processes in pgid before returning."""
    try:
        os.killpg(pgid, 0)
        group_alive = True
    except (ProcessLookupError, OSError):
        group_alive = False

    if group_alive or process.poll() is None:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
        deadline = time.monotonic() + (5.0 if force_abort else 0.2)
        while time.monotonic() < deadline:
            try:
                os.killpg(pgid, 0)
            except (ProcessLookupError, OSError):
                break
            time.sleep(0.05)
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass

    if process.poll() is None:
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
                process.wait(timeout=1.0)
            except (OSError, subprocess.TimeoutExpired):
                pass


def _close_stream(stream) -> None:
    if stream is None:
        return
    try:
        stream.close()
    except (OSError, ValueError):
        pass


def _run_guarded_subprocess(
    cmd: list[str],
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    stdin_bytes: bytes | None = None,
    timeout_s: float = 180.0,
    guard=None,
) -> tuple[int, bytes, bytes]:
    """Run subprocess in isolated session with non-blocking pipe I/O and unconditional pgid cleanup."""
    if guard:
        guard()

    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        start_new_session=True,
        stdin=subprocess.PIPE if stdin_bytes is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    pgid = process.pid
    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []
    stdout_len = 0
    stderr_len = 0
    io_threads: list[threading.Thread] = []
    abort_exception: BaseException | None = None

    def _write_stdin():
        try:
            if process.stdin is not None and stdin_bytes is not None:
                chunk_size = 65536
                view = memoryview(stdin_bytes)
                for offset in range(0, len(view), chunk_size):
                    process.stdin.write(view[offset : offset + chunk_size])
                process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass
        finally:
            _close_stream(process.stdin)

    def _read_pipe(pipe, chunks: list[bytes], limit: int):
        nonlocal stdout_len, stderr_len
        captured = 0
        try:
            if pipe is None:
                return
            while True:
                data = pipe.read(65536)
                if not data:
                    break
                if captured < limit:
                    take = data[: limit - captured]
                    chunks.append(take)
                    captured += len(take)
        except (OSError, ValueError):
            pass

    if stdin_bytes is not None:
        tw = threading.Thread(target=_write_stdin, daemon=True)
        tw.start()
        io_threads.append(tw)

    to = threading.Thread(target=_read_pipe, args=(process.stdout, stdout_chunks, MAX_PROVIDER_OUTPUT_BYTES), daemon=True)
    te = threading.Thread(target=_read_pipe, args=(process.stderr, stderr_chunks, MAX_PROVIDER_OUTPUT_BYTES), daemon=True)
    to.start()
    te.start()
    io_threads.extend((to, te))

    try:
        started = time.monotonic()
        while True:
            if guard:
                guard()
            if time.monotonic() - started > timeout_s:
                raise ProviderCallError("provider_timeout", f"AI provider timed out after {timeout_s:.0f}s")
            rc = process.poll()
            if rc is not None:
                break
            time.sleep(0.1)
    except BaseException as exc:
        abort_exception = exc
        raise
    finally:
        # Strict cleanup order: terminate pgid -> close pipes -> join I/O threads -> preserve active exception.
        _terminate_process_group(pgid, process, force_abort=abort_exception is not None)
        _close_stream(process.stdin)
        _close_stream(process.stdout)
        _close_stream(process.stderr)
        for t in io_threads:
            t.join(timeout=1.0)

    return process.returncode or 0, b"".join(stdout_chunks), b"".join(stderr_chunks)


def _gemini_http_entrypoint() -> int:
    """Child process entrypoint for terminable Gemini HTTPS requests (secrets strictly over stdin)."""
    try:
        raw_req = sys.stdin.buffer.read()
        req_data = json.loads(raw_req.decode("utf-8"))
        url = str(req_data["url"])
        api_key = str(req_data["api_key"])
        payload = req_data["payload"]
        timeout = float(req_data.get("timeout", 180.0))
        body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            resp_bytes = response.read(MAX_PROVIDER_OUTPUT_BYTES)
        sys.stdout.buffer.write(resp_bytes)
        sys.stdout.buffer.flush()
        return 0
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read(512).decode("utf-8", errors="replace")
        except Exception:
            pass
        sys.stderr.write(sanitize_error_message(f"HTTP {exc.code}: {detail}"))
        return 2
    except Exception as exc:
        sys.stderr.write(sanitize_error_message(str(exc)))
        return 1


def _build_batch_prompt(batch: list[dict], batch_dir: Path, tier: str, include_file_names: bool) -> str:
    manifest = [
        {
            "candidate_id": c["candidate_id"],
            "file": f"{c['candidate_id']}.jpg" if include_file_names else None,
            "start_s": c["start_s"],
            "end_s": c["end_s"],
            "duration_s": c["duration_s"],
            "event_count": c["event_count"],
            "peak_wrist_speed": c.get("peak_wrist_speed", 0.0),
            "active_ratio": c["active_ratio"],
        }
        for c in batch
    ]
    lines = [
        "You are inspecting local tennis practice contact sheets extracted from pose-detected candidate intervals.",
        f"Inspection tier: {tier}.",
        "Classify every candidate interval independently using these labels:",
        "- categories: rally, serve, drill, warmup, shadow_swing, ball_pickup, walking, idle, uncertain",
        "- action: keep (tennis play fills the candidate), trim (part is real tennis play and part is dead time/walking/pickup; set trimmed_start_s and trimmed_end_s inside candidate bounds), drop (no actual stroke/rally/serve)",
        "Only use category=uncertain if frames are ambiguous; do not invent strokes you cannot see.",
        "Return ONLY valid JSON matching the required schema with one entry per candidate_id.",
        "Candidates JSON:",
        json.dumps(manifest, separators=(",", ":"), allow_nan=False),
    ]
    return "\n".join(lines)


def _extract_json_object(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned.strip())
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first != -1 and last > first:
        parsed = json.loads(cleaned[first : last + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ProviderCallError("invalid_ai_response_json", "AI provider returned non-JSON output")


def _model_supports_effort(model_name: str) -> bool:
    return not bool(re.search(r"-(low|medium|high|xhigh|max)$", model_name))


def _invoke_provider_batch(
    model_id: str,
    batch: list[dict],
    tier: str,
    cfg: dict,
    ai_cfg: dict,
    provider_runtime: dict,
    guard=None,
) -> tuple[dict, str, str]:
    """Execute one candidate batch against the target AI model using an isolated temp dir."""
    prompt_manifest = [
        {
            k: c.get(k, 0.0 if k == "peak_wrist_speed" else None)
            for k in ("candidate_id", "start_s", "end_s", "duration_s", "event_count", "peak_wrist_speed", "active_ratio", "sheet_sha256")
        }
        for c in batch
    ]
    input_manifest_sha256 = _canonical_sha256(prompt_manifest)

    custom_invoker = cfg.get("_ai_provider_invoke")
    timeout_s = float(ai_cfg.get("request_timeout_seconds", 180.0))

    with tempfile.TemporaryDirectory(prefix="hybrid-ai-batch-") as tmp_root:
        batch_dir = Path(tmp_root)
        for cand in batch:
            shutil.copyfile(cand["sheet_path"], batch_dir / f"{cand['candidate_id']}.jpg")

        provider, model_name = model_id.split(":", 1)
        prompt_text = _build_batch_prompt(batch, batch_dir, tier, include_file_names=(provider != "gemini"))
        prompt_sha256 = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()

        if custom_invoker is not None:
            if guard:
                guard()
            resp = custom_invoker(
                model_id=model_id,
                batch=copy.deepcopy(prompt_manifest),
                tier=tier,
                prompt=prompt_text,
                batch_dir=batch_dir,
            )
            if not isinstance(resp, dict):
                raise ProviderCallError("invalid_ai_response_json", "Custom provider returned non-dict response")
            return resp, prompt_sha256, input_manifest_sha256

        if provider == "antigravity":
            agy_bin = provider_runtime.get("antigravity_bin") or _resolve_binary(
                ai_cfg.get("antigravity_bin"), ["agy", "/opt/homebrew/bin/agy", "/usr/local/bin/agy"]
            )
            if not agy_bin:
                raise ProviderCallError("provider_unavailable", "Antigravity CLI binary unavailable")
            prompt_parts = [prompt_text, "Inspect attached image files: " + ", ".join(f"{c['candidate_id']}.jpg" for c in batch)]
            cmd = [agy_bin, "agent", "--model", model_name]
            if _model_supports_effort(model_name):
                cmd.extend(["--effort", TIER_CONFIG[tier]["effort"]])
            cmd.extend(["--", "\n".join(prompt_parts)])
            rc, stdout_b, stderr_b = _run_guarded_subprocess(cmd, cwd=str(batch_dir), timeout_s=timeout_s, guard=guard)
            if rc != 0:
                raise ProviderCallError("provider_cli_failed", sanitize_error_message(stderr_b.decode("utf-8", errors="replace") or f"exit {rc}"))
            parsed = _extract_json_object(stdout_b.decode("utf-8", errors="replace"))
            return parsed, prompt_sha256, input_manifest_sha256

        if provider == "codex":
            codex_bin = provider_runtime.get("codex_bin") or _resolve_binary(
                ai_cfg.get("codex_bin"),
                ["codex", "/Applications/ChatGPT.app/Contents/Resources/codex-cli/CodexCLI.app/Contents/MacOS/codex"],
            )
            if not codex_bin:
                raise ProviderCallError("provider_unavailable", "Codex CLI binary unavailable")
            schema_file = batch_dir / "decision-schema.json"
            out_file = batch_dir / "decision-output.json"
            schema_file.write_text(_canonical_json(DECISION_JSON_SCHEMA), encoding="utf-8")
            cmd = [
                codex_bin,
                "exec",
                "--ephemeral",
                "--skip-git-repo-check",
                "-s",
                "read-only",
                "--color",
                "never",
                "--output-schema",
                str(schema_file),
                "-o",
                str(out_file),
            ]
            if model_name != "default":
                cmd.extend(["--model", model_name])
            for cand in batch:
                cmd.extend(["-i", str(batch_dir / f"{cand['candidate_id']}.jpg")])
            cmd.append(prompt_text)
            rc, stdout_b, stderr_b = _run_guarded_subprocess(cmd, cwd=str(batch_dir), timeout_s=timeout_s, guard=guard)
            if rc != 0:
                raise ProviderCallError("provider_cli_failed", sanitize_error_message(stderr_b.decode("utf-8", errors="replace") or f"exit {rc}"))
            raw_text = out_file.read_text(encoding="utf-8") if out_file.is_file() else stdout_b.decode("utf-8", errors="replace")
            return _extract_json_object(raw_text), prompt_sha256, input_manifest_sha256

        if provider == "gemini":
            api_key = _resolve_gemini_api_key(ai_cfg)
            if not api_key:
                raise ProviderCallError("provider_unavailable", "Gemini API key missing")
            parts: list[dict] = [{"text": prompt_text}]
            for cand in batch:
                img_bytes = (batch_dir / f"{cand['candidate_id']}.jpg").read_bytes()
                parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64.b64encode(img_bytes).decode("ascii"),
                    }
                })
            url = GEMINI_ENDPOINT_TEMPLATE.format(model=model_name)
            payload = {
                "contents": [{"parts": parts}],
                "generationConfig": {
                    "response_mime_type": "application/json",
                    "response_schema": DECISION_JSON_SCHEMA,
                    "temperature": 0.1,
                },
            }
            req_bytes = _canonical_json({
                "url": url,
                "api_key": api_key,
                "payload": payload,
                "timeout": timeout_s,
            }).encode("utf-8")
            rc, stdout_b, stderr_b = _run_guarded_subprocess(
                [sys.executable, str(Path(__file__).resolve()), "--gemini-http"],
                cwd=str(batch_dir),
                stdin_bytes=req_bytes,
                timeout_s=timeout_s,
                guard=guard,
            )
            if rc != 0:
                raise ProviderCallError("provider_http_failed", sanitize_error_message(stderr_b.decode("utf-8", errors="replace") or f"exit {rc}"))
            envelope = json.loads(stdout_b.decode("utf-8"))
            text_out = ""
            for cand_resp in envelope.get("candidates") or []:
                for part in (cand_resp.get("content") or {}).get("parts") or []:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        text_out += part["text"]
            return _extract_json_object(text_out), prompt_sha256, input_manifest_sha256

        raise ProviderCallError("unsupported_provider", f"Unsupported AI provider {provider}")


def _validate_and_parse_batch_decisions(batch: list[dict], response: dict, duration: float) -> dict[str, dict]:
    """Validate 1-to-1 candidate decisions and strict trim bounds before policy evaluation."""
    if not isinstance(response, dict) or not isinstance(response.get("decisions"), list):
        raise ProviderCallError("invalid_ai_response_schema", "Response must contain decisions array")

    cand_map = {c["candidate_id"]: c for c in batch}
    parsed: dict[str, dict] = {}
    for item in response["decisions"]:
        if not isinstance(item, dict):
            raise ProviderCallError("invalid_ai_response_schema", "Decision item must be an object")
        cid = item.get("candidate_id")
        if not isinstance(cid, str) or cid not in cand_map or cid in parsed:
            raise ProviderCallError("invalid_ai_candidate_mapping", "Duplicate or unknown candidate_id")
        action = item.get("action")
        category = item.get("category")
        conf = item.get("confidence")
        reason = item.get("reason")
        if (
            action not in ALLOWED_ACTIONS
            or category not in ALLOWED_CATEGORIES
            or isinstance(conf, bool)
            or not isinstance(conf, (int, float))
            or not math.isfinite(conf)
            or not 0.0 <= float(conf) <= 1.0
            or not isinstance(reason, str)
        ):
            raise ProviderCallError("invalid_ai_response_schema", "Malformed decision fields")

        cand = cand_map[cid]
        trimmed_span = None
        trim_too_short = False
        if action == "trim":
            ts = item.get("trimmed_start_s")
            te = item.get("trimmed_end_s")
            if (
                isinstance(ts, bool)
                or isinstance(te, bool)
                or not isinstance(ts, (int, float))
                or not isinstance(te, (int, float))
                or not math.isfinite(ts)
                or not math.isfinite(te)
            ):
                raise ProviderCallError("invalid_ai_trim_bounds", "Missing or non-finite trim bounds")
            ts_f, te_f = float(ts), float(te)
            if ts_f >= te_f:
                raise ProviderCallError("invalid_ai_trim_bounds", "Inverted trim bounds")
            if ts_f < cand["start_s"] - 2.0 or te_f > cand["end_s"] + 2.0:
                raise ProviderCallError("invalid_ai_trim_bounds", "Trim bounds exceed candidate margin")
            overlap_start = max(cand["start_s"], ts_f)
            overlap_end = min(cand["end_s"], te_f)
            if overlap_end <= overlap_start:
                raise ProviderCallError("invalid_ai_trim_bounds", "Trim bounds disjoint from candidate")
            snapped_start = max(cand["start_s"], max(0.0, math.floor(overlap_start / FRAME_GRID_SECONDS) * FRAME_GRID_SECONDS))
            snapped_end = min(cand["end_s"], min(float(duration), math.ceil(overlap_end / FRAME_GRID_SECONDS) * FRAME_GRID_SECONDS))
            snapped_start = round(snapped_start, 6)
            snapped_end = round(snapped_end, 6)
            if snapped_end - snapped_start < 1.0:
                trim_too_short = True
            else:
                trimmed_span = [snapped_start, snapped_end]

        parsed[cid] = {
            "candidate_id": cid,
            "action": action,
            "category": category,
            "confidence": round(float(conf), 4),
            "reason_code": sanitize_error_message(reason, max_len=120),
            "trimmed_span": trimmed_span,
            "trim_too_short": trim_too_short,
        }

    if set(parsed) != set(cand_map):
        raise ProviderCallError("invalid_ai_candidate_mapping", "AI omitted one or more candidate decisions")
    return parsed


def _apply_candidate_decision(cand: dict, dec: dict, aggressiveness: str) -> tuple[list[float] | None, str]:
    """Evaluate exact (aggressiveness x category x action x confidence) decision matrix (§2.2.C)."""
    full = [round(float(cand["start_s"]), 6), round(float(cand["end_s"]), 6)]
    action = dec["action"]
    category = dec["category"]
    conf = float(dec["confidence"])
    trimmed = dec["trimmed_span"]
    too_short = bool(dec["trim_too_short"])

    if aggressiveness == "conservative":
        if category == "uncertain":
            return full, ("kept" if action != "drop" else "conservative_retained_uncertain")
        if category in ("rally", "serve", "drill", "warmup", "shadow_swing"):
            return full, ("kept" if action != "drop" else "conservative_retained_low_conf_drop")
        # ball_pickup | walking | idle
        if action == "drop" and conf >= 0.85:
            return None, "dropped"
        return full, ("kept" if action != "drop" else "conservative_retained_low_conf_drop")

    if aggressiveness == "balanced":
        if category in ("rally", "serve", "drill", "warmup"):
            if action == "keep":
                return full, "kept"
            if action == "trim":
                return (None, "short_trimmed_span_dropped") if too_short else (trimmed, "trimmed")
            if conf >= 0.80:
                return None, "dropped"
            return full, "conservative_retained_low_conf_drop"
        if category == "shadow_swing":
            if action == "keep":
                return (full, "kept") if conf < 0.75 else (None, "dropped")
            if action == "trim":
                return (trimmed, "trimmed") if (conf < 0.75 and not too_short) else (None, "short_trimmed_span_dropped" if conf < 0.75 else "dropped")
            return None, "dropped"
        if category == "uncertain":
            if action == "keep":
                return full, "balanced_retained_uncertain"
            if action == "trim":
                return (None, "short_trimmed_span_dropped") if too_short else (trimmed, "trimmed")
            return (full, "balanced_retained_uncertain") if conf < 0.70 else (None, "dropped")
        # ball_pickup | walking | idle
        if action == "keep":
            return (None, "dropped") if conf >= 0.75 else (full, "kept")
        if action == "trim":
            if conf >= 0.75 or too_short:
                return None, ("short_trimmed_span_dropped" if (conf < 0.75 and too_short) else "dropped")
            return trimmed, "trimmed"
        return None, "dropped"

    # aggressive
    if category in ("rally", "serve", "drill"):
        if action == "keep":
            return full, "kept"
        if action == "trim":
            return (None, "short_trimmed_span_dropped") if too_short else (trimmed, "trimmed")
        return (None, "dropped") if conf >= 0.65 else (full, "conservative_retained_low_conf_drop")
    if category == "uncertain":
        if action == "keep" and conf < 0.50:
            return full, "kept"
        return None, "dropped"
    # warmup | shadow_swing | ball_pickup | walking | idle
    return None, "dropped"


def _build_model_chain(requested_model: str, allow_fallback: bool, profile: dict) -> list[str]:
    available_ids = [m["id"] for m in profile.get("ai_models") or []]
    available_set = set(available_ids)
    default_model = profile.get("default_ai_model")

    if not allow_fallback:
        target = default_model if requested_model == "auto" else requested_model
        if not target or target not in available_set:
            raise ProcessingTaskError("ai_model_unavailable", f"Requested model {requested_model} is unavailable on this worker.")
        return [target]

    chain: list[str] = []
    if requested_model != "auto" and requested_model in available_set:
        chain.append(requested_model)
    if default_model and default_model in available_set and default_model not in chain:
        chain.append(default_model)
    for fb in profile.get("fallback_models") or []:
        if fb in available_set and fb not in chain:
            chain.append(fb)
    for m_id in sorted(
        available_ids,
        key=lambda ident: (0 if ("flash" in ident or "hydrogen" in ident or ident.endswith("-low")) else 1, ident),
    ):
        if m_id not in chain:
            chain.append(m_id)
    if not chain:
        raise ProcessingTaskError("ai_model_unavailable", "No AI models discovered on this worker.")
    return chain


def bound_ai_refinement_findings(summary: dict, max_bytes: int = MAX_AI_REFINEMENT_FINDINGS_BYTES) -> dict:
    """Enforce serialized UTF-8 completion findings budget while keeping full provenance on disk."""
    out = copy.deepcopy({k: v for k, v in summary.items() if not str(k).startswith("_")})
    out["provenance_truncated"] = False
    batches = out.get("batch_provenance")
    if "batch_count_total" not in out:
        out["batch_count_total"] = len(batches) if isinstance(batches, list) else 0

    def encoded_len() -> int:
        return len(_canonical_json(out).encode("utf-8"))

    while encoded_len() > max_bytes:
        if isinstance(out.get("attempted_models"), list) and out["attempted_models"]:
            out["attempted_models"].pop()
            out["provenance_truncated"] = True
        elif isinstance(out.get("batch_provenance"), list) and out["batch_provenance"]:
            out["batch_provenance"].pop()
            out["provenance_truncated"] = True
        else:
            break
    return out


def run_hybrid_ai_analysis(proxy: Path, meta: dict, cfg: dict) -> tuple[list[list[float]], dict]:
    """Run local pose proposal, contact-sheet rendering, guarded AI filtering, and re-merge."""
    import detector

    guard = cfg.get("_guard")
    params = normalize_task_parameters(cfg.get("_task_parameters") or {
        k: v for k, v in cfg.items() if k in PARAMETERS
    })
    explicit_keys = set(cfg.get("_explicit_parameter_keys") or set())
    eff_cfg, ai_tuning = effective_detector_config(cfg, params["ai_tuning"], explicit_keys)

    records, proxy_meta, duration, raw_cache_hit = load_or_extract_pose_records(proxy, meta, eff_cfg)
    observations_path = Path(proxy).parent / "pose-observations.json"
    proposal_intervals, base_findings = detector.detect_from_records(
        records, proxy_meta, duration, eff_cfg, observations_path=observations_path
    )
    events = base_findings["event_intervals"]

    requested_model = params.get("ai_model", "auto")
    allow_fallback = bool(params.get("allow_model_fallback", True))
    tier = ai_tuning["accuracy_tier"]
    agg = ai_tuning["aggressiveness"]
    schema_sha256 = _canonical_sha256(DECISION_JSON_SCHEMA)

    candidates = build_bounded_candidates(proposal_intervals, events, records, duration, MAX_CANDIDATE_SECONDS)
    if not candidates:
        summary = {
            "detector": VERSION,
            "pose_backbone": detector.VERSION,
            "processing_strategy": "hybrid_ai_vision",
            "requested_ai_model": requested_model,
            "actual_ai_model": None,
            "actual_ai_models": [],
            "mixed_models_used": False,
            "allow_model_fallback": allow_fallback,
            "fallback_used": False,
            "ai_tuning": ai_tuning,
            "ai_call_count": 0,
            "candidate_count": 0,
            "kept_count": 0,
            "trimmed_count": 0,
            "dropped_count": 0,
            "uncertain_retained_count": 0,
            "uncertain_dropped_count": 0,
            "raw_pose_cache_hit": raw_cache_hit,
            "schema_sha256": schema_sha256,
            "batch_count_total": 0,
            "batch_provenance": [],
            "attempted_models": [],
        }
        bounded_summary = bound_ai_refinement_findings(summary)
        full_provenance = {**summary, "provenance_truncated": False, "candidate_decisions": []}
        return [], {
            **base_findings,
            "detector": VERSION,
            "pose_backbone": detector.VERSION,
            "effective_detector_settings": {
                k: eff_cfg.get(k) for k in ("swing_speed", "arm_reach", "min_player_height", "padding_before", "padding_after", "merge_gap") if k in eff_cfg
            },
            "ai_refinement": bounded_summary,
            "_ai_provenance_payload": full_provenance,
        }

    ai_profile = cfg.get("_ai_profile")
    if ai_profile is None:
        ai_profile = discover_worker_ai_profile({"ai": cfg.get("_ai_config") or {}})
    model_chain = _build_model_chain(requested_model, allow_fallback, ai_profile)
    ai_cfg = cfg.get("_ai_config") if isinstance(cfg.get("_ai_config"), dict) else {}
    provider_runtime = ai_profile.get("provider_runtime") if isinstance(ai_profile.get("provider_runtime"), dict) else {}

    work_dir = Path(proxy).parent / "ai-work"
    shutil.rmtree(work_dir, ignore_errors=True)
    rendered = render_contact_sheets(Path(proxy), candidates, work_dir, tier, guard=guard)

    batch_size = TIER_CONFIG[tier]["batch_size"]
    batches = [rendered[i : i + batch_size] for i in range(0, len(rendered), batch_size)]

    model_idx = 0
    ai_call_count = 0
    batch_provenance: list[dict] = []
    attempted_models: list[dict] = []
    accepted_models_order: list[str] = []
    all_decisions: dict[str, dict] = {}
    candidate_outcomes: list[dict] = []

    for b_num, batch in enumerate(batches, 1):
        batch_id = f"batch-{b_num:03d}"
        attempt_count = 0
        succeeded_model = None
        parsed_decisions = None
        prompt_sha = None
        input_sha = None

        while model_idx < len(model_chain):
            cand_model = model_chain[model_idx]
            attempt_count += 1
            ai_call_count += 1
            try:
                raw_resp, prompt_sha, input_sha = _invoke_provider_batch(
                    cand_model, batch, tier, cfg, ai_cfg, provider_runtime, guard=guard
                )
                parsed_decisions = _validate_and_parse_batch_decisions(batch, raw_resp, duration)
                succeeded_model = cand_model
                attempted_models.append({"model": cand_model, "batch_id": batch_id, "status": "succeeded"})
                break
            except ProviderCallError as exc:
                attempted_models.append({
                    "model": cand_model,
                    "batch_id": batch_id,
                    "status": "failed",
                    "error_code": exc.error_code,
                })
                if not allow_fallback:
                    raise ProcessingTaskError("ai_provider_failed", exc.message, retryable=False) from exc
                model_idx += 1

        if succeeded_model is None or parsed_decisions is None:
            raise ProcessingTaskError(
                "ai_all_models_failed",
                "All configured AI models failed to refine candidate batches.",
                retryable=False,
            )

        if succeeded_model not in accepted_models_order:
            accepted_models_order.append(succeeded_model)
        batch_provenance.append({
            "batch_id": batch_id,
            "candidate_ids": [c["candidate_id"] for c in batch],
            "accepted_model": succeeded_model,
            "attempt_count": attempt_count,
            "prompt_sha256": prompt_sha,
            "input_manifest_sha256": input_sha,
        })
        for c in batch:
            dec = parsed_decisions[c["candidate_id"]]
            dec["accepted_model"] = succeeded_model
            all_decisions[c["candidate_id"]] = dec

    kept_spans: list[list[float]] = []
    kept_count = 0
    trimmed_count = 0
    dropped_count = 0
    uncertain_retained = 0
    uncertain_dropped = 0

    for cand in candidates:
        dec = all_decisions[cand["candidate_id"]]
        span, outcome = _apply_candidate_decision(cand, dec, agg)
        if span is None:
            dropped_count += 1
            if dec["category"] == "uncertain":
                uncertain_dropped += 1
        elif outcome == "trimmed":
            trimmed_count += 1
            kept_spans.append(span)
        else:
            kept_count += 1
            if dec["category"] == "uncertain":
                uncertain_retained += 1
            kept_spans.append(span)
        candidate_outcomes.append({
            "candidate_id": cand["candidate_id"],
            "start_s": cand["start_s"],
            "end_s": cand["end_s"],
            "action": dec["action"],
            "category": dec["category"],
            "confidence": dec["confidence"],
            "applied_outcome": outcome,
            "applied_span": span,
            "accepted_model": dec["accepted_model"],
            "reason_code": dec["reason_code"],
        })

    # Merge adjacent retained/trimmed intervals smoothly across effective_merge_gap.
    eff_merge_gap = float(eff_cfg.get("merge_gap", 0.5))
    merged_intervals: list[list[float]] = []
    for a, b in sorted(kept_spans):
        a = max(0.0, float(a))
        b = min(float(duration), float(b))
        if b <= a:
            continue
        if merged_intervals and a <= merged_intervals[-1][1] + eff_merge_gap:
            merged_intervals[-1][1] = max(merged_intervals[-1][1], b)
        else:
            merged_intervals.append([round(a, 6), round(b, 6)])

    if len(accepted_models_order) == 0:
        actual_ai_model = None
    elif len(accepted_models_order) == 1:
        actual_ai_model = accepted_models_order[0]
    else:
        actual_ai_model = "mixed"

    fallback_used = (
        len(attempted_models) > len(batches)
        or (
            requested_model != "auto"
            and actual_ai_model is not None
            and actual_ai_model != requested_model
        )
    )

    summary = {
        "detector": VERSION,
        "pose_backbone": detector.VERSION,
        "processing_strategy": "hybrid_ai_vision",
        "requested_ai_model": requested_model,
        "actual_ai_model": actual_ai_model,
        "actual_ai_models": accepted_models_order,
        "mixed_models_used": len(accepted_models_order) > 1,
        "allow_model_fallback": allow_fallback,
        "fallback_used": fallback_used,
        "ai_tuning": ai_tuning,
        "ai_call_count": ai_call_count,
        "candidate_count": len(candidates),
        "kept_count": kept_count,
        "trimmed_count": trimmed_count,
        "dropped_count": dropped_count,
        "uncertain_retained_count": uncertain_retained,
        "uncertain_dropped_count": uncertain_dropped,
        "raw_pose_cache_hit": raw_cache_hit,
        "schema_sha256": schema_sha256,
        "batch_count_total": len(batch_provenance),
        "batch_provenance": batch_provenance,
        "attempted_models": attempted_models,
    }
    full_provenance = {
        **summary,
        "provenance_truncated": False,
        "candidate_decisions": candidate_outcomes,
    }
    bounded_summary = bound_ai_refinement_findings(summary)

    findings = {
        **base_findings,
        "detector": VERSION,
        "pose_backbone": detector.VERSION,
        "effective_detector_settings": {
            k: eff_cfg.get(k) for k in ("swing_speed", "arm_reach", "min_player_height", "padding_before", "padding_after", "merge_gap") if k in eff_cfg
        },
        "ai_refinement": bounded_summary,
        "_ai_provenance_payload": full_provenance,
    }
    return merged_intervals, findings


if __name__ == "__main__":
    if "--gemini-http" in sys.argv[1:]:
        raise SystemExit(_gemini_http_entrypoint())

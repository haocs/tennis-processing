"""Unit and real POSIX subprocess tests for hybrid AI + vision video processing."""
from __future__ import annotations

import hashlib
import http.server
import json
import os
from pathlib import Path
import socketserver
import sys
import threading
import time

import numpy as np
from PIL import Image
import pytest

import cut_constraints
import detector
import hybrid_ai
import pipeline
from publication import metadata_for_session, public_session


def _synthetic_pose_records(
    active_ranges: list[tuple[float, float]],
    duration: float = 10.0,
    fps: float = 8.0,
    width: int = 320,
    height: int = 180,
) -> tuple[list[dict], dict, float]:
    """Generate deterministic pose records with near-player wrist motion in active_ranges."""
    records = []
    n_frames = max(1, int(round(duration * fps)))
    dt = 1.0 / fps
    for idx in range(n_frames):
        t = round(idx * dt, 6)
        active = any(a <= t <= b for a, b in active_ranges)
        if not active:
            kp = np.zeros((1, 17, 2), dtype=np.float32)
            kc = np.zeros((1, 17), dtype=np.float32)
            # Still keep standing person visible so target_visible_fraction >= 0.10.
            kp[0, 5] = [140.0, 60.0]
            kp[0, 6] = [180.0, 60.0]
            kp[0, 11] = [145.0, 120.0]
            kp[0, 12] = [175.0, 120.0]
            kp[0, 9] = [135.0, 95.0]
            kp[0, 10] = [185.0, 95.0]
            kc[0, [5, 6, 9, 10, 11, 12]] = 0.9
        else:
            phase = 1.0 if (idx % 2 == 0) else -1.0
            kp = np.zeros((1, 17, 2), dtype=np.float32)
            kc = np.zeros((1, 17), dtype=np.float32)
            kp[0, 5] = [140.0, 60.0]
            kp[0, 6] = [180.0, 60.0]
            kp[0, 11] = [145.0, 120.0]
            kp[0, 12] = [175.0, 120.0]
            kp[0, 9] = [135.0, 90.0 + phase * 35.0]
            kp[0, 10] = [185.0 + phase * 30.0, 85.0]
            kc[0, [5, 6, 9, 10, 11, 12]] = 0.95
        boxes = np.array([[110.0, 40.0, 210.0, 150.0]], dtype=np.float32)
        records.append({"t": t, "boxes": boxes, "k": kp, "kc": kc})
    proxy_meta = {
        "width": width,
        "height": height,
        "target_fps": fps,
        "actual_fps": fps,
        "pts_timebase": "1/1000",
        "frame_count": len(records),
    }
    return records, proxy_meta, float(duration)


def _make_proxy_video(path: Path, duration: float, width: int = 320, height: int = 180, fps: int = 8) -> dict:
    log_path = path.parent / "ffmpeg-test.log"
    pipeline.run_media(
        [
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={width}x{height}:rate={fps}",
            "-t",
            f"{duration:.2f}",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        log_path,
    )
    return pipeline.probe(path)


def test_discover_worker_ai_profile_providers_and_disabled(tmp_path, monkeypatch):
    agy_script = tmp_path / "fake-agy"
    agy_script.write_text(
        "#!/bin/sh\n"
        "printf '\\033[32mgemini-3-flash-low   Gemini 3 Flash (Low)   default\\n'\n"
        "printf 'gemini-4-hydrogen    Gemini 4 Hydrogen\\033[0m\\n'\n"
    )
    agy_script.chmod(0o755)

    codex_script = tmp_path / "fake-codex"
    codex_script.write_text("#!/bin/sh\necho 'Logged in using ChatGPT'\n")
    codex_script.chmod(0o755)

    monkeypatch.setenv("SYNTHETIC_GEMINI_KEY", "AIzaSyntheticDiscoveryKey123456")

    disabled = hybrid_ai.discover_worker_ai_profile({"ai": {"enabled": False}})
    assert disabled["ai_models"] == []
    assert disabled["capabilities_extra"] == []

    profile = hybrid_ai.discover_worker_ai_profile(
        {
            "ai": {
                "enabled": True,
                "providers": ["antigravity", "gemini", "codex"],
                "antigravity_bin": str(agy_script),
                "codex_bin": str(codex_script),
                "gemini_api_key_env": "SYNTHETIC_GEMINI_KEY",
                "gemini_models": ["gemini-2.5-flash"],
                "codex_models": ["default"],
                "fallback_models": ["gemini:gemini-2.5-flash", "antigravity:nonexistent"],
            }
        }
    )
    assert profile["capabilities_extra"] == [hybrid_ai.VERSION]
    assert profile["ai_providers"] == ["antigravity", "gemini", "codex"]
    ids = [m["id"] for m in profile["ai_models"]]
    assert ids == [
        "antigravity:gemini-3-flash-low",
        "antigravity:gemini-4-hydrogen",
        "gemini:gemini-2.5-flash",
        "codex:default",
    ]
    assert profile["default_ai_model"] == "antigravity:gemini-3-flash-low"
    assert profile["fallback_models"] == ["gemini:gemini-2.5-flash"]


def test_parameter_validation_normalization_and_ai_only_rejected():
    norm = hybrid_ai.normalize_task_parameters({"processing_strategy": "hybrid_ai_vision"})
    assert norm == {
        "processing_strategy": "hybrid_ai_vision",
        "ai_model": "auto",
        "allow_model_fallback": True,
        "ai_tuning": {"aggressiveness": "balanced", "accuracy_tier": "standard"},
    }

    with pytest.raises(ValueError, match="ai_parameters_require_ai_strategy"):
        hybrid_ai.validate_task_parameters({"processing_strategy": "vision_only", "ai_model": "auto"})

    with pytest.raises(ValueError, match="ai_parameters_require_ai_strategy"):
        hybrid_ai.validate_task_parameters({"ai_tuning": {"aggressiveness": "aggressive"}})

    with pytest.raises(hybrid_ai.ProcessingTaskError) as exc_info:
        cut_constraints.validate_parameter_overrides({"processing_strategy": "ai_only"})
    assert exc_info.value.error_code == "unsupported_processing_strategy"


def test_bounded_candidates_valley_split_and_continuous_rally_subdivision():
    # Pass 1: >= 2.0s gap splits cleanly along internal gap midpoint when sub-spans fit <= 45.0s.
    events_p1 = [[12.0, 35.0], [40.0, 68.0]]
    records_p1, _, _ = _synthetic_pose_records([(12.0, 35.0), (40.0, 68.0)], duration=75.0, fps=2.0)
    cands_p1 = hybrid_ai.build_bounded_candidates([[10.0, 70.0]], events_p1, records_p1, duration=75.0)
    assert len(cands_p1) == 2
    assert cands_p1[0]["end_s"] == cands_p1[1]["start_s"] == 37.5
    assert all(0 < c["duration_s"] <= 45.0 for c in cands_p1)
    assert "peak_wrist_speed" in cands_p1[0]

    # Pass 2: continuous 120s rally with zero >= 2.0s gaps strictly subdivided <= 45.0s and non-overlapping.
    records_p2, _, _ = _synthetic_pose_records([(0.5, 119.5)], duration=120.0, fps=4.0)
    cands_p2 = hybrid_ai.build_bounded_candidates([[0.0, 120.0]], [[1.0, 119.0]], records_p2, duration=120.0)
    assert len(cands_p2) >= 3
    prev_end = 0.0
    for c in cands_p2:
        assert 0.0 < c["duration_s"] <= 45.0
        assert c["start_s"] >= prev_end - 1e-6
        prev_end = c["end_s"]
    assert abs(cands_p2[0]["start_s"] - 0.0) < 1e-6
    assert abs(cands_p2[-1]["end_s"] - 120.0) < 1e-6


def test_render_contact_sheets_dimensions_timestamps_exif_free(tmp_path):
    proxy = tmp_path / "analysis.mp4"
    _make_proxy_video(proxy, duration=5.0)
    candidates = [
        {"candidate_id": "C001", "start_s": 0.5, "end_s": 4.5, "duration_s": 4.0, "event_count": 3, "active_ratio": 0.9}
    ]
    work_dir = tmp_path / "ai-work"
    rendered = hybrid_ai.render_contact_sheets(proxy, candidates, work_dir, "fast")
    assert len(rendered) == 1
    sheet_path = Path(rendered[0]["sheet_path"])
    assert sheet_path.is_file()
    assert len(rendered[0]["sheet_sha256"]) == 64
    with Image.open(sheet_path) as img:
        assert img.width == 3 * 320
        assert img.height > 28
        assert len(img.getexif()) == 0
        arr = np.asarray(img)
        assert arr.std() > 5.0


def test_validate_trim_bounds_and_short_span_handling():
    batch = [{"candidate_id": "C001", "start_s": 10.0, "end_s": 25.0, "duration_s": 15.0}]
    bad_payloads = [
        {"candidate_id": "C001", "action": "trim", "category": "rally", "confidence": 0.9, "reason": "bad", "trimmed_start_s": 20.0, "trimmed_end_s": 15.0},
        {"candidate_id": "C001", "action": "trim", "category": "rally", "confidence": 0.9, "reason": "bad", "trimmed_start_s": 5.0, "trimmed_end_s": 12.0},
        {"candidate_id": "C001", "action": "trim", "category": "rally", "confidence": 0.9, "reason": "bad", "trimmed_start_s": True, "trimmed_end_s": 15.0},
        {"candidate_id": "C001", "action": "trim", "category": "rally", "confidence": 0.9, "reason": "bad", "trimmed_start_s": 26.0, "trimmed_end_s": 26.9},
        {"candidate_id": "C001", "action": "trim", "category": "rally", "confidence": 0.9, "reason": "bad"},
    ]
    for item in bad_payloads:
        with pytest.raises(hybrid_ai.ProviderCallError) as exc_info:
            hybrid_ai._validate_and_parse_batch_decisions(batch, {"decisions": [item]}, duration=40.0)
        assert exc_info.value.error_code == "invalid_ai_trim_bounds"

    # Valid trim yielding < 1.0s span is flagged trim_too_short -> short_trimmed_span_dropped.
    short_item = {"candidate_id": "C001", "action": "trim", "category": "rally", "confidence": 0.9, "reason": "tiny", "trimmed_start_s": 12.1, "trimmed_end_s": 12.4}
    parsed = hybrid_ai._validate_and_parse_batch_decisions(batch, {"decisions": [short_item]}, duration=40.0)
    span, outcome = hybrid_ai._apply_candidate_decision(batch[0], parsed["C001"], "balanced")
    assert span is None and outcome == "short_trimmed_span_dropped"


def test_apply_candidate_decision_exhaustive_matrix():
    cand = {"candidate_id": "C001", "start_s": 10.0, "end_s": 20.0}

    def run(agg, action, category, conf, trimmed=None, too_short=False):
        return hybrid_ai._apply_candidate_decision(
            cand,
            {
                "action": action,
                "category": category,
                "confidence": conf,
                "trimmed_span": trimmed,
                "trim_too_short": too_short,
            },
            agg,
        )

    # Row 1: Conservative x uncertain -> always retain full
    assert run("conservative", "drop", "uncertain", 0.99) == ([10.0, 20.0], "conservative_retained_uncertain")
    assert run("conservative", "keep", "uncertain", 0.1) == ([10.0, 20.0], "kept")
    # Row 2: Conservative x rally|serve|drill|warmup|shadow_swing -> always retain full
    for cat in ("rally", "serve", "drill", "warmup", "shadow_swing"):
        assert run("conservative", "drop", cat, 0.99) == ([10.0, 20.0], "conservative_retained_low_conf_drop")
        assert run("conservative", "trim", cat, 0.99, trimmed=[11.0, 18.0]) == ([10.0, 20.0], "kept")
    # Row 3: Conservative x ball_pickup|walking|idle -> drop only if action=='drop' and conf>=0.85
    assert run("conservative", "drop", "walking", 0.85) == (None, "dropped")
    assert run("conservative", "drop", "walking", 0.84) == ([10.0, 20.0], "conservative_retained_low_conf_drop")
    assert run("conservative", "keep", "ball_pickup", 0.99) == ([10.0, 20.0], "kept")

    # Row 4: Balanced x rally|serve|drill|warmup
    assert run("balanced", "keep", "rally", 0.5) == ([10.0, 20.0], "kept")
    assert run("balanced", "trim", "serve", 0.8, trimmed=[11.0, 18.0]) == ([11.0, 18.0], "trimmed")
    assert run("balanced", "drop", "drill", 0.80) == (None, "dropped")
    assert run("balanced", "drop", "warmup", 0.79) == ([10.0, 20.0], "conservative_retained_low_conf_drop")
    # Row 5: Balanced x shadow_swing
    assert run("balanced", "drop", "shadow_swing", 0.2) == (None, "dropped")
    assert run("balanced", "keep", "shadow_swing", 0.75) == (None, "dropped")
    assert run("balanced", "keep", "shadow_swing", 0.74) == ([10.0, 20.0], "kept")
    assert run("balanced", "trim", "shadow_swing", 0.74, trimmed=[12.0, 17.0]) == ([12.0, 17.0], "trimmed")
    assert run("balanced", "trim", "shadow_swing", 0.75, trimmed=[12.0, 17.0]) == (None, "dropped")
    # Row 6: Balanced x uncertain
    assert run("balanced", "keep", "uncertain", 0.2) == ([10.0, 20.0], "balanced_retained_uncertain")
    assert run("balanced", "trim", "uncertain", 0.8, trimmed=[11.0, 16.0]) == ([11.0, 16.0], "trimmed")
    assert run("balanced", "drop", "uncertain", 0.70) == (None, "dropped")
    assert run("balanced", "drop", "uncertain", 0.69) == ([10.0, 20.0], "balanced_retained_uncertain")
    # Row 7: Balanced x ball_pickup|walking|idle
    assert run("balanced", "drop", "idle", 0.1) == (None, "dropped")
    assert run("balanced", "keep", "walking", 0.75) == (None, "dropped")
    assert run("balanced", "keep", "walking", 0.74) == ([10.0, 20.0], "kept")
    assert run("balanced", "trim", "ball_pickup", 0.74, trimmed=[11.0, 15.0]) == ([11.0, 15.0], "trimmed")

    # Row 8: Aggressive x rally|serve|drill
    assert run("aggressive", "keep", "rally", 0.1) == ([10.0, 20.0], "kept")
    assert run("aggressive", "trim", "serve", 0.8, trimmed=[11.0, 15.0]) == ([11.0, 15.0], "trimmed")
    assert run("aggressive", "drop", "drill", 0.65) == (None, "dropped")
    assert run("aggressive", "drop", "rally", 0.64) == ([10.0, 20.0], "conservative_retained_low_conf_drop")
    # Row 9: Aggressive x uncertain
    assert run("aggressive", "keep", "uncertain", 0.49) == ([10.0, 20.0], "kept")
    assert run("aggressive", "keep", "uncertain", 0.50) == (None, "dropped")
    assert run("aggressive", "trim", "uncertain", 0.10, trimmed=[11.0, 15.0]) == (None, "dropped")
    # Row 10: Aggressive x warmup|shadow_swing|ball_pickup|walking|idle
    for cat in ("warmup", "shadow_swing", "ball_pickup", "walking", "idle"):
        assert run("aggressive", "keep", cat, 0.99) == (None, "dropped")


def test_hybrid_fallback_mixed_batches_and_strict_no_fallback(tmp_path, monkeypatch):
    proxy = tmp_path / "analysis.mp4"
    meta = _make_proxy_video(proxy, duration=15.0)
    # 5 separate short active bursts (> 1.5s apart) -> with thorough tier (batch_size=4) yields 2 batches!
    ranges = [(0.5, 1.5), (4.0, 5.0), (7.5, 8.5), (10.5, 11.5), (13.2, 14.2)]
    records, proxy_meta, duration = _synthetic_pose_records(ranges, duration=15.0, fps=8.0)
    monkeypatch.setattr(detector, "extract_pose_records", lambda *args, **kwargs: (records, proxy_meta, duration))

    ai_profile = {
        "capabilities_extra": [hybrid_ai.VERSION],
        "ai_providers": ["gemini"],
        "ai_models": [
            {"id": "gemini:m1", "provider": "gemini", "model": "m1", "label": "M1", "is_default": True},
            {"id": "gemini:m2", "provider": "gemini", "model": "m2", "label": "M2", "is_default": False},
        ],
        "default_ai_model": "gemini:m1",
        "fallback_models": ["gemini:m2"],
        "provider_runtime": {},
    }

    calls = []

    def fake_invoke(*, model_id, batch, tier, prompt, batch_dir):
        calls.append((model_id, [c["candidate_id"] for c in batch]))
        # Fail m1 on batch 2 so m2 handles batch 2 -> "mixed" provenance.
        if model_id == "gemini:m1" and len(batch) < 4:
            raise hybrid_ai.ProviderCallError("rate_limit", "429 quota")
        return {
            "decisions": [
                {
                    "candidate_id": c["candidate_id"],
                    "action": "keep",
                    "category": "rally",
                    "confidence": 0.92,
                    "reason": "rally stroke",
                }
                for c in batch
            ]
        }

    cfg = {
        "_task_parameters": {
            "processing_strategy": "hybrid_ai_vision",
            "ai_model": "gemini:m1",
            "allow_model_fallback": True,
            "ai_tuning": {"aggressiveness": "balanced", "accuracy_tier": "thorough"},
        },
        "_ai_profile": ai_profile,
        "_ai_provider_invoke": fake_invoke,
        "_source_sha256": "a" * 64,
        "_state_dir": str(tmp_path / "state"),
        "_runtime": {"model_sha256": None},
        "padding_before": 0.2,
        "padding_after": 0.2,
        "merge_gap": 0.3,
    }
    intervals, findings = hybrid_ai.run_hybrid_ai_analysis(proxy, meta, cfg)
    assert len(intervals) == 5
    ref = findings["ai_refinement"]
    assert ref["actual_ai_model"] == "mixed"
    assert ref["actual_ai_models"] == ["gemini:m1", "gemini:m2"]
    assert ref["mixed_models_used"] is True
    assert ref["fallback_used"] is True
    assert ref["ai_call_count"] == 3
    assert ref["batch_count_total"] == 2
    assert ref["batch_provenance"][0]["accepted_model"] == "gemini:m1"
    assert ref["batch_provenance"][1]["accepted_model"] == "gemini:m2"
    assert ref["batch_provenance"][1]["attempt_count"] == 2
    assert len(calls) == 3

    # Strict allow_model_fallback=False fails immediately on m1 error without trying m2.
    calls.clear()
    cfg_strict = {
        **cfg,
        "_task_parameters": {
            "processing_strategy": "hybrid_ai_vision",
            "ai_model": "gemini:m1",
            "allow_model_fallback": False,
            "ai_tuning": {"aggressiveness": "balanced", "accuracy_tier": "thorough"},
        },
        "_ai_provider_invoke": lambda **kw: (_ for _ in ()).throw(hybrid_ai.ProviderCallError("provider_down", "boom")),
    }
    with pytest.raises(hybrid_ai.ProcessingTaskError) as exc_strict:
        hybrid_ai.run_hybrid_ai_analysis(proxy, meta, cfg_strict)
    assert exc_strict.value.error_code == "ai_provider_failed"

    # Unavailable exact model fails immediately with ai_model_unavailable.
    cfg_missing = {
        **cfg,
        "_task_parameters": {
            "processing_strategy": "hybrid_ai_vision",
            "ai_model": "gemini:missing",
            "allow_model_fallback": False,
        },
    }
    with pytest.raises(hybrid_ai.ProcessingTaskError) as exc_missing:
        hybrid_ai.run_hybrid_ai_analysis(proxy, meta, cfg_missing)
    assert exc_missing.value.error_code == "ai_model_unavailable"

    # All models fail -> ai_all_models_failed (never falls back to vision_only).
    cfg_all_fail = {
        **cfg,
        "_ai_provider_invoke": lambda **kw: (_ for _ in ()).throw(hybrid_ai.ProviderCallError("bad_json", "fail")),
    }
    with pytest.raises(hybrid_ai.ProcessingTaskError) as exc_all:
        hybrid_ai.run_hybrid_ai_analysis(proxy, meta, cfg_all_fail)
    assert exc_all.value.error_code == "ai_all_models_failed"


def test_real_posix_subprocess_large_stdin_cancel_no_deadlock():
    class Cancelled(RuntimeError):
        pass

    started = time.monotonic()

    def guard():
        if time.monotonic() - started >= 0.25:
            raise Cancelled("lease cancelled")

    large_payload = b"x" * (4 * 1024 * 1024)
    with pytest.raises(Cancelled):
        hybrid_ai._run_guarded_subprocess(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdin_bytes=large_payload,
            timeout_s=30.0,
            guard=guard,
        )
    elapsed = time.monotonic() - started
    assert elapsed < 3.0


def test_real_posix_subprocess_exited_leader_open_pipe_descendant_cleanup(tmp_path):
    pid_file = tmp_path / "descendant.pid"
    script = (
        "import pathlib, subprocess, sys; "
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        f"pathlib.Path({str(pid_file)!r}).write_text(str(p.pid)); "
        "sys.exit(0)"
    )
    started = time.monotonic()
    rc, _, _ = hybrid_ai._run_guarded_subprocess([sys.executable, "-c", script], timeout_s=10.0)
    elapsed = time.monotonic() - started
    assert rc == 0
    assert elapsed < 3.0
    descendant_pid = int(pid_file.read_text().strip())
    # Poll briefly in case OS reaping takes a millisecond after SIGKILL.
    dead = False
    for _ in range(20):
        try:
            os.kill(descendant_pid, 0)
        except ProcessLookupError:
            dead = True
            break
        time.sleep(0.05)
    assert dead, f"Descendant PID {descendant_pid} survived process-group cleanup"


def test_real_gemini_http_subprocess_cancelled_mid_headers_no_fallback(tmp_path, monkeypatch):
    class Cancelled(RuntimeError):
        pass

    request_count = 0
    unblock_server = threading.Event()
    server_entered = threading.Event()

    class SlowHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            nonlocal request_count
            request_count += 1
            length = int(self.headers.get("Content-Length", "0"))
            if length > 0:
                self.rfile.read(length)
            server_entered.set()
            unblock_server.wait(timeout=15.0)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args):
            pass

    class ReusableServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True

    server = ReusableServer(("127.0.0.1", 0), SlowHandler)
    port = server.server_address[1]
    srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
    srv_thread.start()

    try:
        monkeypatch.setattr(
            hybrid_ai,
            "GEMINI_ENDPOINT_TEMPLATE",
            f"http://127.0.0.1:{port}/models/{{model}}:generateContent",
        )
        proxy = tmp_path / "analysis.mp4"
        meta = _make_proxy_video(proxy, duration=3.0)
        records, proxy_meta, duration = _synthetic_pose_records([(0.5, 2.5)], duration=3.0, fps=8.0)
        monkeypatch.setattr(detector, "extract_pose_records", lambda *args, **kwargs: (records, proxy_meta, duration))

        ai_profile = {
            "capabilities_extra": [hybrid_ai.VERSION],
            "ai_providers": ["gemini"],
            "ai_models": [
                {"id": "gemini:gemini-2.5-flash", "provider": "gemini", "model": "gemini-2.5-flash", "label": "Flash", "is_default": True},
                {"id": "gemini:gemini-2.5-flash-lite", "provider": "gemini", "model": "gemini-2.5-flash-lite", "label": "Lite", "is_default": False},
            ],
            "default_ai_model": "gemini:gemini-2.5-flash",
            "fallback_models": ["gemini:gemini-2.5-flash-lite"],
            "provider_runtime": {"gemini_api_key_configured": True},
        }

        cancel_armed = False

        def guard():
            nonlocal cancel_armed
            if server_entered.is_set() and cancel_armed:
                raise Cancelled("cancelled during HTTP header wait")
            if server_entered.is_set():
                cancel_armed = True

        started = time.monotonic()
        with pytest.raises(Cancelled):
            hybrid_ai.run_hybrid_ai_analysis(
                proxy,
                meta,
                {
                    "_guard": guard,
                    "_task_parameters": {
                        "processing_strategy": "hybrid_ai_vision",
                        "ai_model": "gemini:gemini-2.5-flash",
                        "allow_model_fallback": True,
                        "ai_tuning": {"aggressiveness": "balanced", "accuracy_tier": "fast"},
                    },
                    "_ai_profile": ai_profile,
                    "_ai_config": {"enabled": True, "gemini_api_key": "AIzaSyntheticLocalServerKey123", "request_timeout_seconds": 30},
                    "_source_sha256": "b" * 64,
                    "_state_dir": str(tmp_path / "state"),
                    "_runtime": {"model_sha256": None},
                },
            )
        elapsed = time.monotonic() - started
        assert elapsed < 5.0
        assert request_count == 1
    finally:
        unblock_server.set()
        server.shutdown()
        server.server_close()


def test_shared_pose_cache_reused_while_derived_observations_recompute(tmp_path, monkeypatch):
    scratch_a = tmp_path / "tasks" / "t_a"
    scratch_b = tmp_path / "tasks" / "t_b"
    scratch_a.mkdir(parents=True)
    scratch_b.mkdir(parents=True)
    state_dir = tmp_path / "state"

    proxy_a = scratch_a / "analysis.mp4"
    proxy_b = scratch_b / "analysis.mp4"
    meta = _make_proxy_video(proxy_a, duration=3.0)
    proxy_b.write_bytes(proxy_a.read_bytes())

    extract_calls = 0

    def fake_extract(video_path, cfg):
        nonlocal extract_calls
        extract_calls += 1
        return _synthetic_pose_records([(0.5, 2.5)], duration=3.0, fps=8.0)

    monkeypatch.setattr(detector, "extract_pose_records", fake_extract)

    ai_profile = {
        "capabilities_extra": [hybrid_ai.VERSION],
        "ai_providers": ["gemini"],
        "ai_models": [{"id": "gemini:m1", "provider": "gemini", "model": "m1", "label": "M1", "is_default": True}],
        "default_ai_model": "gemini:m1",
        "fallback_models": [],
        "provider_runtime": {},
    }

    def keep_all(*, batch, **kwargs):
        return {
            "decisions": [
                {"candidate_id": c["candidate_id"], "action": "keep", "category": "rally", "confidence": 0.9, "reason": "ok"}
                for c in batch
            ]
        }

    base_cfg = {
        "_ai_profile": ai_profile,
        "_ai_provider_invoke": keep_all,
        "_source_sha256": "c" * 64,
        "_state_dir": str(state_dir),
        "_runtime": {"model_sha256": "d" * 64},
    }
    _, findings_a = hybrid_ai.run_hybrid_ai_analysis(
        proxy_a,
        meta,
        {
            **base_cfg,
            "_task_parameters": {
                "processing_strategy": "hybrid_ai_vision",
                "ai_tuning": {"aggressiveness": "balanced", "accuracy_tier": "fast"},
            },
            "target_region": [0.0, 0.0, 1.0, 1.0],
        },
    )
    assert extract_calls == 1
    assert findings_a["ai_refinement"]["raw_pose_cache_hit"] is False
    assert (scratch_a / "pose-observations.json").is_file()
    assert list((state_dir / "cache" / "pose").glob("*.json"))

    # Second task with different scratch, AI tuning, and non-overlapping ROI reuses raw pose cache yet recomputes derived observations.
    _, findings_b = hybrid_ai.run_hybrid_ai_analysis(
        proxy_b,
        meta,
        {
            **base_cfg,
            "_task_parameters": {
                "processing_strategy": "hybrid_ai_vision",
                "ai_tuning": {"aggressiveness": "conservative", "accuracy_tier": "fast"},
            },
            "target_region": [0.0, 0.0, 0.4, 1.0],
        },
    )
    assert extract_calls == 1
    assert findings_b["ai_refinement"]["raw_pose_cache_hit"] is True
    assert findings_b["ai_refinement"]["candidate_count"] == 0
    obs_a = json.loads((scratch_a / "pose-observations.json").read_text())
    obs_b = json.loads((scratch_b / "pose-observations.json").read_text())
    assert any(f["person_present"] for f in obs_a["frame_observations"])
    assert all(not f["person_present"] for f in obs_b["frame_observations"])


def test_ai_findings_32kib_cap_and_lossless_ai_provenance_artifact(tmp_path):
    big_batches = [
        {
            "batch_id": f"batch-{i:03d}",
            "candidate_ids": [f"C{i:03d}"],
            "accepted_model": "gemini:gemini-2.5-flash",
            "attempt_count": 1,
            "prompt_sha256": hashlib.sha256(f"p{i}".encode()).hexdigest(),
            "input_manifest_sha256": hashlib.sha256(f"m{i}".encode()).hexdigest(),
        }
        for i in range(1, 220)
    ]
    raw_summary = {
        "detector": hybrid_ai.VERSION,
        "pose_backbone": detector.VERSION,
        "processing_strategy": "hybrid_ai_vision",
        "requested_ai_model": "gemini:gemini-2.5-flash",
        "actual_ai_model": "gemini:gemini-2.5-flash",
        "actual_ai_models": ["gemini:gemini-2.5-flash"],
        "mixed_models_used": False,
        "allow_model_fallback": True,
        "fallback_used": False,
        "ai_tuning": {"aggressiveness": "balanced", "accuracy_tier": "fast"},
        "ai_call_count": len(big_batches),
        "candidate_count": len(big_batches),
        "kept_count": len(big_batches),
        "trimmed_count": 0,
        "dropped_count": 0,
        "uncertain_retained_count": 0,
        "uncertain_dropped_count": 0,
        "schema_sha256": "e" * 64,
        "batch_provenance": big_batches,
        "attempted_models": [{"model": "gemini:gemini-2.5-flash", "batch_id": b["batch_id"], "status": "succeeded"} for b in big_batches],
    }
    bounded = hybrid_ai.bound_ai_refinement_findings(raw_summary)
    assert bounded["provenance_truncated"] is True
    assert bounded["batch_count_total"] == 219
    assert len(hybrid_ai._canonical_json(bounded).encode("utf-8")) <= hybrid_ai.MAX_AI_REFINEMENT_FINDINGS_BYTES

    # Verify pipeline.render_session writes lossless ai-provenance.json and binds sha256 into session + public artifacts.
    src = tmp_path / "source.mp4"
    meta = _make_proxy_video(src, duration=2.0)
    out = tmp_path / "session-out"
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    full_prov = {
        **raw_summary,
        "provenance_truncated": False,
        "candidate_decisions": [{"candidate_id": f"C{i:03d}", "action": "keep"} for i in range(1, 220)],
    }
    analysis = {
        "detector": hybrid_ai.VERSION,
        "target_visible_fraction": 1.0,
        "effective_detector_settings": {"swing_speed": 1.6, "padding_before": 0.2, "padding_after": 0.2, "merge_gap": 0.3},
        "ai_refinement": bounded,
        "_ai_provenance_payload": full_prov,
    }
    cfg = {
        "processing_strategy": "hybrid_ai_vision",
        "ai_model": "gemini:gemini-2.5-flash",
        "allow_model_fallback": True,
        "ai_tuning": {"aggressiveness": "balanced", "accuracy_tier": "fast"},
        "_allow_empty_analysis": True,
        "_task_id": "t_prov",
        "_attempt_id": "a_prov",
    }
    report = pipeline.render_session(src, out, cfg, digest, "source.mp4", [], analysis, meta)
    prov_file = out / "ai-provenance.json"
    assert prov_file.is_file()
    on_disk = json.loads(prov_file.read_text())
    assert on_disk["provenance_truncated"] is False
    assert len(on_disk["batch_provenance"]) == 219
    assert len(on_disk["candidate_decisions"]) == 219
    prov_sha = hashlib.sha256(prov_file.read_bytes()).hexdigest()
    assert report["analysis"]["ai_refinement"]["provenance_artifact"] == "ai-provenance.json"
    assert report["analysis"]["ai_refinement"]["provenance_sha256"] == prov_sha
    assert report["settings"]["processing_strategy"] == "hybrid_ai_vision"
    _, pub = metadata_for_session(out)
    assert "ai-provenance.json" in pub["artifacts"]
    diag = json.loads((out / "heuristic-diagnostics.json").read_text())
    assert diag["provenance"]["software"]["detector_version"] == hybrid_ai.VERSION
    assert diag["provenance"]["cut_assembly_config"]["padding_before"] == 0.2


def test_provider_wire_contracts_and_codex_subscription_isolation(tmp_path, monkeypatch):
    """Verify actual request formats rather than only parsed mocked decisions."""
    schema = hybrid_ai.DECISION_JSON_SCHEMA
    decision = schema['properties']['decisions']['items']
    assert set(decision['required']) == set(decision['properties'])
    for key in ('trimmed_start_s', 'trimmed_end_s'):
        assert decision['properties'][key]['type'] == ['number', 'null']
    sheet = tmp_path / 'sheet.jpg'
    Image.new('RGB', (32, 32), 'green').save(sheet)
    batch = [dict(candidate_id='C001', start_s=0.0, end_s=2.0, duration_s=2.0,
                  event_count=1, active_ratio=0.5, sheet_path=str(sheet), sheet_sha256='a'*64)]
    response = {'decisions': [dict(candidate_id='C001', action='keep', category='rally',
                                  confidence=0.9, reason='Synthetic', trimmed_start_s=None, trimmed_end_s=None)]}
    monkeypatch.setenv('OPENAI_API_KEY', 'synthetic-unwanted-api-key')
    monkeypatch.setenv('CODEX_API_KEY', 'synthetic-unwanted-api-key')
    calls = []

    def invoke(cmd, **kw):
        calls.append(cmd)
        if 'exec' in cmd:
            assert '--ignore-user-config' in cmd and '--ephemeral' in cmd and 'read-only' in cmd
            assert cmd[-1] == '-' and b'Response JSON Schema' in kw['stdin_bytes']
            assert not any(k in kw['env'] for k in ('OPENAI_API_KEY', 'CODEX_API_KEY'))
            for feature in ('shell_tool', 'unified_exec', 'apps', 'plugins', 'browser_use', 'computer_use', 'in_app_browser', 'code_mode'):
                assert any(cmd[i:i+2] == ['--disable', feature] for i in range(len(cmd)-1))
            assert json.loads(Path(cmd[cmd.index('--output-schema')+1]).read_text()) == schema
            Path(cmd[cmd.index('-o')+1]).write_text(json.dumps(response))
            return 0, b'', b''
        if '--gemini-http' in cmd:
            payload = json.loads(kw['stdin_bytes'])['payload']['generationConfig']
            assert payload['response_json_schema'] == schema and 'response_schema' not in payload
            return 0, json.dumps({'candidates':[{'content':{'parts':[{'text':json.dumps(response)}]}}]}).encode(), b''
        assert '--sandbox' in cmd and '--json-schema' in cmd and '-p' in cmd
        assert '@C001.jpg' in cmd[-1] and hybrid_ai._canonical_json(schema) in cmd[-1]
        return 0, json.dumps({'status':'SUCCESS', 'structured_output':response}).encode(), b''

    monkeypatch.setattr(hybrid_ai, '_run_guarded_subprocess', invoke)
    for model in ('codex:default', 'gemini:synthetic-model', 'antigravity:synthetic-model'):
        actual, _, _ = hybrid_ai._invoke_provider_batch(
            model, batch, 'fast', {}, {'gemini_api_key':'synthetic-key'},
            {'codex_bin':'synthetic-codex', 'antigravity_bin':'synthetic-agy'})
        assert actual == response
    assert len(calls) == 3


def test_antigravity_current_tab_separated_model_catalog():
    models = hybrid_ai._parse_agy_models_output("gemini-4-hydrogen-low\tGemini 4 Hydrogen (Low)\ngemini-3.8-flash-low\tGemini Flash (Low)\n")
    assert [m["id"] for m in models] == ["antigravity:gemini-4-hydrogen-low", "antigravity:gemini-3.8-flash-low"]


def test_malformed_wrapped_json_remains_fallback_eligible():
    with pytest.raises(hybrid_ai.ProviderCallError) as exc:
        hybrid_ai._extract_json_object("Model output: {broken JSON}")
    assert exc.value.error_code == "invalid_ai_response_json"

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

import api_worker
import detector
import diagnostics
import pipeline
import publication
import worker


def _pose_frame(
    t,
    boxes=None,
    wrist_offset=(0.0, 0.0),
    torso_conf=0.9,
    wrist_conf=0.9,
    shoulder_xy=(480.0, 220.0),
    hip_xy=(480.0, 320.0),
):
    if boxes is None:
        boxes = [[420.0, 160.0, 540.0, 420.0]]
    n = len(boxes)
    k = np.zeros((n, 17, 2), dtype=float)
    kc = np.zeros((n, 17), dtype=float)
    for idx in range(n):
        sx = shoulder_xy[0] + idx * 180.0
        sy = shoulder_xy[1]
        hx = hip_xy[0] + idx * 180.0
        hy = hip_xy[1]
        k[idx, 5] = [sx - 20.0, sy]
        k[idx, 6] = [sx + 20.0, sy]
        k[idx, 11] = [hx - 18.0, hy]
        k[idx, 12] = [hx + 18.0, hy]
        k[idx, 9] = [sx + wrist_offset[0], sy + wrist_offset[1]]
        k[idx, 10] = [sx + wrist_offset[0], sy + wrist_offset[1]]
        kc[idx, [5, 6, 11, 12]] = torso_conf if isinstance(torso_conf, (int, float)) else torso_conf[idx]
        kc[idx, [9, 10]] = wrist_conf if isinstance(wrist_conf, (int, float)) else wrist_conf[idx]
    return {
        "t": float(t),
        "boxes": np.asarray(boxes, dtype=float),
        "k": k,
        "kc": kc,
    }


def test_signal_statuses_epistemic_limits_and_hint_codes():
    digest = "a" * 64
    duration = 60.0
    records = []
    # 8 fps synthetic stream across 60s
    for idx in range(int(duration * 8)):
        t = idx * 0.125
        # Gap 1: [0, 5s] -> active motion from 1.0s to 3.5s (2.5s sustained ROI motion in gap)
        if 1.0 <= t < 3.5:
            off = (95.0 if idx % 2 else -95.0, 20.0)
            records.append(_pose_frame(t, wrist_offset=off))
        elif t < 5.0:
            records.append(_pose_frame(t, boxes=[]))
        # Clip 1: [5.0, 12.0] -> clean active rally
        elif 5.0 <= t < 12.0:
            off = (90.0 if idx % 2 else -90.0, 10.0)
            records.append(_pose_frame(t, wrist_offset=off))
        # Gap 2: [12.0, 15.0] -> empty gap
        elif 12.0 <= t < 15.0:
            records.append(_pose_frame(t, boxes=[]))
        # Clip 2: [15.0, 26.0] -> active [15.0, 21.5], then person track lost near end [21.5, 26.0] (4.5s)
        elif 15.0 <= t < 21.5:
            off = (85.0 if idx % 2 else -85.0, 15.0)
            records.append(_pose_frame(t, wrist_offset=off))
        elif 21.5 <= t < 26.0:
            records.append(_pose_frame(t, boxes=[]))
        # Gap 3: [26.0, 28.0] -> empty
        elif 26.0 <= t < 28.0:
            records.append(_pose_frame(t, boxes=[]))
        # Clip 3: [28.0, 42.0] -> active [28.0, 30.0], extended mid-clip low motion [30.0, 39.5] (9.5s), active [39.5, 42.0]
        elif 28.0 <= t < 30.0 or 39.5 <= t < 42.0:
            off = (90.0 if idx % 2 else -90.0, 10.0)
            records.append(_pose_frame(t, wrist_offset=off))
        elif 30.0 <= t < 39.5:
            records.append(_pose_frame(t, wrist_offset=(0.0, 0.0)))
        # Gap 4: [42.0, 44.0] -> empty
        elif 42.0 <= t < 44.0:
            records.append(_pose_frame(t, boxes=[]))
        # Clip 4: [44.0, 52.0] -> ambiguous multiple persons (two comparable height boxes in ROI)
        elif 44.0 <= t < 52.0:
            off = (80.0 if idx % 2 else -80.0, 10.0)
            boxes = [
                [260.0, 160.0, 380.0, 420.0],
                [520.0, 170.0, 640.0, 415.0],
            ]
            records.append(_pose_frame(t, boxes=boxes, wrist_offset=off))
        # Clip 5: [52.0, 58.0] -> degraded low keypoint confidence (unknown)
        elif 52.0 <= t < 58.0:
            records.append(_pose_frame(t, torso_conf=0.10, wrist_conf=0.10))
        else:
            records.append(_pose_frame(t, boxes=[]))

    intervals = [[5.0, 12.0], [15.0, 26.0], [28.0, 42.0], [44.0, 52.0], [52.0, 58.0]]
    cfg = {"target_region": [0, 0, 1, 1], "min_player_height": 0.12}
    meta = {"duration": duration, "width": 1920, "height": 1080, "transfer": 1}
    payload = diagnostics.build_heuristic_diagnostics(
        source_sha256=digest,
        duration_seconds=duration,
        intervals=intervals,
        cfg=cfg,
        meta=meta,
        analysis={"pose_records": records},
    )

    assert payload["truth_status"] == "heuristic_only_not_play_or_swing_truth"
    assert payload["training_label_status"] == "not_released"
    assert payload["generator"]["kind"] == "worker_heuristic_v1"
    assert payload["generator"]["benchmark_truth"] is False
    assert payload["generator"]["verified_play_or_swing"] is False
    assert payload["provenance"]["proxy_spec"]["target_fps"] == 8.0
    assert payload["provenance"]["proxy_spec"]["width"] == 960
    assert payload["provenance"]["proxy_spec"]["height"] == 540

    clips = {c["clip_index"]: c for c in payload["clips"]}
    assert clips[1]["heuristic_status"] == "no_detector_warnings"
    assert clips[1]["identity_status"] == "unverified_single_primary_track"
    assert clips[1]["hints"] == []

    assert clips[2]["heuristic_status"] == "check_boundary_activity"
    tail_hints = [h for h in clips[2]["hints"] if h["code"] == "person_track_lost_near_end"]
    assert len(tail_hints) == 1
    assert tail_hints[0]["hint_id"] == "h_clip02_tail"
    assert tail_hints[0]["candidate_end_ms"] == tail_hints[0]["original_start_ms"]
    assert tail_hints[0]["mapped_reasons"] == ["end_late", "lost_player"]

    assert clips[3]["heuristic_status"] == "extended_low_motion"
    assert any(h["code"] == "extended_low_motion" for h in clips[3]["hints"])

    assert clips[4]["heuristic_status"] == "ambiguous_multiple_persons"
    assert clips[4]["identity_status"] == "ambiguous_multiple_persons"
    assert any(h["code"] == "ambiguous_multiple_persons" for h in clips[4]["hints"])

    assert clips[5]["heuristic_status"] == "unknown"
    assert clips[5]["identity_status"] == "unknown"
    assert any(h["code"] == "detector_visibility_degraded" for h in clips[5]["hints"])

    gaps = {g["gap_index"]: g for g in payload["removed_gaps"]}
    assert gaps[1]["heuristic_status"] == "check_boundary_activity"
    assert any(h["code"] == "sustained_roi_motion_in_gap" and h["mapped_reasons"] == ["missed_play"] for h in gaps[1]["hints"])
    assert gaps[2]["heuristic_status"] == "no_detector_warnings"


def test_hint_capping_and_omission_counts_are_explicit():
    digest = "b" * 64
    duration = 40.0
    # Create 7 distinct sustained motion bursts inside a single removed gap [0, 35.0]
    observations = []
    burst_windows = [(1.0, 3.0), (4.0, 6.2), (7.5, 9.8), (11.0, 13.5), (15.0, 17.8), (19.5, 22.0), (24.0, 27.0)]
    for idx in range(int(duration * 8)):
        t = idx * 0.125
        active = any(a <= t < b for a, b in burst_windows) or (36.0 <= t < 39.0)
        observations.append(
            {
                "t": round(t, 6),
                "person_present": active,
                "active_motion": active,
                "ambiguous_multiple": False,
                "low_confidence_candidate": False,
                "roi_candidates": 1 if active else 0,
                "wrist_speed": 2.5 if active else 0.0,
                "body_speed": 0.4 if active else 0.0,
            }
        )

    payload = diagnostics.build_heuristic_diagnostics(
        source_sha256=digest,
        duration_seconds=duration,
        intervals=[[35.0, 40.0]],
        cfg={"target_region": [0, 0, 1, 1], "min_player_height": 0.12},
        meta={"duration": duration, "width": 960, "height": 540, "transfer": 1},
        frame_observations=observations,
    )
    gap = payload["removed_gaps"][0]
    assert len(gap["hints"]) == 5
    assert gap["total_candidate_hints"] == 7
    assert gap["omitted_hints_count"] == 2
    assert gap["hints_truncated"] is True
    assert payload["omitted_hints_count"] == 2
    assert payload["limits"]["omitted_gap_hints_count"] == 2
    assert payload["limits"]["omitted_hints_count"] == 2
    assert payload["limits"]["hints_truncated"] is True


def test_extraction_cache_key_invalidates_on_every_upstream_dimension():
    digest = "c" * 64
    base_meta = {"duration": 10.0, "width": 1920, "height": 1080, "transfer": 1, "original_metadata": {"rotation_degrees": 0}}
    base_cfg = {"target_region": [0.05, 0.10, 0.95, 0.95], "min_player_height": 0.12, "pose_device": "coreml"}
    base_proxy = {"width": 960, "height": 540, "target_fps": 8.0, "pts_timebase": "1/16384"}
    base_model = "1" * 64
    base_code = "2" * 64

    base_key, base_prov = diagnostics.build_extraction_provenance(
        digest, base_meta, base_cfg, proxy_meta=base_proxy, model_sha256=base_model, extractor_code_digest=base_code
    )
    assert base_prov["proxy_spec"]["target_fps"] == 8.0

    # Downstream cut-assembly changes (padding_before/after, merge_gap) keep the same upstream extraction_cache_key
    downstream_cfg = dict(base_cfg, padding_before=3.5, padding_after=2.5, merge_gap=1.2)
    same_key, _ = diagnostics.build_extraction_provenance(
        digest, base_meta, downstream_cfg, proxy_meta=base_proxy, model_sha256=base_model, extractor_code_digest=base_code
    )
    assert same_key == base_key

    # Changing any upstream input must change extraction_cache_key
    variants = [
        ("source_sha256", ("d" * 64, base_meta, base_cfg, base_proxy, base_model, base_code)),
        ("roi", (digest, base_meta, dict(base_cfg, target_region=[0.1, 0.1, 0.9, 0.9]), base_proxy, base_model, base_code)),
        ("min_height", (digest, base_meta, dict(base_cfg, min_player_height=0.18), base_proxy, base_model, base_code)),
        ("backend", (digest, base_meta, dict(base_cfg, pose_device="cpu"), base_proxy, base_model, base_code)),
        ("policy_version", (digest, base_meta, dict(base_cfg, target_selection_policy_version="primary-track-v2"), base_proxy, base_model, base_code)),
        ("player_revision", (digest, base_meta, dict(base_cfg, player_assignment_revision="rev-2"), base_proxy, base_model, base_code)),
        ("proxy_fps", (digest, base_meta, base_cfg, dict(base_proxy, target_fps=12.0), base_model, base_code)),
        ("proxy_size", (digest, base_meta, base_cfg, dict(base_proxy, width=640, height=360), base_model, base_code)),
        ("proxy_timebase", (digest, base_meta, base_cfg, dict(base_proxy, pts_timebase="1/1000"), base_model, base_code)),
        ("rotation", (digest, dict(base_meta, original_metadata={"rotation_degrees": 90}), base_cfg, base_proxy, base_model, base_code)),
        ("color_transfer", (digest, dict(base_meta, transfer=18), base_cfg, base_proxy, base_model, base_code)),
        ("model_sha256", (digest, base_meta, base_cfg, base_proxy, "3" * 64, base_code)),
        ("code_digest", (digest, base_meta, base_cfg, base_proxy, base_model, "4" * 64)),
    ]
    for label, (s_sha, m, c, p, mod, cd) in variants:
        k, _ = diagnostics.build_extraction_provenance(s_sha, m, c, proxy_meta=p, model_sha256=mod, extractor_code_digest=cd)
        assert k != base_key, f"Expected cache key invalidation for {label}"


def test_range_and_schema_validation_rejects_out_of_bounds_and_invalid_types():
    digest = "e" * 64
    payload = diagnostics.build_heuristic_diagnostics(
        source_sha256=digest,
        duration_seconds=10.0,
        intervals=[[2.0, 6.0]],
        cfg={"target_region": [0, 0, 1, 1], "min_player_height": 0.12},
        meta={"duration": 10.0, "width": 960, "height": 540, "transfer": 1},
    )

    # Clip hint outside parent clip bounds must fail validation
    bad_clip_hint = json.loads(json.dumps(payload))
    bad_clip_hint["clips"][0]["hints"] = [
        {
            "hint_id": "h_bad_clip",
            "code": "person_track_lost_near_end",
            "original_start_ms": 5000,
            "original_end_ms": 6500,
            "summary": "Exceeds parent clip end 6000",
            "mapped_reasons": ["end_late"],
        }
    ]
    bad_clip_hint["clips"][0]["total_candidate_hints"] = 1
    with pytest.raises(ValueError, match="violates parent bounds"):
        diagnostics.validate_heuristic_diagnostics(bad_clip_hint)

    # Gap hint outside parent gap bounds must fail validation
    bad_gap_hint = json.loads(json.dumps(payload))
    bad_gap_hint["removed_gaps"][0]["hints"] = [
        {
            "hint_id": "h_bad_gap",
            "code": "sustained_roi_motion_in_gap",
            "original_start_ms": 0,
            "original_end_ms": 2500,
            "summary": "Exceeds gap end 2000",
            "mapped_reasons": ["missed_play"],
        }
    ]
    bad_gap_hint["removed_gaps"][0]["total_candidate_hints"] = 1
    with pytest.raises(ValueError, match="violates parent bounds"):
        diagnostics.validate_heuristic_diagnostics(bad_gap_hint)

    # Bool-as-int on millisecond field must be rejected
    bad_bool = json.loads(json.dumps(payload))
    bad_bool["clips"][0]["original_start_ms"] = True
    with pytest.raises(ValueError):
        diagnostics.validate_heuristic_diagnostics(bad_bool)

    # Claiming play/swing truth must be rejected
    bad_truth = json.loads(json.dumps(payload))
    bad_truth["generator"]["truth_status"] = "verified_play"
    with pytest.raises(ValueError, match="truth_status"):
        diagnostics.validate_heuristic_diagnostics(bad_truth)


def test_local_shared_observation_cache_recovers_from_corruption(tmp_path):
    cache_dir = tmp_path / "obs-cache"
    key = "f" * 64
    obs = [{"t": 0.0, "person_present": True, "active_motion": True, "ambiguous_multiple": False, "low_confidence_candidate": False, "roi_candidates": 1, "wrist_speed": 2.2, "body_speed": 0.3}]
    entry = diagnostics.save_shared_observation_cache(cache_dir, key, obs, proxy_meta={"target_fps": 8.0}, events=[[0.0, 0.5]], visibility=1.0)
    loaded = diagnostics.load_shared_observation_cache(cache_dir, key)
    assert loaded is not None and loaded["frame_observations"] == obs

    # Corrupt cache file -> returns None and cleans up corrupt files
    entry.write_text("{corrupt")
    assert diagnostics.load_shared_observation_cache(cache_dir, key) is None
    assert not entry.exists()


def test_end_to_end_packaging_frozen_publication_and_dryrun(tmp_path, monkeypatch):
    source_dir = tmp_path / "originals"
    processing_dir = tmp_path / "processing"
    library_dir = tmp_path / "library"
    state_dir = tmp_path / "state"
    for d in (source_dir, processing_dir, library_dir, state_dir):
        d.mkdir()
    (library_dir / ".tennis-library-id").write_text("lib-1\n")

    source_video = source_dir / "session.mp4"
    log_path = tmp_path / "ffmpeg-fixture.log"
    pipeline.run_media(
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=30",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000",
            "-t",
            "6",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(source_video),
        ],
        log_path,
    )
    digest = api_worker.checksum(source_video)

    def fake_detect(proxy_path, cfg):
        # Create dense observations in local scratch beside proxy, as detector.detect does
        obs = []
        for idx in range(48):
            t = idx * 0.125
            active = 1.0 <= t <= 4.5
            obs.append(
                {
                    "t": round(t, 6),
                    "person_present": active,
                    "active_motion": active,
                    "ambiguous_multiple": False,
                    "low_confidence_candidate": False,
                    "roi_candidates": 1 if active else 0,
                    "wrist_speed": 2.4 if active else 0.0,
                    "body_speed": 0.35 if active else 0.0,
                }
            )
        proxy_meta = {"width": 320, "height": 180, "target_fps": 8.0, "actual_fps": 8.0, "pts_timebase": "1/16384", "frame_count": len(obs)}
        diagnostics.save_local_observations(Path(proxy_path).parent / "pose-observations.json", obs, proxy_meta=proxy_meta)
        return [[1.0, 4.5]], {
            "detector": detector.VERSION,
            "target_visible_fraction": 0.6,
            "motion_events": 1,
            "event_intervals": [[1.5, 3.0]],
            "proxy_observation_meta": proxy_meta,
            "note": "test",
        }

    monkeypatch.setattr(pipeline, "detect", fake_detect)

    # 1. Verify folder/library publication freezes published_at and packages heuristic-diagnostics.json
    scratch_out = tmp_path / "local-job/result"
    cfg = {"target_region": [0, 0, 1, 1], "min_player_height": 0.12, "pose_device": "coreml"}
    report = pipeline.process_video(source_video, scratch_out, cfg, digest, "session.mp4")
    assert (scratch_out.parent / "pose-observations.json").is_file()
    assert not (scratch_out / "pose-observations.json").exists()
    assert (scratch_out / "heuristic-diagnostics.json").is_file()
    assert "heuristic-diagnostics.json" in report["output_checksums"]

    _, pub1 = publication.metadata_for_session(scratch_out)
    _, pub2 = publication.metadata_for_session(scratch_out)
    assert pub1["published_at"] == pub2["published_at"] == report["published_at"]

    media_dest = publication.publish_library(scratch_out, library_dir, digest=digest)
    assert publication.validate_library(media_dest, digest)
    manifest = json.loads((library_dir / "metadata/sessions" / f"{digest}.json").read_text())
    diag_lib_path = library_dir / "metadata/sessions" / digest / "heuristic-diagnostics.json"
    assert diag_lib_path.is_file()
    assert not (library_dir / "metadata/sessions" / digest / "pose-observations.json").exists()
    diag_lib = json.loads(diag_lib_path.read_text())
    assert diag_lib["published_at"] == manifest["published_at"]
    assert diag_lib["publication"]["published_at"] == manifest["published_at"]
    assert diag_lib["publication"]["clip_ids"] == [c["id"] for c in manifest["clips"]]
    assert diag_lib["video_version_id"].startswith("v_")

    # 2. Verify API worker normal process vs dryrun local-only behavior & runtime build_digest
    calls = []

    def fake_call(self, path, body=None, key=None, bootstrap=False):
        calls.append((path, body, key))
        if path == "/workers/register":
            return {"worker_id": "w_1", "worker_token": "tok_1"}
        if path.endswith("/heartbeat"):
            return {"lease_expires_at": 9999999999.0, "server_time": 1000.0}
        if path.endswith("/events"):
            return {"acknowledged_sequence": body["events"][-1]["sequence"]}
        if path.endswith("/complete"):
            return {"status": "done"}
        return {}

    monkeypatch.setattr(api_worker.API, "call", fake_call)
    worker_cfg = {
        "name": "test-worker",
        "server_url": "http://localhost:9999",
        "registration_token": "reg",
        "state_dir": str(state_dir),
        "storage": {
            "originals": {"path": str(source_dir)},
            "processing": {"path": str(processing_dir)},
        },
        "processor": cfg,
    }
    agent = api_worker.Agent(worker_cfg)
    st = source_video.stat()

    # Dryrun task: creates local bundle with heuristic-diagnostics.json, writes NOTHING to NAS processing_dir
    dryrun_assignment = {
        "task": {
            "id": "task_dry",
            "type": "dryrun",
            "input": {"storage_id": "originals", "path": "session.mp4", "bytes": st.st_size, "modified_ns": st.st_mtime_ns, "sha256": digest},
            "parameters": {},
        },
        "attempt": {
            "id": "att_dry_1",
            "number": 1,
            "lease_token": "lease_dry",
            "lease_expires_at": 9999999999.0,
            "output_prefix": {"storage_id": "processing", "path": "runs/att_dry_1"},
        },
    }
    agent.execute(dryrun_assignment)
    assert not (processing_dir / "runs/att_dry_1").exists()
    dry_bundle_diag = json.loads((state_dir / "tasks/task_dry/bundle/metadata/heuristic-diagnostics.json").read_text())
    assert dry_bundle_diag["task_id"] == "task_dry"
    assert dry_bundle_diag["attempt_id"] == "att_dry_1"
    assert not list((state_dir / "tasks/task_dry/bundle").rglob("pose-observations.json"))

    # Normal process task: uploads verified bundle including heuristic-diagnostics.json to processing_dir, no dense observations on NAS
    proc_assignment = {
        "task": {
            "id": "task_proc",
            "type": "process",
            "input": {"storage_id": "originals", "path": "session.mp4", "bytes": st.st_size, "modified_ns": st.st_mtime_ns, "sha256": digest},
            "parameters": {},
        },
        "attempt": {
            "id": "att_proc_1",
            "number": 1,
            "lease_token": "lease_proc",
            "lease_expires_at": 9999999999.0,
            "output_prefix": {"storage_id": "processing", "path": "runs/att_proc_1"},
        },
    }
    agent.execute(proc_assignment)
    nas_run = processing_dir / "runs/att_proc_1"
    assert (nas_run / "result.json").is_file()
    assert (nas_run / "metadata/heuristic-diagnostics.json").is_file()
    assert not list(nas_run.rglob("pose-observations.json"))
    result_manifest = json.loads((nas_run / "result.json").read_text())
    nas_diag = json.loads((nas_run / "metadata/heuristic-diagnostics.json").read_text())
    assert "heuristic-diagnostics.json" in result_manifest["session"]["artifacts"]
    assert result_manifest["session"]["metadata_checksums"]["heuristic-diagnostics.json"] == api_worker.checksum(
        nas_run / "metadata/heuristic-diagnostics.json"
    )
    assert nas_diag["attempt_id"] == "att_proc_1"
    assert nas_diag["task_id"] == "task_proc"
    assert nas_diag["published_at"] == result_manifest["session"]["published_at"]
    assert nas_diag["publication"]["video_version_id"] == nas_diag["video_version_id"]

    # Tampering with heuristic-diagnostics.json in published library invalidates validate_library
    diag_lib_path.write_text(json.dumps({**diag_lib, "truth_status": "tampered"}))
    assert not publication.validate_library(media_dest, digest)


def test_runtime_build_digest_includes_diagnostics_and_legacy_outputs_remain_valid(tmp_path,monkeypatch):
    # Legacy output without heuristic-diagnostics.json remains valid
    legacy_dir = tmp_path / "legacy-result"
    legacy_dir.mkdir()
    (legacy_dir / "Practice.mp4").write_bytes(b"video")
    worker.atomic_json(
        legacy_dir / "session.json",
        {"source_sha256": "a" * 64, "output_checksums": {"Practice.mp4": worker.sha256(legacy_dir / "Practice.mp4")}},
    )
    assert worker.validate_publication(legacy_dir, "a" * 64)

    # Verify api_worker build_digest hashes diagnostics.py and publication.py
    expected = hashlib.sha256()
    root = Path(api_worker.__file__).parent
    for name in ("api_worker.py", "pipeline.py", "detector.py", "cut_constraints.py", "diagnostics.py", "publication.py", "color.py", "playback.py", "timeline.py"):
        expected.update((root / name).read_bytes())
    state_dir = tmp_path / "wstate"
    state_dir.mkdir()
    (state_dir / "worker.json").write_text(json.dumps({"worker_id": "w_existing", "worker_token": "tok"}))
    monkeypatch.setattr(api_worker.API,"call",lambda *args,**kwargs:{"version":1})
    agent = api_worker.Agent(
        {
            "name": "w",
            "server_url": "http://localhost:9999",
            "registration_token": "r",
            "state_dir": str(state_dir),
            "storage": {},
            "processor": {},
        }
    )
    assert agent.runtime["build_digest"] == expected.hexdigest()



def test_version_identity_matches_website_contract():
    public = {"id": "a"*64, "published_at": 1234.125, "clips": []}
    expected = "v_" + hashlib.sha256(json.dumps([public["id"], 1234.125], separators=(",", ":")).encode()).hexdigest()[:32]
    assert diagnostics.compute_video_version_id(public, attempt_id="b"*32)[0] == expected
    assert diagnostics.compute_video_version_id(dict(public, media_directory="elsewhere"))[0] == expected

from pathlib import Path
import json
import time

import pytest

import worker
from detector import merge_intervals


def test_upload_must_stay_unchanged(tmp_path):
    db = worker.database(tmp_path / "state.db")
    row = worker.observe(db, "a.MOV", "100:1", 10)
    assert row["stable_since"] == 10
    assert worker.observe(db, "a.MOV", "100:1", 20)["stable_since"] == 10
    assert worker.observe(db, "a.MOV", "200:2", 21)["stable_since"] == 21
    assert worker.observe(db, "a.MOV", "200:2", 30)["status"] == "waiting"


def test_outputs_hidden_files_and_symlinks_not_reingested(tmp_path):
    for name in ["a.MOV", "Originals/b.mp4", "Processed/c.mp4", ".partial/d.mov", "@eaDir/e.mp4", ".upload.mov"]:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    (tmp_path / "alias.mov").symlink_to(tmp_path / "a.MOV")
    assert {str(p.relative_to(tmp_path)) for p in worker.candidates(tmp_path)} == {"a.MOV", "Originals/b.mp4"}


def test_disconnected_mount_does_not_create_directories(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(worker.OfflineError):
        worker.check_storage({"nas_root": str(missing), "mount_point": str(missing)})
    assert not missing.exists()


def make_result(path):
    path.mkdir()
    (path / "Practice.mp4").write_bytes(b"verified-video")
    worker.atomic_json(path / "session.json", {"source_sha256": "abc", "output_checksums": {"Practice.mp4": worker.sha256(path / "Practice.mp4")}})


def test_publication_is_verifiable_and_restart_safe(tmp_path):
    local, nas = tmp_path / "result", tmp_path / "nas"
    nas.mkdir()
    make_result(local)
    final = worker.publish(local, nas, "session", "abc")
    assert worker.validate_publication(final, "abc")
    assert worker.publish(local, nas, "session", "abc") == final
    (final / "Practice.mp4").write_bytes(b"truncated")
    assert not worker.validate_publication(final, "abc")
    with pytest.raises(RuntimeError):
        worker.publish(local, nas, "session", "abc")


def test_interrupted_upload_replaced(tmp_path):
    local, nas = tmp_path / "result", tmp_path / "nas"
    (nas / "Processed/.upload-session").mkdir(parents=True)
    (nas / "Processed/.upload-session/partial").touch()
    make_result(local)
    assert worker.validate_publication(worker.publish(local, nas, "session", "abc"), "abc")


def test_padding_merges_overlap_and_clamps():
    assert merge_intervals([[1, 2], [3, 4], [9, 9]], 10, 2, 1.5, .5) == [[0, 5.5], [7, 10]]
    with pytest.raises(ValueError):
        merge_intervals([[3, 2]], 10)


def test_failed_job_is_retried_and_then_flagged(tmp_path, monkeypatch):
    root, state = tmp_path / "nas", tmp_path / "state"
    root.mkdir(); (root / "a.mov").write_bytes(b"incomplete")
    cfg = {"nas_root": str(root), "state_dir": str(state), "settle_seconds": 0, "max_attempts": 2}
    def fail(*args):
        raise RuntimeError("decode failed")
    monkeypatch.setattr(worker, "process_source", fail)
    worker.run_once(cfg)
    db = worker.database(state / "jobs.sqlite3")
    row = db.execute("SELECT * FROM files").fetchone()
    assert row["status"] == "retry" and row["attempts"] == 1
    worker.run_once(cfg)
    assert db.execute("SELECT attempts FROM files").fetchone()[0] == 1
    db.execute("UPDATE files SET next_attempt=0"); db.commit()
    worker.run_once(cfg)
    assert db.execute("SELECT status FROM files").fetchone()[0] == "needs_attention"
    assert (root / "a.mov").read_bytes() == b"incomplete"


def test_same_content_under_another_name_is_not_processed_twice(tmp_path, monkeypatch):
    import pipeline
    nas, state = tmp_path / "nas", tmp_path / "state"
    nas.mkdir(); state.mkdir()
    for name in ["a.mov", "b.mov"]:
        (nas / name).write_bytes(b"same original")
    calls = []
    def fake(source, out, cfg, digest, original_name):
        calls.append(original_name)
        out.mkdir(); (out / "Practice.mp4").write_bytes(b"result")
        worker.atomic_json(out / "session.json", {"source_sha256": digest, "output_checksums": {"Practice.mp4": worker.sha256(out / "Practice.mp4")}})
    monkeypatch.setattr(pipeline, "process_video", fake)
    cfg = {"nas_root": str(nas), "state_dir": str(state), "settle_seconds": 0}
    worker.run_once(cfg)
    assert calls == ["a.mov"]
    db = worker.database(state / "jobs.sqlite3")
    assert [r[0] for r in db.execute("SELECT status FROM files")] == ["done", "done"]


def test_abandoned_scratch_is_reclaimed_without_touching_other_files(tmp_path):
    nas, state = tmp_path / "nas", tmp_path / "state"
    nas.mkdir()
    owned = state / "cache" / ("a" * 32)
    owned.mkdir(parents=True); (owned / "source.mov").write_bytes(b"temporary")
    other = state / "cache" / "unrelated"
    other.mkdir(); (other / "keep").touch()
    worker.run_once({"nas_root": str(nas), "state_dir": str(state)})
    assert not owned.exists()
    assert (other / "keep").exists()


def test_offline_status_recovers_on_next_scan(tmp_path):
    state, nas = tmp_path / "state", tmp_path / "nas"
    cfg = {"nas_root": str(nas), "state_dir": str(state)}
    with pytest.raises(worker.OfflineError):
        worker.run_once(cfg)
    assert json.loads((state / "status.json").read_text())["storage"] == "unavailable"
    nas.mkdir()
    worker.run_once(cfg)
    status = json.loads((state / "status.json").read_text())
    assert status["storage"] == "ready" and status["scan_error"] is None

import json
import threading
import time
from pathlib import Path

from monitor import Monitor, status_text, write_snapshot
import worker


def test_local_heartbeat_survives_blocked_nas(tmp_path):
    entered, release = threading.Event(), threading.Event()

    def blocked_check(cfg):
        entered.set()
        release.wait(5)
        raise OSError("NAS offline")

    monitor = Monitor({"state_dir": str(tmp_path)}, blocked_check)
    monitor.stage("Analyzing player movement", 20, 100, "video seconds", "Originals/new.mov")
    monitor.start()
    try:
        assert entered.wait(2)
        deadline = time.monotonic() + 2
        while not (tmp_path / "Processing-status.txt").exists() and time.monotonic() < deadline:
            time.sleep(.01)
        data = json.loads((tmp_path / "Processing-status.json").read_text())
        assert data["current_video"] == "Originals/new.mov"
        assert data["progress"]["current"] == 20
        assert data["heartbeat"] >= data["progress_at"]
        assert "does not guarantee" in (tmp_path / "Processing-status.txt").read_text()
    finally:
        monitor.stop_event.set()
        release.set()


def test_queue_errors_and_stale_progress_are_visible(tmp_path):
    monitor = Monitor({"state_dir": str(tmp_path)}, lambda cfg: tmp_path)
    monitor.stage("Saving and verifying results on NAS", video="a.mov")
    monitor.queue({"updated": time.time(), "storage": "unavailable", "scan_error": "NAS disconnected",
                   "files": [{"path": "b.mov", "status": "retry", "error": "decode failed"}]})
    data = monitor.snapshot()
    data["heartbeat"] = data["progress_at"] + 601
    text = status_text(data)
    assert "over 10 minutes" in text and "NAS disconnected" in text and "decode failed" in text
    write_snapshot(tmp_path, data)
    assert json.loads((tmp_path / "Processing-status.json").read_text())["files"][0]["status"] == "retry"


def test_scan_publishes_complete_queue_and_does_not_ingest_status_files(tmp_path, monkeypatch):
    nas, state = tmp_path / "nas", tmp_path / "state"
    nas.mkdir(); state.mkdir()
    (nas / "a.mov").write_bytes(b"a")
    (nas / "b.mov").write_bytes(b"b")
    cfg = {"nas_root": str(nas), "state_dir": str(state), "settle_seconds": 0}
    monitor = Monitor(cfg, worker.check_storage)
    cfg["_monitor"] = monitor
    seen = []

    def fake_process(cfg, source, row, db):
        seen.append([r["path"] for r in monitor.snapshot()["files"]])
        db.execute("UPDATE files SET status='done' WHERE path=?", (row["path"],))
        db.commit()

    monkeypatch.setattr(worker, "process_source", fake_process)
    worker.run_once(cfg)
    assert seen == [["a.mov", "b.mov"], ["a.mov", "b.mov"]]
    monitor.mirror_once()
    data = json.loads((nas / "Processing-status.json").read_text())
    assert data["stage"] == "Idle" and all(r["status"] == "done" for r in data["files"])
    assert {p.name for p in worker.candidates(nas)} == {"a.mov", "b.mov"}

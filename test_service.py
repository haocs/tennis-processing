import fcntl
import json
from types import SimpleNamespace
import service


def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "ROOT", tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"nas_root": "/nas"}))
    runtime = tmp_path / ".runtime/venv/bin/python"
    runtime.parent.mkdir(parents=True)
    runtime.touch()
    monkeypatch.setattr(service, "loaded", lambda domain: SimpleNamespace(returncode=0, stdout="state = running"))
    monkeypatch.setattr(service.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("Unexpected service mutation")))


def test_start_is_idempotent(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    assert service.main(["start"]) == 0
    assert service.main(["run"]) == 0


def test_restart_refuses_busy_job(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    (tmp_path / ".state").mkdir()
    with (tmp_path / ".state/worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert service.main(["restart"]) == 2

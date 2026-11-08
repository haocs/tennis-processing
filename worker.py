#!/usr/bin/env python3
"""Local, restart-safe NAS video worker. Originals are read-only inputs."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import uuid
from monitor import Monitor, progress

APP = Path(__file__).resolve().parent
LOG = logging.getLogger("tennis")
EXTENSIONS = {".mov", ".mp4", ".m4v"}


def stamp(path):
    s = Path(path).stat()
    return f"{s.st_size}:{s.st_mtime_ns}"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


class OfflineError(RuntimeError):
    pass


def check_storage(cfg):
    """Never create a fake NAS directory when the share is disconnected."""
    root = Path(cfg.get("input_root", cfg.get("nas_root", "")))
    mount = cfg.get("mount_point")
    if mount and not os.path.ismount(mount):
        raise OfflineError(f"NAS is not mounted: {mount}")
    if not root.is_dir():
        raise OfflineError(f"NAS folder is unavailable: {root}")
    expected = cfg.get("storage_id")
    if expected and (root / ".tennis-storage-id").read_text().strip() != expected:
        raise OfflineError("NAS identity does not match configured storage")
    return root


def ensure_storage(cfg):
    try:
        return check_storage(cfg)
    except (OfflineError, OSError):
        # Ask macOS to reconnect using its saved SMB credentials, never a password
        # in this config. Throttle attempts so an offline NAS does not cause loops.
        state = Path(cfg["state_dir"])
        marker = state / "last-reconnect"
        last = marker.stat().st_mtime if marker.exists() else 0
        if cfg.get("smb_url") and not os.path.ismount(cfg["mount_point"]) and time.time() - last > 600:
            marker.touch()
            LOG.info("Reconnecting to the NAS using macOS saved credentials")
            try:
                subprocess.run(["/usr/bin/osascript", "-e", "on run argv", "-e", "mount volume (item 1 of argv)", "-e", "end run", cfg["smb_url"]], capture_output=True, timeout=45, check=True)
            except (subprocess.SubprocessError, OSError) as exc:
                LOG.warning("NAS reconnect unavailable: %s", exc)
        return check_storage(cfg)


def candidates(root, excluded=None):
    excluded = Path(excluded).resolve() if excluded else None
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in {"Processed", "@eaDir", "#recycle"} and not (Path(directory) / d).is_symlink() and (Path(directory) / d).resolve() != excluded)
        for name in sorted(files):
            p = Path(directory) / name
            if not name.startswith(".") and p.suffix.lower() in EXTENSIONS and not p.is_symlink():
                yield p


def database(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, signature TEXT, stable_since REAL, status TEXT, attempts INTEGER DEFAULT 0, next_attempt REAL DEFAULT 0, digest TEXT, output TEXT, error TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS jobs (digest TEXT PRIMARY KEY, output TEXT, completed REAL)")
    db.commit()
    return db


def observe(db, relative, signature, now):
    row = db.execute("SELECT * FROM files WHERE path=?", (relative,)).fetchone()
    if not row or row["signature"] != signature:
        db.execute("INSERT OR REPLACE INTO files(path,signature,stable_since,status) VALUES(?,?,?,'waiting')", (relative, signature, now))
        db.commit()
    return db.execute("SELECT * FROM files WHERE path=?", (relative,)).fetchone()


def write_status(db, state, storage="ready", error=None, cfg=None):
    rows = [dict(r) for r in db.execute("SELECT path,status,attempts,output,error FROM files ORDER BY path")]
    data = {"updated": time.time(), "storage": storage, "scan_error": error, "files": rows}
    atomic_json(state / "status.json", data)
    if cfg and cfg.get("_monitor"):
        cfg["_monitor"].queue(data)


def validate_publication(dest, digest):
    """Only a complete, byte-verified publication counts as a finished job."""
    try:
        if dest.parent.name == "media" and not (dest / "session.json").exists():
            from publication import validate_library
            return validate_library(dest, digest)
        manifest = json.loads((dest / "session.json").read_text())
        if manifest["source_sha256"] != digest:
            return False
        for name, expected in manifest["output_checksums"].items():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                return False
            p = dest / name
            if not p.is_relative_to(dest) or sha256(p) != expected:
                return False
            if name == "heuristic-diagnostics.json":
                from diagnostics import validate_heuristic_diagnostics
                validate_heuristic_diagnostics(json.loads(p.read_text()), expected_source_sha256=digest)
        return bool(manifest["output_checksums"])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def publish(local, root, name, digest):
    check = root / "Processed"
    check.mkdir(exist_ok=True)
    final = check / name
    if final.exists():
        if validate_publication(final, digest):
            return final
        raise RuntimeError(f"Existing output failed verification; retained for inspection: {final}")
    temp = check / (".upload-" + name)
    # This name belongs to this worker/job only; interrupted uploads are replaceable.
    if temp.exists():
        shutil.rmtree(temp)
    shutil.copytree(local, temp)
    if not validate_publication(temp, digest):
        raise RuntimeError("NAS copy failed checksum verification")
    temp.rename(final)
    return final


def process_source(cfg, source, row, db):
    from timings import JobTiming, save_report
    cfg = dict(cfg)  # Per-job instrumentation must not leak into later scans.
    timing = cfg["_timing"] = JobTiming(source.name, row["stable_since"], row["attempts"] + 1)
    status = "failed"
    try:
        result = _process_source(cfg, source, row, db)
        status = "done"
        return result
    finally:
        try:
            save_report(cfg, timing, status)
        except Exception:
            # Diagnostics must never turn a committed publication into a retry.
            LOG.exception("Could not save processing timing report")


def _process_source(cfg, source, row, db):
    from pipeline import process_video
    root = check_storage(cfg)
    if cfg.get("library_root"):
        from storage_config import check_library_storage
        check_library_storage(cfg)
    before = stamp(source)
    if before != row["signature"]:
        raise RuntimeError("Source changed before copying; waiting for upload")
    cache = Path(cfg["state_dir"]) / "cache"
    cache.mkdir(exist_ok=True)
    # Unique per job; originals on NAS are never moved or deleted.
    scratch = cache / uuid.uuid4().hex
    scratch.mkdir()
    try:
        progress(cfg, "Copying original from NAS", video=str(source.relative_to(root)))
        if shutil.disk_usage(cache).free < source.stat().st_size * 3 + 1024**3:
            raise RuntimeError("Not enough local scratch space (need 3× source size + 1 GB)")
        local = scratch / ("source" + source.suffix.lower())
        shutil.copyfile(source, local)
        if stamp(source) != before:
            raise RuntimeError("Source changed while copying; retrying after upload")
        digest = sha256(local)
        cfg["_timing"].digest = digest
        progress(cfg, "Checking for an existing result")
        known = db.execute("SELECT output FROM jobs WHERE digest=?", (digest,)).fetchone()
        if known and validate_publication(Path(known["output"]), digest):
            cfg["_timing"].reused = True
            output = Path(known["output"])
        else:
            safe_stem = re.sub(r"[^\w.-]+", "_", source.stem)[:90]
            name = f"{safe_stem}--{digest[:12]}"
            existing = (Path(cfg["library_root"]) / "media" / digest) if cfg.get("library_root") else root / "Processed" / name
            if validate_publication(existing, digest):
                cfg["_timing"].reused = True
                output = existing  # Recover publication completed before a crash.
            else:
                local_out = scratch / "result"
                report = process_video(local, local_out, cfg, digest, str(source.relative_to(root)))
                if report:
                    meta = report.get("source_metadata", {})
                    cfg["_timing"].info = {k: meta[k] for k in ("duration", "width", "height", "fps", "transfer") if k in meta}
                    cfg["_timing"].info["clips"] = report.get("clips")
                progress(cfg, "Verifying original on NAS")
                if stamp(source) != before or sha256(source) != digest:
                    raise RuntimeError("Source changed during analysis; results not published")
                check_storage(cfg)
                progress(cfg, "Saving and verifying results on NAS")
                if cfg.get("library_root"):
                    from publication import publish_library
                    from storage_config import check_library_storage
                    output = publish_library(local_out, check_library_storage(cfg), name, digest)
                else:
                    output = publish(local_out, root, name, digest)
            db.execute("INSERT OR REPLACE INTO jobs VALUES(?,?,?)", (digest, str(output), time.time()))
        db.execute("UPDATE files SET status='done',digest=?,output=?,error=NULL WHERE path=?", (digest, str(output), row["path"]))
        db.commit()
        LOG.info("Completed %s → %s", source.name, output)
    finally:
        progress(cfg, "Cleaning local scratch")
        shutil.rmtree(scratch, ignore_errors=True)


def run_once(cfg):
    state = Path(cfg["state_dir"])
    state.mkdir(parents=True, exist_ok=True)
    with (state / "worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            LOG.info("Another worker is already running")
            return
        # The lock proves no other worker owns these private scratch directories.
        # Recover space after a process kill/power loss that bypassed finally.
        cache = state / "cache"
        if cache.exists():
            for abandoned in cache.iterdir():
                if abandoned.is_dir() and not abandoned.is_symlink() and re.fullmatch(r"[0-9a-f]{32}", abandoned.name):
                    shutil.rmtree(abandoned)
        db = database(state / "jobs.sqlite3")
        scan_error = None
        try:
            progress(cfg, "Checking NAS")
            root = ensure_storage(cfg)
            if cfg.get("library_root"):
                from storage_config import ensure_library_storage
                ensure_library_storage(cfg)
            now = time.time()
            sources = list(candidates(root, cfg.get("library_root")))
            for source in sources:
                observe(db, str(source.relative_to(root)), stamp(source), now)
            write_status(db, state, cfg=cfg)
            for source in sources:
                relative = str(source.relative_to(root))
                row = observe(db, relative, stamp(source), now)
                if row["status"] in {"done", "needs_attention"} or now < row["next_attempt"]:
                    continue
                if now - row["stable_since"] < cfg.get("settle_seconds", 180):
                    continue
                LOG.info("Processing %s", relative)
                db.execute("UPDATE files SET status='processing' WHERE path=?", (relative,))
                db.commit()
                write_status(db, state, cfg=cfg)
                try:
                    process_source(cfg, source, row, db)
                except Exception as exc:
                    attempt = row["attempts"] + 1
                    status = "needs_attention" if attempt >= cfg.get("max_attempts", 5) else "retry"
                    db.execute("UPDATE files SET status=?,attempts=?,next_attempt=?,error=? WHERE path=?", (status, attempt, time.time() + min(3600, 60 * 2**attempt), str(exc), relative))
                    db.commit()
                    LOG.exception("Could not finish %s", relative)
                write_status(db, state, cfg=cfg)
            progress(cfg, "Idle")
        except Exception as exc:
            scan_error = str(exc)
            progress(cfg, "NAS unavailable; waiting to retry")
            raise
        finally:
            write_status(db, state, "unavailable" if scan_error else "ready", scan_error, cfg=cfg)
            db.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["once", "watch", "status", "retry"])
    p.add_argument("--config", default=str(APP / "config.json"))
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    state = Path(cfg["state_dir"])
    state.mkdir(parents=True, exist_ok=True)
    from logging.handlers import RotatingFileHandler
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(), RotatingFileHandler(state / "worker.log", maxBytes=2_000_000, backupCount=3)])
    if args.command == "status":
        live = Path(cfg.get("status_dir", cfg["state_dir"])) / "Processing-status.txt"
        if live.exists():
            print(live.read_text())
            return
        path = state / "status.json"
        if not path.exists():
            print("Waiting for the first scan.")
            return
        status = json.loads(path.read_text())
        print("Last scan:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(status["updated"])))
        print("NAS:", "connected" if status.get("storage") == "ready" else "unavailable")
        if status.get("scan_error"):
            print("Connection issue:", status["scan_error"])
        labels = {"done": "Finished", "waiting": "Waiting for upload to finish", "processing": "Processing", "retry": "Will retry", "needs_attention": "Needs attention"}
        for row in status["files"]:
            print(f"\n{row['path']}: {labels.get(row['status'], row['status'])}")
            if row["output"]:
                print("  Saved to:", row["output"])
            if row["error"]:
                print("  Issue:", row["error"])
        return
    if args.command == "retry":
        with database(state / "jobs.sqlite3") as db:
            db.execute("UPDATE files SET status='waiting',attempts=0,next_attempt=0 WHERE status IN ('retry','needs_attention')")
        return
    cfg["_monitor"] = Monitor(cfg, check_storage)
    cfg["_monitor"].start()
    while True:
        try:
            run_once(cfg)
        except Exception:
            LOG.exception("Scan failed; will retry on the next run")
            if args.command == "once":
                raise
        if args.command != "watch":
            return
        time.sleep(cfg.get("poll_seconds", 60))


if __name__ == "__main__":
    main()

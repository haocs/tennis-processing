"""Independent heartbeat with a best-effort NAS status mirror.

The NAS writer has its own thread so a stalled network filesystem cannot stop
local heartbeats. Heartbeats indicate service liveness, not job progress.
"""
import copy
import json
import logging
import os
from pathlib import Path
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

LOG = logging.getLogger("tennis")
ZONE = ZoneInfo("America/Los_Angeles")


def clock_text(value):
    return datetime.fromtimestamp(value, ZONE).strftime("%Y-%m-%d %H:%M:%S %Z") if value else "Not yet"


def status_text(data):
    detail = data.get("progress") or {}
    lines = ["TENNIS PROCESSING STATUS", "", f"Heartbeat: {clock_text(data['heartbeat'])}",
             "If this heartbeat is more than 2 minutes old, this status is stale.",
             "A fresh heartbeat means the service is alive; it does not guarantee the current operation is advancing.",
             f"Service: {data['service']}", f"Current stage: {data['stage']}",
             f"Current video: {data.get('current_video') or 'None'}",
             f"Last progress: {clock_text(data.get('progress_at'))}"]
    if detail:
        lines.append(f"Stage progress: {detail['current']} / {detail['total']} {detail['unit']}")
    if data.get("stage") not in {"Idle", "Starting", "Stopped"} and data['heartbeat'] - data.get('progress_at', data['heartbeat']) > 600:
        lines.append("No progress update for over 10 minutes. A long operation or NAS connection may be delaying work.")
    lines += [f"Last NAS scan: {clock_text(data.get('scan_at'))}", f"NAS at last scan: {data.get('storage', 'unknown')}"]
    if data.get("scan_error"):
        lines.append(f"Scan error: {data['scan_error']}")
    if data.get("mirror_error"):
        lines.append(f"NAS status-file error: {data['mirror_error']}")
    lines += ["", "VIDEOS"]
    labels = {"done": "Finished", "waiting": "Waiting for upload to settle (3 minutes)", "processing": "Processing", "retry": "Waiting to retry", "needs_attention": "Needs attention"}
    for row in data.get("files", []):
        lines.append(f"{row['path']}: {labels.get(row['status'], row['status'])}")
        if row.get("output"):
            lines.append(f"  Saved to: {row['output']}")
        if row.get("error"):
            lines.append(f"  Error: {row['error']}")
    lines += ["", "Refresh/reopen this file to see updates. The Mac must be logged in and awake.",
              "Originals are retained. Finished videos appear under Processed."]
    return "\n".join(lines) + "\n"


def public_snapshot(data):
    data = copy.deepcopy(data)
    data.pop("pid", None)
    if data.get("current_video"):
        data["current_video"] = Path(data["current_video"]).name
    for row in data.get("files", []):
        row["path"] = Path(row["path"]).name
        row["output"] = "Published" if row.get("output") else None
        if row.get("error"):
            row["error"] = "See the processor log on the Mac for details."
    for key in ("scan_error", "mirror_error"):
        if data.get(key):
            data[key] = "See the processor log on the Mac for details."
    return data


def write_snapshot(folder, data):
    folder = Path(folder)
    for name, content in [("Processing-status.json", json.dumps(data, indent=2) + "\n"),
                          ("Processing-status.txt", status_text(data))]:
        temp = folder / ("." + name + ".tmp")
        temp.write_text(content)
        temp.replace(folder / name)


class Monitor:
    def __init__(self, cfg, check_storage):
        self.cfg = cfg
        self.check_storage = check_storage
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        now = time.time()
        self.data = {"service": "Running", "pid": os.getpid(), "stage": "Starting",
                     "current_video": None, "progress_at": now, "heartbeat": now,
                     "files": [], "timezone": "America/Los_Angeles"}
        self.local_folder = Path(cfg.get("status_dir", cfg["state_dir"]))

    def snapshot(self):
        with self.lock:
            result = copy.deepcopy(self.data)
        result["heartbeat"] = time.time()
        return result

    def stage(self, stage, current=None, total=None, unit=None, video=None):
        with self.lock:
            self.data.update(stage=stage, progress_at=time.time(),
                             progress={"current": current, "total": total, "unit": unit} if total is not None else None)
            if video is not None:
                self.data["current_video"] = video
            if stage in {"Idle", "Checking NAS", "Stopped"}:
                self.data["current_video"] = None
        LOG.info("Stage: %s%s", stage, f" ({current}/{total} {unit})" if total is not None else "")

    def queue(self, data):
        with self.lock:
            self.data.update(files=copy.deepcopy(data["files"]), scan_at=data["updated"],
                             storage=data["storage"], scan_error=data["scan_error"])

    def local_loop(self):
        while not self.stop_event.is_set():
            try:
                write_snapshot(self.local_folder, self.snapshot())
            except OSError:
                LOG.exception("Could not update local status files")
            self.stop_event.wait(15)

    def mirror_once(self):
        from storage_config import status_destination
        root = status_destination(self.cfg, self.check_storage)
        data = public_snapshot(self.snapshot()) if self.cfg.get("library_root") else self.snapshot()
        write_snapshot(root, data)

    def nas_loop(self):
        while not self.stop_event.is_set():
            try:
                self.mirror_once()
                with self.lock:
                    self.data["mirror_error"] = None
            except (OSError, RuntimeError) as exc:
                with self.lock:
                    previous = self.data.get("mirror_error")
                    self.data["mirror_error"] = str(exc)
                if previous != str(exc):
                    LOG.warning("NAS status mirror unavailable: %s", exc)
            self.stop_event.wait(30)

    def start(self):
        # Never start another mirror while a previous SMB write is blocked.
        for target in (self.local_loop, self.nas_loop):
            threading.Thread(target=target, daemon=True).start()


def progress(cfg, stage, current=None, total=None, unit=None, video=None):
    monitor = cfg.get("_monitor")
    if monitor:
        monitor.stage(stage, current, total, unit, video)

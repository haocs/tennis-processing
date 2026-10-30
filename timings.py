"""Per-attempt wall-clock timings; no private filesystem paths in reports."""
from __future__ import annotations
from datetime import datetime, timezone
import logging
import time
import uuid
from pathlib import Path

LOG = logging.getLogger("tennis")


class JobTiming:
    def __init__(self, source_name, observed_at, attempt, clock=time.monotonic, wall=time.time):
        self.clock, self.wall = clock, wall
        self.started = wall()
        self.tick = self.begin = clock()
        self.stage = "Starting"
        self.stages = []
        self.digest = None
        self.reused = False
        self.info = {}
        self.id = uuid.uuid4().hex
        self.source_name = source_name
        self.observed_at = observed_at
        self.attempt = attempt

    def transition(self, stage):
        if stage == self.stage:
            return
        now = self.clock()
        self.stages.append({"stage": self.stage, "seconds": round(now - self.tick, 3)})
        self.stage, self.tick = stage, now

    def report(self, status):
        now = self.clock()
        stages = self.stages + [{"stage": self.stage, "seconds": round(now - self.tick, 3)}]
        return {"schema_version": 1, "attempt_id": self.id, "source_name": self.source_name,
                "source_sha256": self.digest, "result_reused": self.reused, "attempt": self.attempt, "status": status,
                "started_at": datetime.fromtimestamp(self.started, timezone.utc).isoformat(),
                "finished_at": datetime.fromtimestamp(self.wall(), timezone.utc).isoformat(),
                "observed_to_start_seconds": round(max(0, self.started - self.observed_at), 3),
                "total_seconds": round(now - self.begin, 3), "stages": stages,
                "source": self.info,
                "measurement": "One attempt: local copy through cleanup, including NAS publication. Excludes queue waiting and writing this report. Observed-to-start includes upload settling and retry backoff; observation resets when the source changes."}


def save_report(cfg, timing, status):
    from worker import atomic_json
    report = timing.report(status)
    folder = Path(cfg["state_dir"]) / "timings"
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder / (timing.id + ".json"), report)
    if timing.digest and cfg.get("library_root"):
        try:
            from storage_config import check_library_storage
            folder = check_library_storage(cfg) / "metadata/processing" / timing.digest
            folder.mkdir(parents=True, exist_ok=True)
            atomic_json(folder / (timing.id + ".json"), report)
        except (OSError, RuntimeError):
            LOG.warning("Timing report retained locally; NAS timing mirror unavailable", exc_info=True)
    LOG.info("Timing %s: %s, %.1f seconds; report %s", timing.source_name, status, report["total_seconds"], timing.id)


def print_summary(cfg):
    """CSV output can be redirected to a spreadsheet-friendly file."""
    import csv
    import json
    import sys
    rows = []
    for path in (Path(cfg['state_dir']) / 'timings').glob('*.json'):
        try:
            rows.append(json.loads(path.read_text()))
        except (ValueError, OSError):
            LOG.warning('Skipping unreadable timing report: %s', path.name)
    writer = csv.writer(sys.stdout)
    writer.writerow(['video', 'attempt', 'status', 'started_at', 'total_seconds', 'observed_to_start_seconds', 'source_seconds', 'stage', 'stage_seconds'])
    for row in sorted(rows, key=lambda r: r['started_at']):
        for stage in row['stages']:
            writer.writerow([row['source_name'], row['attempt'], row['status'], row['started_at'], row['total_seconds'], row['observed_to_start_seconds'], row['source'].get('duration', ''), stage['stage'], stage['seconds']])

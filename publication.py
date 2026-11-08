"""Publish a portable library with separate media and public metadata.

No source filesystem paths, raw private job logs, or credentials are published.
"""
from pathlib import Path
import hashlib
import json
import math
import re
import shutil
import subprocess
import time
import uuid


def clip_id(digest, start, end):
    return hashlib.sha256(f"{digest}:{start:.6f}:{end:.6f}".encode()).hexdigest()[:24]


def public_session(report, playback, media_directory=None):
    digest = report["source_sha256"]
    if not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("Invalid source identity")
    source = report["source_metadata"]
    clips = []
    for item in playback["clips"]:
        item = dict(item)
        item["id"] = clip_id(digest, item["original_start_seconds"], item["original_end_seconds"])
        clips.append(item)
    published_at = report.get("published_at")
    if isinstance(published_at, bool) or not isinstance(published_at, (int, float)) or not math.isfinite(published_at):
        published_at = time.time()
        report["published_at"] = published_at
    artifacts = ["edit-timeline.csv", "edit-timeline.json", "timestamps.csv", "Review.html"]
    if report.get("heuristic_diagnostics") == "heuristic-diagnostics.json":
        artifacts.append("heuristic-diagnostics.json")
    return {"schema_version": 1, "id": digest, "title": Path(report["source"]).stem,
            "source": {"name": Path(report["source"]).name, "sha256": digest,
                       "recorded_at": source.get("recorded_at", ""), "duration_seconds": source["duration"],
                       "location": source.get("location")},
            "published_at": published_at, "media_directory": media_directory or digest,
            "video": {"file": "Practice.mp4", "poster": "poster.jpg", "duration_seconds": playback["duration_seconds"],
                      "width": source["width"], "height": source["height"], "hdr": source["transfer"] in (16, 18)},
            "clips": clips, "kept_seconds": report["kept_seconds"], "removed_seconds": report["removed_seconds"],
            "chapter_alignment": playback["alignment"], "algorithm": report["analysis"].get("detector", "unknown"),
            "encoding": report.get("encoding", {}),
            "artifacts": artifacts}


def poster(session, destination):
    import imageio_ffmpeg
    first = Path(session) / "Clips/clip-001.mp4"
    from pipeline import probe
    meta = probe(first)
    tone = "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=mobius:desat=0.5,zscale=t=bt709:m=bt709:r=tv," if meta["transfer"] in (16, 18) else ""
    from pipeline import run_media
    run_media(["-ss", str(min(2, max(0, meta["duration"] / 2))), "-i", str(first), "-frames:v", "1",
                    "-vf", tone + "scale=960:-2", "-q:v", "3", str(destination)], Path(session).parent / "ffmpeg.log", timeout=90)


def metadata_for_session(session):
    from playback import build_playback
    session = Path(session)
    report = json.loads((session / "session.json").read_text())
    if (session / "heuristic-diagnostics.json").is_file() and "heuristic_diagnostics" not in report:
        report["heuristic_diagnostics"] = "heuristic-diagnostics.json"
    playback = report.get("playback") or build_playback(session, report["intervals"])
    return report, public_session(report, playback)


def publish_library(local, root, name=None, digest=None):
    from worker import atomic_json, sha256
    root, local = Path(root), Path(local)
    report, public = metadata_for_session(local)
    identity = public["id"]
    if digest and digest != identity:
        raise ValueError("Source identity mismatch")
    media = root / "media" / identity
    metadata = root / "metadata/sessions" / identity
    manifest = root / "metadata/sessions" / f"{identity}.json"
    for folder in (media.parent, metadata.parent):
        folder.mkdir(parents=True, exist_ok=True)
    if manifest.exists():
        if validate_library(media, identity):
            return media
        raise ValueError("Existing shared publication is invalid; retained for inspection")
    if media.exists() or metadata.exists():
        # A crash before the JSON commit marker left an unpublished generation.
        # Preserve it for inspection, then publish a complete replacement.
        orphan = root / (".orphan-" + identity + "-" + uuid.uuid4().hex)
        orphan.mkdir()
        if media.exists():
            media.rename(orphan / "media")
        if metadata.exists():
            metadata.rename(orphan / "metadata")
    stage = root / (".publishing-" + uuid.uuid4().hex)
    stage.mkdir()
    try:
        (stage / "media/Clips").mkdir(parents=True)
        (stage / "metadata").mkdir()
        for relative in ["Practice.mp4"] + [clip["file"] for clip in public["clips"]]:
            shutil.copyfile(local / relative, stage / "media" / relative)
            expected = report["output_checksums"][relative]
            if sha256(stage / "media" / relative) != expected:
                raise ValueError(f"Media checksum failed: {relative}")
        poster(local, stage / "media/poster.jpg")
        for artifact in public["artifacts"]:
            if (local / artifact).exists():
                shutil.copyfile(local / artifact, stage / "metadata" / artifact)
                if artifact == "edit-timeline.json":
                    timeline = json.loads((stage / "metadata" / artifact).read_text())
                    timeline["source"] = Path(report["source"]).name
                    atomic_json(stage / "metadata" / artifact, timeline)
                elif artifact == "heuristic-diagnostics.json":
                    from diagnostics import bind_publication_diagnostics
                    bind_publication_diagnostics(stage / "metadata" / artifact, public)
        public["media_checksums"] = {str(p.relative_to(stage / "media")): sha256(p) for p in (stage / "media").rglob("*") if p.is_file()}
        public["metadata_checksums"] = {p.name: sha256(p) for p in (stage / "metadata").iterdir() if p.is_file()}
        (stage / "media").rename(media)
        try:
            (stage / "metadata").rename(metadata)
            atomic_json(manifest, public)  # Commit marker; website only sees sessions after this.
        except Exception:
            if metadata.exists():
                metadata.rename(stage / "metadata")
            media.rename(stage / "media")
            raise
        return media
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def validate_library(media, digest):
    from worker import sha256
    try:
        root = Path(media).parent.parent
        manifest = json.loads((root / "metadata/sessions" / f"{digest}.json").read_text())
        if manifest["id"] != digest or not manifest["media_checksums"]:
            return False
        for name, expected in manifest["media_checksums"].items():
            p = (Path(media) / name).resolve()
            if not p.is_relative_to(Path(media).resolve()) or sha256(p) != expected:
                return False
        for name, expected in manifest["metadata_checksums"].items():
            p = (root / "metadata/sessions" / digest / name).resolve()
            if not p.is_relative_to((root / "metadata/sessions" / digest).resolve()) or sha256(p) != expected:
                return False
            if name == "heuristic-diagnostics.json":
                from diagnostics import validate_heuristic_diagnostics
                validate_heuristic_diagnostics(json.loads(p.read_text()), expected_source_sha256=digest)
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False

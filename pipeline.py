"""Decode, detect, export and verify a session entirely on this Mac."""
from __future__ import annotations
import csv
import html
import json
import logging
import math
from pathlib import Path
import subprocess
import time

import av
import imageio_ffmpeg

from detector import detect
from timeline import write_timeline
from monitor import progress
from color import export_profile, verify_color
from playback import build_playback
import re

LOG = logging.getLogger("tennis")
EXECUTION_GUARD = None


def location_metadata(metadata):
    raw = metadata.get("com.apple.quicktime.location.ISO6709", "")
    match = re.fullmatch(r"([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)(?:([+-]\d+(?:\.\d+)?))?/?", raw)
    if not match:
        return None
    latitude, longitude = float(match[1]), float(match[2])
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        return None
    point = {"latitude": latitude, "longitude": longitude}
    if match[3] is not None:
        altitude = float(match[3])
        if math.isfinite(altitude):
            point["altitude_m"] = altitude
    return point


def original_metadata(container, first_frame=None):
    """Keep original tags/stream properties separately from edited media semantics."""
    import base64
    streams = []
    for stream in container.streams:
        context = stream.codec_context
        item = {"type": stream.type, "index": stream.index, "metadata": dict(stream.metadata),
                "time_base": str(stream.time_base), "duration": stream.duration,
                "codec": getattr(context, "name", None)}
        for key in ("width", "height", "sample_rate", "channels", "bit_rate", "color_trc", "color_primaries", "colorspace", "color_range"):
            try:
                value = getattr(context, key, None)
                if isinstance(value, (str,int,float,bool)) or value is None: item[key] = value
            except (AttributeError, ValueError): pass
        extra = getattr(context, "extradata", None)
        if extra: item["codec_extradata_base64"] = base64.b64encode(extra).decode("ascii")
        streams.append(item)
    frame_data = []
    if first_frame is not None:
        for side in first_frame.side_data.values():
            try:
                frame_data.append({"type":str(side.type),"base64":base64.b64encode(bytes(side)).decode("ascii")})
            except (TypeError, ValueError): pass
    return {"container_tags": dict(container.metadata), "streams": streams,
            "rotation_degrees": getattr(first_frame,"rotation",None), "first_frame_side_data":frame_data,
            "note": "Original metadata snapshot, not rewritten clip metadata. Originals remain the authoritative source, including binary vendor/codec metadata not exposed by this decoder."}


def probe(path):
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        duration = float(s.duration * s.time_base) if s.duration is not None else float(c.duration / av.time_base)
        if duration <= 0:
            raise ValueError("Video has no positive duration")
        first_frame = next(c.decode(s))
        return {"duration": duration, "width": s.width, "height": s.height, "fps": float(s.average_rate or 30), "transfer": int(s.codec_context.color_trc), "primaries": int(s.codec_context.color_primaries), "matrix": int(s.codec_context.colorspace), "range": int(s.codec_context.color_range), "pixel_format": s.codec_context.pix_fmt, "audio": bool(c.streams.audio), "location": location_metadata(c.metadata), "original_metadata": original_metadata(c, first_frame), "recorded_at": c.metadata.get("com.apple.quicktime.creationdate", c.metadata.get("creation_time", ""))}


def run_media(args, log_path, timeout=7200):
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-nostdin", "-loglevel", "error", "-y"] + list(args)
    with Path(log_path).open("ab") as log:
        with subprocess.Popen(command, stdout=log, stderr=log) as process:
            started = time.monotonic()
            try:
                while True:
                    if EXECUTION_GUARD: EXECUTION_GUARD()
                    try:
                        code = process.wait(timeout=1)
                        break
                    except subprocess.TimeoutExpired:
                        if time.monotonic() - started > timeout: raise TimeoutError("Video operation exceeded timeout")
            except BaseException:
                process.terminate()
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
                raise
    if code:
        tail = Path(log_path).read_text(errors="replace")[-1500:]
        raise RuntimeError(f"Video processing failed: {tail}")


def verify_video(path, expected, log):
    meta = probe(path)
    if abs(meta["duration"] - expected) > max(0.35, expected * 0.005):
        raise RuntimeError(f"Export duration mismatch: {path.name}")
    run_media(["-xerror", "-i", str(path), "-map", "0:v:0", "-map", "0:a:0?", "-f", "null", "-"], log)


def prepare_video(source, scratch, cfg):
    meta = probe(source)
    log = Path(scratch) / "ffmpeg.log"
    tone = "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=mobius:desat=0,zscale=t=bt709:m=bt709:r=tv," if meta["transfer"] in (16, 18) else ""
    proxy = Path(scratch) / "analysis.mp4"
    # Keep the tested filter order until a separate quality/performance benchmark.
    run_media(["-xerror", "-i", str(source), "-map", "0:v:0", "-an", "-vf", tone + "scale=w='min(960,iw)':h=-2,fps=8,format=yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", str(proxy)], log)
    if abs(probe(proxy)["duration"] - meta["duration"]) > 0.5:
        raise RuntimeError("Source could not be decoded completely")
    return meta, proxy


def analyze_video(proxy, meta, cfg):
    intervals, analysis = detect(proxy, cfg)
    if not intervals: raise RuntimeError("No practice motion detected. Original preserved; check camera framing or detector settings.")
    if analysis["target_visible_fraction"] < 0.10: raise RuntimeError("Near player visible in fewer than 10% of frames; camera profile needs adjustment")
    return [[a, min(b, meta["duration"])] for a, b in intervals if a < meta["duration"]], analysis


def process_video(source, out, cfg, digest, original_name):
    started = time.monotonic()
    progress(cfg, "Preparing video for analysis", video=original_name)
    out.mkdir(parents=True, exist_ok=True)
    meta, proxy = prepare_video(source, out.parent, cfg)
    progress(cfg, "Analyzing player movement")
    intervals, analysis = analyze_video(proxy, meta, cfg)
    return render_session(source, out, cfg, digest, original_name, intervals, analysis, meta, started)


def render_session(source, out, cfg, digest, original_name, intervals, analysis, meta=None, started=None):
    """Export existing decisions without re-running motion detection."""
    from worker import atomic_json, sha256
    started = time.monotonic() if started is None else started
    meta = probe(source) if meta is None else meta
    out.mkdir(parents=True, exist_ok=True)
    log = out.parent / "ffmpeg.log"
    color_filters, encoder_args, encoding = export_profile(meta)
    clips = out / "Clips"
    clips.mkdir(exist_ok=True)
    fps = 60 if meta["fps"] >= 45 else 30
    for index, (start, end) in enumerate(intervals, 1):
        progress(cfg, "Exporting clips", index, len(intervals), "clips")
        path = clips / f"clip-{index:03d}.mp4"
        LOG.info("Exporting clip %d/%d (%.1f–%.1f s)", index, len(intervals), start, end)
        run_media(["-ss", f"{start:.6f}", "-i", str(source), "-t", f"{end-start:.6f}", "-map", "0:v:0", "-map", "0:a:0?", "-map_metadata", "-1", "-vf", f"scale=trunc(iw/2)*2:trunc(ih/2)*2,fps={fps}," + color_filters] + encoder_args + ["-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", "-metadata", f"creation_time={meta['recorded_at']}", str(path)], log)
        verify_video(path, end - start, log)
        verify_color(meta, probe(path))
    progress(cfg, "Combining clips")
    concat = out / "concat.txt"
    concat.write_text("".join(f"file 'Clips/clip-{i:03d}.mp4'\n" for i in range(1, len(intervals)+1)))
    combined = out / "Practice.mp4"
    run_media(["-f", "concat", "-safe", "1", "-i", str(concat), "-c", "copy", "-movflags", "+faststart", str(combined)], log)
    kept = sum(b-a for a, b in intervals)
    # AAC packet rounding accumulates across joined clips, so use measured durations.
    expected_join = sum(probe(p)["duration"] for p in sorted(clips.glob("*.mp4")))
    progress(cfg, "Verifying combined video")
    verify_video(combined, expected_join, log)
    verify_color(meta, probe(combined))
    concat.unlink()
    with (out / "timestamps.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip", "original_start_seconds", "original_end_seconds"])
        writer.writerows((f"clip-{i:03d}.mp4", f"{a:.3f}", f"{b:.3f}") for i, (a, b) in enumerate(intervals, 1))
    from diagnostics import build_heuristic_diagnostics, load_local_observations
    from publication import public_session
    raw_analysis = dict(analysis or {})
    inline_observations = raw_analysis.pop("_frame_observations", None) or cfg.get("_frame_observations")
    inline_records = raw_analysis.pop("pose_records", None) or raw_analysis.pop("_pose_records", None) or cfg.get("pose_records")
    local_observations, local_proxy_meta = load_local_observations(out.parent / "pose-observations.json")
    if local_proxy_meta and "proxy_observation_meta" not in raw_analysis:
        raw_analysis["proxy_observation_meta"] = local_proxy_meta
    analysis_for_diag = dict(raw_analysis)
    if inline_records is not None:
        analysis_for_diag["pose_records"] = inline_records
    published_at = time.time()
    report = {"source": original_name, "source_sha256": digest, "source_metadata": meta, "published_at": published_at, "kept_seconds": round(kept, 3), "removed_seconds": round(meta["duration"]-kept, 3), "clips": len(intervals), "intervals": intervals, "analysis": raw_analysis, "processing_seconds": round(time.monotonic()-started, 1), "settings": {k: cfg[k] for k in ("target_region", "min_player_height", "swing_speed", "arm_reach", "padding_before", "padding_after", "merge_gap") if k in cfg}}
    report["encoding"] = encoding
    progress(cfg, "Aligning playback chapters")
    report["playback"] = build_playback(out, intervals)
    progress(cfg, "Writing timeline and checksums")
    write_review(out, report)
    report["heuristic_diagnostics"] = "heuristic-diagnostics.json"
    frozen_public = public_session(report, report["playback"])
    diagnostics_payload = build_heuristic_diagnostics(
        source_sha256=digest,
        duration_seconds=meta["duration"],
        intervals=intervals,
        cfg=cfg,
        meta=meta,
        analysis=analysis_for_diag,
        frame_observations=inline_observations if inline_observations is not None else local_observations,
        public=frozen_public,
        task_id=cfg.get("_task_id"),
        attempt_id=cfg.get("_attempt_id"),
    )
    atomic_json(out / "heuristic-diagnostics.json", diagnostics_payload)
    report["output_checksums"] = {str(p.relative_to(out)): sha256(p) for p in out.rglob("*") if p.is_file()}
    atomic_json(out / "session.json", report)
    return report


def write_review(out, report):
    segments = write_timeline(out, report)
    timeline_rows = "".join(
        f'<tr><td>{segment["action"].capitalize()}</td>'
        f'<td>{segment["original_start"]}</td><td>{segment["original_end"]}</td>'
        f'<td>{segment["duration_seconds"]:.3f} s</td><td>'
        + (f'<a href="{segment["clip"]}">{Path(segment["clip"]).stem}</a>' if segment["clip"] else "—")
        + '</td></tr>' for segment in segments)
    timeline_html = ('<h2>Kept and removed sections</h2><p>Times are relative to the beginning of the original video (hours:minutes:seconds.milliseconds). Cut sections remain in the original. Boundaries describe edit decisions and may differ from encoded clips by a frame.</p>'
                     '<div style="overflow-x:auto"><table><thead><tr><th>Action</th><th>Original start</th><th>Original end</th><th>Duration</th><th>Clip</th></tr></thead><tbody>'
                     + timeline_rows + '</tbody></table></div><p><a href="edit-timeline.csv">Download timeline CSV</a> · <a href="edit-timeline.json">Timeline JSON</a></p>')
    rows = "".join(f'<li><a href="Clips/clip-{i:03d}.mp4">Clip {i}</a><span>Original {a:.1f}–{b:.1f} seconds</span></li>' for i, (a, b) in enumerate(report["intervals"], 1))
    (out / "Review.html").write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Tennis practice review</title><style>body{font:17px system-ui;background:#101c24;color:#edf3f4;max-width:1000px;margin:40px auto;padding:0 24px}h1{font-size:36px}video{width:100%;border-radius:12px;background:#000}a{color:#b5ed83}p,span{color:#b8c9cf}ul{padding:0;list-style:none}li{display:flex;justify-content:space-between;padding:16px;border-bottom:1px solid #34454c}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:10px;border-bottom:1px solid #34454c;white-space:nowrap}.badge{color:#b5ed83;letter-spacing:.12em;font-size:12px}</style><div class="badge">TENNIS · PRACTICE REVIEW</div><h1>''' + html.escape(Path(report["source"]).stem) + f'''</h1><p>{report['kept_seconds']:.0f} seconds retained · {report['removed_seconds']:.0f} seconds removed · {report['clips']} clips</p><video controls preload="metadata" src="Practice.mp4"></video><p>Automatic practice cut. Includes preparation and recovery around detected movement. May retain non-playing activity or miss strokes; the original remains available.</p><ul>''' + rows + '</ul>' + timeline_html + '<p><a href="timestamps.csv">Original timestamps</a></p></html>')

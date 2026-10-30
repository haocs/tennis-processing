"""Complete edit decisions on the original video's elapsed-time axis."""
import csv
import json
import math


def timecode(seconds):
    millis = round(seconds * 1000)
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    seconds, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def build_timeline(duration, intervals):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Source duration must be positive and finite")
    segments = []

    def add(action, start, end, clip=None):
        segments.append({"action": action, "original_start_seconds": start,
                         "original_end_seconds": end, "duration_seconds": end - start,
                         "original_start": timecode(start), "original_end": timecode(end),
                         "clip": clip})

    cursor = 0.0
    for i, (start, end) in enumerate(intervals, 1):
        if not all(math.isfinite(v) for v in (start, end)) or not cursor <= start < end <= duration:
            raise ValueError("Kept intervals must be ordered, non-overlapping and inside the source")
        if start > cursor:
            add("cut", cursor, start)
        add("preserved", start, end, f"Clips/clip-{i:03d}.mp4")
        cursor = end
    if cursor < duration:
        add("cut", cursor, duration)
    return segments


def write_timeline(out, report):
    duration = report["source_metadata"]["duration"]
    segments = build_timeline(duration, report["intervals"])
    metadata = {"schema_version": 1, "source": report["source"],
                "source_sha256": report["source_sha256"], "source_duration_seconds": duration,
                "time_basis": "Elapsed time from the beginning of the original video, starting at zero.",
                "boundary_convention": "Start inclusive, end exclusive. Human-readable timestamps round to milliseconds; numeric seconds retain precision. Export boundaries may vary by a video frame.",
                "segments": segments}
    (out / "edit-timeline.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (out / "edit-timeline.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(segments[0]))
        writer.writeheader()
        writer.writerows(segments)
    report["edit_timeline"] = "edit-timeline.json"
    return segments

"""Measure chapter starts in the joined video's own timeline."""
import av
from pathlib import Path


def timing(path):
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        return {"duration_seconds": float(s.duration * s.time_base),
                "container_seconds": float(c.duration / av.time_base),
                "fps": float(s.average_rate or 30)}


def build_playback(out, intervals):
    out = Path(out)
    clips = []
    offset = 0.0
    for index, (start, end) in enumerate(intervals, 1):
        name = f"Clips/clip-{index:03d}.mp4"
        meta = timing(out / name)
        clips.append({"index": index, "file": name, "original_start_seconds": start,
                      "original_end_seconds": end, "duration_seconds": meta["duration_seconds"],
                      "expected_start_seconds": offset})
        offset += meta["container_seconds"]
    combined = timing(out / "Practice.mp4")
    # Each individually encoded clip begins on a keyframe. Compare expected
    # concat offsets to actual packet PTS; this includes AAC priming/rounding.
    keys = []
    with av.open(str(out / "Practice.mp4")) as c:
        stream = c.streams.video[0]
        for packet in c.demux(stream):
            if packet.is_keyframe and packet.pts is not None:
                keys.append(float(packet.pts * packet.time_base))
    if not keys:
        raise ValueError("Combined video has no chapter keyframes")
    tolerance = max(.1, 3 / combined["fps"])
    previous = -1
    for clip in clips:
        expected = clip.pop("expected_start_seconds")
        actual = min(keys, key=lambda value: abs(value - expected))
        if abs(actual - expected) > tolerance or actual <= previous:
            raise ValueError(f"Could not align chapter {clip['index']} to exported video")
        clip["playback_start_seconds"] = 0.0 if clip["index"] == 1 else actual
        previous = actual
    for i, clip in enumerate(clips):
        clip["playback_end_seconds"] = clips[i+1]["playback_start_seconds"] if i+1 < len(clips) else combined["duration_seconds"]
    return {"duration_seconds": combined["duration_seconds"], "alignment": "joined-video-keyframe-pts", "clips": clips}

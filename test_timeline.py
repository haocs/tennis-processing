import json
import csv
import pytest
from timeline import build_timeline, timecode, write_timeline


def test_timeline_covers_every_instant_once():
    rows = build_timeline(12.34567, [[2, 4], [4, 6], [9, 10]])
    assert [r["action"] for r in rows] == ["cut", "preserved", "preserved", "cut", "preserved", "cut"]
    assert rows[0]["original_start_seconds"] == 0
    assert rows[-1]["original_end_seconds"] == 12.34567
    assert all(a["original_end_seconds"] == b["original_start_seconds"] for a, b in zip(rows, rows[1:]))
    assert sum(r["duration_seconds"] for r in rows) == pytest.approx(12.34567)
    assert rows[2]["clip"] == "Clips/clip-002.mp4"


def test_all_cut_or_all_preserved():
    assert [r["action"] for r in build_timeline(10, [])] == ["cut"]
    assert [r["action"] for r in build_timeline(10, [[0, 10]])] == ["preserved"]
    assert timecode(3599.9996) == "01:00:00.000"


@pytest.mark.parametrize("duration,intervals", [(0, []), (float('nan'), []), (10, [[2, 5], [4, 6]]), (10, [[-1, 2]]), (10, [[2, 11]]), (10, [[2, 2]])])
def test_invalid_decisions_rejected(duration, intervals):
    with pytest.raises(ValueError):
        build_timeline(duration, intervals)


def test_json_csv_and_review_match_source_decisions(tmp_path):
    from pipeline import write_review
    report = {"source": "a<&.MOV", "source_sha256": "abc", "source_metadata": {"duration": 10}, "intervals": [[1, 5]], "kept_seconds": 4, "removed_seconds": 6, "clips": 1}
    write_review(tmp_path, report)
    data = json.loads((tmp_path / "edit-timeline.json").read_text())
    rows = list(csv.DictReader((tmp_path / "edit-timeline.csv").open()))
    assert len(rows) == len(data["segments"]) == 3
    assert rows[-1]["original_end"] == "00:00:10.000"
    assert data["segments"][1]["clip"] == "Clips/clip-001.mp4"
    page = (tmp_path / "Review.html").read_text()
    assert "00:00:05.000" in page and "a&lt;&amp;" in page
    assert 'edit-timeline.csv' in page and 'Cut' in page and 'Preserved' in page

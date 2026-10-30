import json
from timings import JobTiming, save_report
from monitor import progress


def test_monotonic_stage_totals_repeated_progress_and_wait():
    tick = [10.0]
    wall = [100.0]
    timing = JobTiming('example.mov', 75, 2, clock=lambda: tick[0], wall=lambda: wall[0])
    cfg = {'_timing': timing}
    tick[0] = 12
    progress(cfg, 'Analyzing', 1, 10)
    tick[0] = 14
    progress(cfg, 'Analyzing', 2, 10)
    tick[0] = 17
    progress(cfg, 'Exporting')
    tick[0] = 20
    wall[0] = 90  # System wall clock adjustment cannot distort elapsed durations.
    report = timing.report('done')
    assert report['total_seconds'] == 10
    assert report['observed_to_start_seconds'] == 25
    assert report['stages'] == [{'stage': 'Starting', 'seconds': 2}, {'stage': 'Analyzing', 'seconds': 5}, {'stage': 'Exporting', 'seconds': 3}]
    assert sum(s['seconds'] for s in report['stages']) == report['total_seconds']


def test_reports_persist_separate_attempts_and_failures(tmp_path):
    cfg = {'state_dir': str(tmp_path)}
    one = JobTiming('example.mov', 0, 1)
    two = JobTiming('example.mov', 0, 2)
    save_report(cfg, one, 'failed')
    save_report(cfg, two, 'done')
    reports = [json.loads(p.read_text()) for p in (tmp_path / 'timings').glob('*.json')]
    assert len(reports) == 2
    assert {r['status'] for r in reports} == {'done', 'failed'}
    assert {r['attempt'] for r in reports} == {1, 2}
    assert all(str(tmp_path) not in json.dumps(r) for r in reports)


def test_library_mirror_failure_keeps_local_report(tmp_path, monkeypatch):
    import storage_config
    def offline(cfg):
        raise RuntimeError('offline')
    monkeypatch.setattr(storage_config, 'check_library_storage', offline)
    timing = JobTiming('example.mov', 0, 1)
    timing.digest = 'a' * 64
    save_report({'state_dir': str(tmp_path), 'library_root': '/unavailable'}, timing, 'done')
    assert json.loads((tmp_path / 'timings' / (timing.id + '.json')).read_text())['status'] == 'done'


def test_success_and_failed_job_wrappers_record_and_do_not_leak(tmp_path, monkeypatch):
    import worker
    import pytest
    cfg = {'state_dir': str(tmp_path)}
    row = {'stable_since': 0, 'attempts': 0}
    def succeed(cfg, source, row, db):
        progress(cfg, 'Exporting')
    monkeypatch.setattr(worker, '_process_source', succeed)
    worker.process_source(cfg, tmp_path / 'example.mov', row, None)
    def fail(cfg, source, row, db):
        progress(cfg, 'Copying original from NAS')
        raise OSError('unavailable')
    monkeypatch.setattr(worker, '_process_source', fail)
    with pytest.raises(OSError):
        worker.process_source(cfg, tmp_path / 'example.mov', row, None)
    reports = [json.loads(p.read_text()) for p in (tmp_path / 'timings').glob('*.json')]
    assert {r['status'] for r in reports} == {'done', 'failed'}
    assert '_timing' not in cfg


def test_csv_contains_stage_times(tmp_path, capsys):
    from timings import print_summary
    timing = JobTiming('example.mov', 0, 1)
    save_report({'state_dir': str(tmp_path)}, timing, 'done')
    print_summary({'state_dir': str(tmp_path)})
    result = capsys.readouterr().out
    assert 'stage_seconds' in result
    assert 'example.mov,1,done,' in result


def test_successful_library_mirror_has_no_source_paths(tmp_path, monkeypatch):
    import storage_config
    library = tmp_path / 'library'
    library.mkdir()
    monkeypatch.setattr(storage_config, 'check_library_storage', lambda cfg: library)
    timing = JobTiming('example.mov', 0, 1)
    timing.digest = 'b' * 64
    save_report({'state_dir': str(tmp_path / 'state'), 'library_root': str(library)}, timing, 'done')
    mirrored = library / 'metadata/processing' / timing.digest / (timing.id + '.json')
    local = tmp_path / 'state/timings' / (timing.id + '.json')
    assert mirrored.read_bytes() == local.read_bytes()
    assert str(tmp_path) not in mirrored.read_text()

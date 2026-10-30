#!/usr/bin/env python3
"""Command-line controls for the macOS tennis background service."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import plistlib
import subprocess
import time

ROOT = Path(__file__).resolve().parent
LABEL = 'local.tennis-autocut'


def loaded(domain):
    return subprocess.run(['launchctl', 'print', f'{domain}/{LABEL}'], capture_output=True, text=True)


def mark_stopped(cfg, python):
    from monitor import write_snapshot
    folder = Path(cfg.get('status_dir', cfg['state_dir']))
    path = folder / 'Processing-status.json'
    data = json.loads(path.read_text()) if path.exists() else {'files': []}
    data.update(service='Stopped', stage='Stopped', current_video=None, progress=None,
                heartbeat=time.time(), progress_at=time.time())
    for row in data['files']:
        if row['status'] == 'processing':
            row['status'] = 'interrupted'
            row['error'] = 'Stopped by command. The job will retry when the service starts.'
    write_snapshot(folder, data)
    # A network filesystem stall must not make the stop command hang indefinitely.
    code = 'import json,sys; from pathlib import Path; from worker import check_storage; from monitor import write_snapshot,public_snapshot; from storage_config import status_destination; c=json.loads(Path(sys.argv[1]).read_text()); d=json.loads(Path(sys.argv[2]).read_text()); write_snapshot(status_destination(c,check_storage),public_snapshot(d) if c.get("library_root") else d)'
    try:
        subprocess.run([str(python), '-c', code, str(ROOT / 'config.json'), str(path)],
                       cwd=ROOT, timeout=10, check=True, capture_output=True)
    except (subprocess.SubprocessError, OSError):
        print('Local status says stopped; NAS status could not be updated and may be stale.')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['start', 'run', 'install', 'stop', 'restart', 'status', 'logs', 'retry', 'timings'])
    p.add_argument('--force', action='store_true', help='Allow restart to interrupt an active job; it retries from the beginning')
    p.add_argument('--follow', '-f', action='store_true', help='Follow logs until Ctrl-C')
    args = p.parse_args(argv)
    domain = f'gui/{os.getuid()}'
    plist = Path.home() / 'Library/LaunchAgents' / f'{LABEL}.plist'
    python = ROOT / '.runtime/venv/bin/python'
    state = ROOT / '.state'
    state.mkdir(exist_ok=True)
    cfg = json.loads((ROOT / 'config.json').read_text())
    action = 'start' if args.action in {'run', 'install'} else args.action
    if action == 'timings':
        from timings import print_summary
        print_summary(cfg)
        return 0
    if action == 'status':
        result = loaded(domain)
        label = 'running' if result.returncode == 0 and 'state = running' in result.stdout else 'registered; waiting/restarting' if result.returncode == 0 else 'stopped'
        print('Automatic processing:', label, flush=True)
        return subprocess.run([str(python), str(ROOT / 'worker.py'), 'status']).returncode
    if action == 'logs':
        log = state / 'worker.log'
        log.touch(exist_ok=True)
        try:
            return subprocess.run(['tail', '-n', '80'] + (['-F'] if args.follow else []) + [str(log)]).returncode
        except KeyboardInterrupt:
            return 0
    if action == 'retry':
        subprocess.run([str(python), str(ROOT / 'worker.py'), 'retry'], check=True)
        print('Failed videos will retry on the next scan while the service is running.')
        return 0
    with (state / 'control.lock').open('a') as control:
        fcntl.flock(control, fcntl.LOCK_EX)
        existing = loaded(domain)
        if action == 'start' and existing.returncode == 0:
            print('Automatic processing is already registered. Use status or restart.')
            return 0
        if action != 'stop' and not python.exists():
            raise SystemExit('Run setup_runtime.py first')
        with (state / 'worker.lock').open('a') as job_lock:
            if action == 'restart' and not args.force:
                try:
                    fcntl.flock(job_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    print('A job or scan is active. Retry when idle, or use restart --force to interrupt and retry it.')
                    return 2
            if existing.returncode == 0:
                subprocess.run(['launchctl', 'bootout', f'{domain}/{LABEL}'], check=True)
            if action == 'stop':
                plist.unlink(missing_ok=True)
                mark_stopped(cfg, python)
                print('Automatic processing stopped and login registration removed. Videos retained.')
                return 0
            content = {'Label': LABEL, 'ProgramArguments': ['/usr/bin/caffeinate', '-i', str(python), str(ROOT / 'worker.py'), 'watch', '--config', str(ROOT / 'config.json')], 'WorkingDirectory': str(ROOT), 'RunAtLoad': True, 'KeepAlive': True, 'ThrottleInterval': 30, 'ProcessType': 'Background', 'Nice': 10, 'StandardOutPath': str(state / 'service.log'), 'StandardErrorPath': str(state / 'service.log')}
            plist.parent.mkdir(parents=True, exist_ok=True)
            plist.write_bytes(plistlib.dumps(content))
            subprocess.run(['launchctl', 'bootstrap', domain, str(plist)], check=True)
        print(f'Automatic processing enabled for {cfg.get("input_root", cfg.get("nas_root"))}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

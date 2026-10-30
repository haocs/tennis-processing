#!/usr/bin/env python3
"""Initialize and migrate a separate published library; never changes originals."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import uuid

from publication import publish_library, validate_library
from worker import APP, atomic_json, database


def initialize(root, mount=None):
    root = Path(root).resolve()
    if mount and not os.path.ismount(mount):
        raise ValueError('Destination volume is not mounted')
    if str(root).startswith('/Volumes/') and not mount:
        raise ValueError('Supply --mount-point for a mounted NAS destination')
    root.mkdir(parents=True, exist_ok=True)
    marker = root / '.tennis-library-id'
    if not marker.exists():
        unrelated = [p for p in root.iterdir() if p.name != '.DS_Store']
        if unrelated:
            raise ValueError('Use an empty destination folder when initializing a library')
        marker.write_text(str(uuid.uuid4())+'\n')
    (root / 'media').mkdir(exist_ok=True)
    (root / 'metadata/sessions').mkdir(parents=True, exist_ok=True)
    return marker.read_text().strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(APP/'config.json'))
    sub = parser.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init');init.add_argument('destination');init.add_argument('--mount-point')
    migrate = sub.add_parser('migrate');migrate.add_argument('session');migrate.add_argument('destination')
    configure = sub.add_parser('configure');configure.add_argument('destination');configure.add_argument('--mount-point');configure.add_argument('--input-root');configure.add_argument('--mount-source')
    verify = sub.add_parser('verify');verify.add_argument('destination')
    args = parser.parse_args()
    if args.command == 'init':
        identity=initialize(args.destination,args.mount_point)
        print(json.dumps({'library_root':str(Path(args.destination).resolve()),'library_storage_id':identity,'library_mount_point':args.mount_point},indent=2))
        return
    cfg_path=Path(args.config)
    cfg=json.loads(cfg_path.read_text())
    root=Path(args.destination).resolve()
    if not (root/'.tennis-library-id').is_file():
        raise SystemExit('Initialize the destination with libraryctl init first')
    if args.command == 'verify':
        for p in (root/'metadata/sessions').glob('*.json'):
            if not validate_library(root/'media'/p.stem,p.stem):
                raise SystemExit(f'Publication failed verification: {p.stem}')
        print('All committed library sessions passed checksum verification')
        return
    state=Path(cfg['state_dir'])
    with (state/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.command == 'configure':
            from storage_config import check_library_storage
            updated=dict(cfg,library_root=str(root),library_storage_id=(root/'.tennis-library-id').read_text().strip(),library_mount_point=args.mount_point,library_mount_source=args.mount_source)
            if args.input_root:
                updated['input_root']=str(Path(args.input_root).resolve())
            check_library_storage(updated)
            # Prevent completed jobs from silently retaining references to private outputs.
            with database(state/'jobs.sqlite3') as db:
                for row in db.execute('SELECT digest FROM jobs'):
                    if not (root/'metadata/sessions'/f'{row[0]}.json').is_file():
                        raise SystemExit('Migrate every completed session before configuring the shared library')
            backup=cfg_path.with_name(cfg_path.name+f'.before-library-{int(time.time())}')
            backup.write_bytes(cfg_path.read_bytes())
            atomic_json(cfg_path,updated)
            print('Separate shared output configured. Restart the processor to load it.')
        else:
            original=Path(args.session).resolve()
            report=json.loads((original/'session.json').read_text())
            output=publish_library(original,root,digest=report['source_sha256'])
            with database(state/'jobs.sqlite3') as db:
                db.execute('UPDATE files SET output=? WHERE digest=?',(str(output),report['source_sha256']))
                db.execute('UPDATE jobs SET output=? WHERE digest=?',(str(output),report['source_sha256']))
            print(f'Published and verified {report["source_sha256"]}; originals and previous output retained')


if __name__=='__main__':
    main()

"""Mac supervisor for the portable API worker; no scheduling lives here."""
import argparse
import json
import os
from pathlib import Path
import plistlib
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
LABEL='local.tennis-api-worker'

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['start','stop','restart','status','logs','run']);p.add_argument('--config',default=str(ROOT/'api-worker.json'));p.add_argument('--force',action='store_true');args=p.parse_args()
    cfg_path=Path(args.config).resolve();cfg=json.loads(cfg_path.read_text());state=Path(cfg['state_dir']);state.mkdir(parents=True,exist_ok=True)
    python=ROOT/'.runtime/venv/bin/python'
    if args.action=='run': return subprocess.call([str(python),str(ROOT/'api_worker.py'),'--config',str(cfg_path)])
    if args.action=='logs': return subprocess.call(['tail','-n','80',str(state/'service.log')])
    if sys.platform!='darwin': raise SystemExit('Use the portable api_worker.py entry point; Windows service installation is a follow-up.')
    target=f'gui/{os.getuid()}/{LABEL}';domain=f'gui/{os.getuid()}'
    def loaded():return subprocess.run(['launchctl','print',target],capture_output=True,text=True)
    if args.action=='status':
        result=loaded();print('Worker registered:' ,result.returncode==0)
        snapshot=state/'worker.json'
        if snapshot.exists():
            data=json.loads(snapshot.read_text());print('Worker:',data.get('worker_id'));print('Stage:',data.get('stage','Idle'));print('Task:',(data.get('active') or {}).get('task',{}).get('id'))
        return 0
    with (state/'supervisor.lock').open('a') as control:
        import fcntl
        fcntl.flock(control,fcntl.LOCK_EX)
        current=loaded()
        if current.returncode==0 and args.action=='start':print('Worker already registered');return 0
        if current.returncode==0:
            snapshot=state/'worker.json'
            if not args.force and snapshot.exists() and json.loads(snapshot.read_text()).get('active'):
                raise SystemExit('Worker has an active attempt. Wait until idle, or use --force; its lease/checkpoints permit recovery.')
            subprocess.run(['launchctl','bootout',target],check=True)
            for _ in range(30):
                if loaded().returncode:break
                time.sleep(.2)
        plist=Path.home()/'Library/LaunchAgents'/f'{LABEL}.plist'
        if args.action=='stop':
            plist.unlink(missing_ok=True);print('Worker stopped; task history and local checkpoints retained');return 0
        value={'Label':LABEL,'ProgramArguments':['/usr/bin/caffeinate','-i',str(python),str(ROOT/'api_worker.py'),'--config',str(cfg_path)],'WorkingDirectory':str(ROOT),'RunAtLoad':True,'KeepAlive':True,'ThrottleInterval':30,'ProcessType':'Background','Nice':10,'StandardOutPath':str(state/'service.log'),'StandardErrorPath':str(state/'service.log')}
        plist.parent.mkdir(parents=True,exist_ok=True);plist.write_bytes(plistlib.dumps(value))
        subprocess.run(['launchctl','bootstrap',domain,str(plist)],check=True)
        print('API worker enabled; server controls task assignment')
    return 0

if __name__=='__main__':raise SystemExit(main())

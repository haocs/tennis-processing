#!/usr/bin/env python3
"""Install from GitHub with standalone Python; no IDE is required."""
import argparse,json,os,shlex,shutil,subprocess,sys,venv
from pathlib import Path
from setup_runtime import REVISION
ROOT=Path(__file__).resolve().parent
SUPPORT=Path.home()/'Library/Application Support/Tennis Worker'

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--no-start',action='store_true');args=p.parse_args()
    if sys.platform!='darwin' or sys.version_info[:2]!=(3,12) or 'codex' in sys.base_prefix.lower():
        raise SystemExit('Use standalone macOS Python 3.12, e.g. Homebrew python@3.12.')
    SUPPORT.mkdir(parents=True,exist_ok=True);source=SUPPORT/'source'
    if ROOT!=source:
        if source.exists(): subprocess.run(['git','-C',str(source),'pull','--ff-only'],check=True)
        else: subprocess.run(['git','clone','https://github.com/haocs/tennis-processing.git',str(source)],check=True)
        os.execv(sys.executable,[sys.executable,str(source/'install_worker.py'),'--config',str(args.config.resolve())]+(['--no-start'] if args.no_start else []))
    env=SUPPORT/'venv'
    if not (env/'bin/python').exists(): venv.EnvBuilder(with_pip=True).create(env)
    python=env/'bin/python'
    subprocess.run([str(python),'-m','pip','install','-r',str(ROOT/'requirements.lock')],check=True)
    cfg=json.loads(args.config.read_text());state=Path(cfg['state_dir']);snapshot=state/'worker.json'
    if snapshot.exists() and json.loads(snapshot.read_text()).get('active'): raise SystemExit('Wait for the active task to finish before installing.')
    dependency=SUPPORT/'dependencies/RallyClip'
    if not dependency.exists():
        dependency.parent.mkdir(exist_ok=True)
        subprocess.run(['git','clone','--no-checkout','https://github.com/iroblesrazzaq/RallyClip.git',str(dependency)],check=True)
    subprocess.run(['git','-C',str(dependency),'fetch','--depth','1','origin',REVISION],check=True)
    subprocess.run(['git','-C',str(dependency),'checkout','--detach',REVISION],check=True)
    cache=Path(cfg['processor']['rallyclip_path'])/'models'
    if cache.is_dir() and not (dependency/'models').exists(): shutil.copytree(cache,dependency/'models')
    subprocess.run([str(python),str(dependency/'scripts/fetch_artifact.py')],check=True)
    deployed=SUPPORT/'state'
    if state.resolve()!=deployed.resolve() and not deployed.exists(): shutil.copytree(state,deployed,ignore=shutil.ignore_patterns('agent.lock','supervisor.lock','service.log'))
    deployed.mkdir(exist_ok=True);cfg['state_dir']=str(deployed);cfg['processor']['rallyclip_path']=str(dependency)
    private=SUPPORT/'worker.json';private.write_text(json.dumps(cfg,indent=2)+'\n');private.chmod(0o600)
    bin_dir=SUPPORT/'bin';bin_dir.mkdir(exist_ok=True);cli=bin_dir/'workerctl'
    cli.write_text('#!/bin/sh\nexec '+shlex.quote(str(python))+' '+shlex.quote(str(ROOT/'api_service.py'))+' "$@" --config '+shlex.quote(str(private))+'\n');cli.chmod(0o755)
    if not args.no_start: subprocess.run([str(cli),'restart'],check=True)
    print('Installed. Controls:',cli)

if __name__=='__main__': main()

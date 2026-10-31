"""Package a named Mac worker; development stays in the source checkout."""
from pathlib import Path
import json
import os
import platform
import plistlib
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parent

def copy_file(source,target):
    # Local hardlinks avoid duplicating the installed video runtime on disk.
    try:os.link(source,target)
    except OSError:shutil.copy2(source,target)
    return target


def build(config):
    config=Path(config).resolve();cfg=json.loads(config.read_text())
    support=Path.home()/'Library/Application Support/Tennis Worker'
    support.mkdir(parents=True,exist_ok=True)
    code=support/'code';code.mkdir(exist_ok=True)
    for path in ROOT.glob('*.py'):
        target=code/path.name
        temporary=target.with_suffix('.new');shutil.copy2(path,temporary);os.replace(temporary,target)
    python=support/'venv/bin/python'
    runtime=json.loads(subprocess.check_output([str(python),'-c',
        'import sys,sysconfig,json; print(json.dumps({"base":sys.base_prefix,"packages":sysconfig.get_path("purelib")}))'],text=True))
    if 'codex' in runtime['base'].lower():
        raise SystemExit('Install a standalone Python runtime first')
    packages=Path(runtime['packages'])
    model=Path(cfg['processor']['rallyclip_path']).resolve()
    state=support/'state'
    if not state.exists():
        shutil.copytree(cfg['state_dir'],state,copy_function=shutil.copy2,ignore=shutil.ignore_patterns('service.log','supervisor.lock','agent.lock'))
    cfg['state_dir']=str(state);cfg['processor']['rallyclip_path']=str(model)
    private=support/'worker.json';private.write_text(json.dumps(cfg,indent=2));private.chmod(0o600)
    # Keep the operator CLI pointing at the deployed state, without moving source.
    # User configuration is copied privately; the source checkout stays generic.
    app=Path.home()/'Applications/Tennis Worker.app';contents=app/'Contents'
    (contents/'MacOS').mkdir(parents=True,exist_ok=True);(contents/'Resources').mkdir(exist_ok=True)
    info={'CFBundleIdentifier':'io.haocs.tennis-worker','CFBundleName':'Tennis Worker','CFBundleDisplayName':'Tennis Worker','CFBundleExecutable':'TennisWorker','CFBundlePackageType':'APPL','CFBundleVersion':'2','CFBundleShortVersionString':'2.0.0','LSMinimumSystemVersion':'14.0','CFBundleSupportedPlatforms':['MacOSX'],'LSUIElement':True,'NSPrincipalClass':'NSApplication','NSLocalNetworkUsageDescription':'Connect to your NAS processing server to receive tennis tasks and report progress. Your recordings stay on your local network.'}
    (contents/'Info.plist').write_bytes(plistlib.dumps(info))
    launch={'python':str(python),'python_home':runtime['base'],'site_packages':str(packages),'script':str(code/'api_worker.py'),'config':str(private),'directory':str(support)}
    (contents/'Resources/worker-launch.json').write_text(json.dumps(launch))
    binary=contents/'MacOS/TennisWorker';temporary=binary.with_suffix('.new')
    # Python/config updates do not require recompiling an unchanged native wrapper.
    swift=ROOT/'mac_worker.swift'
    if not binary.exists() or binary.stat().st_mtime < swift.stat().st_mtime:
        sdk=subprocess.check_output(['xcrun','--sdk','macosx','--show-sdk-path'],text=True).strip()
        subprocess.run(['xcrun','swiftc','-sdk',sdk,'-target',platform.machine()+'-apple-macos14.0',str(swift),'-framework','AppKit','-o',str(temporary)],check=True)
        os.replace(temporary,binary)
    subprocess.run(['codesign','--force','--sign','-','--identifier','io.haocs.tennis-worker',str(app)],check=True)
    return binary

if __name__=='__main__':
    if sys.platform!='darwin':raise SystemExit('macOS only')
    print(build(sys.argv[1] if len(sys.argv)>1 else ROOT/'api-worker.json'))

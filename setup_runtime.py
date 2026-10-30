#!/usr/bin/env python3
"""One-time local installation; downloads code/models, never uploads footage."""
import json
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent
REVISION = "8d8dd7a534a5431bb192e18a94eac787a615118f"


def main():
    if sys.version_info < (3, 12):
        raise SystemExit("Python 3.12 or newer is required for this pinned environment.")
    runtime = ROOT / ".runtime"
    runtime.mkdir(exist_ok=True)
    env = runtime / "venv"
    if not (env / "bin/python").exists():
        venv.EnvBuilder(with_pip=True).create(env)
    python = env / "bin/python"
    subprocess.run([str(python), "-m", "pip", "install", "-r", str(ROOT / "requirements.lock")], check=True)
    repo = runtime / "RallyClip"
    if not repo.exists():
        subprocess.run(["git", "init", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "remote", "add", "origin", "https://github.com/iroblesrazzaq/RallyClip.git"], check=True)
        subprocess.run(["git", "-C", str(repo), "fetch", "--depth", "1", "origin", REVISION], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", "--detach", REVISION], check=True)
    actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if actual != REVISION:
        raise SystemExit("Dependency checkout differs from tested revision; not changing it automatically.")
    subprocess.run([str(python), str(repo / "scripts/fetch_artifact.py")], check=True)
    config = ROOT / "config.json"
    if not config.exists():
        example = ROOT / "config.example.json"
        if not example.exists():
            raise SystemExit("Create config.json from config.example.json first")
        config.write_bytes(example.read_bytes())
    cfg = json.loads(config.read_text())
    cfg["rallyclip_path"] = str(repo)
    cfg["state_dir"] = str(ROOT / ".state")
    cfg["status_dir"] = str(ROOT)
    config.write_text(json.dumps(cfg, indent=2) + "\n")
    print("Runtime installed. Configure the NAS path, then run service.py install.")


if __name__ == "__main__":
    main()

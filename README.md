# Tennis processing pipeline

A local Mac processor that reads private tennis recordings and publishes a separate shared viewing library. Originals are retained. The website is a separate application and never needs the private input mount.

## Platform and dependencies

The current validated processor targets Apple Silicon macOS with Python 3.12+. HDR exports use Apple's VideoToolbox HEVC Main10 encoder. CPU pose inference is available as a fallback. This release does not claim Linux support for HDR export; the separate website container is portable to Linux/Synology.

`setup_runtime.py` creates an isolated runtime, installs the pinned packages, fetches the pinned RallyClip source revision, and verifies model downloads with upstream checksums. No upstream source or model weights are vendored here. See [external dependency review](DEPENDENCIES.txt) before redistributing any upstream code, model, or bundled runtime. This public source repository includes none of those artifacts.

## Configure

Copy `config.example.json` to `config.json` and set paths appropriate to your machine:

- `input_root`: private folder containing original recordings.
- `mount_point`: actual mounted input volume; prevents accidental writes/reads against a disconnected mount.
- `storage_id`: optional private-storage identity marker.
- `library_root`: separate, non-nested shared output folder.
- `library_mount_point`: mounted shared volume.
- `library_mount_source`: optional password-free SMB source, such as `//nas-user@nas.local/Apps`, for Mac reconnection using saved credentials.
- `library_storage_id`: identity printed by library initialization.
- `state_dir`, `status_dir`, `rallyclip_path`: setup assigns installation-local paths.

Do not place passwords in config. Keep config, runtime, job state, private media, and status files out of Git. The legacy `nas_root` setting is accepted as an input-root alias for existing installations.

## Install and initialize

```sh
python3 setup_runtime.py
.runtime/venv/bin/python libraryctl.py init /path/to/shared/library --mount-point /path/to/mounted/share
```

Paste the printed library settings into your private config. For a plain local filesystem destination, omit `--mount-point`. Original and shared library roots must be separate and must not contain one another.

## Service commands

```sh
./tennisctl start
./tennisctl stop
./tennisctl restart
./tennisctl status
./tennisctl logs --follow
./tennisctl retry
```

The macOS login service starts after login, is restarted by launchd if it exits, and keeps the Mac from idle system sleep while enabled. Stop disables login registration; a later start enables it again. Restart refuses while a job/scan owns the worker lock unless `--force` is specified. Forced interruption retries a job from the beginning. Use one installed processor instance; do not install a second copy over a running installation without stopping the first.

Files must remain unchanged across scans for three minutes. Jobs run one at a time. Persistent queue state, backoff, content deduplication, mount identities, local scratch copies, decode checks, SHA-256 verification, and commit-last publication protect against partial uploads and interrupted publication. If the shared destination is unavailable, it does not silently create a replacement local folder or publish into private storage.

## Output and status

See [session contract](SESSION-CONTRACT.md). The shared library separates `media/` from `metadata/`; the website owns its own state volume. Metadata includes recording timestamp/GPS (including optional altitude in metres) when present, original cut/preserved intervals, measured clip durations, stable identities, algorithm version, and chapters aligned against actual combined-video keyframe timestamps.

Heartbeat files update locally every 15 seconds and on shared storage roughly every 30 seconds. A heartbeat older than two minutes is stale. A live heartbeat does not prove an operation is progressing; long-operation progress time is shown separately. Shared status omits private input/output paths and raw error details. Detailed errors remain in local processor logs.

HDR exports preserve 10-bit HLG/PQ base-layer color and BT.2020 tags, with verification. They remain lossy re-encodes and do not retain Dolby Vision dynamic metadata. SDR exports use H.264. Both preserve the primary audio track. Temporary analysis proxies are separate from final exports.

## Migrate an existing installation

Stop the old service before the migration window. Initialize the shared library, then publish each existing session without rerunning detection:

```sh
.runtime/venv/bin/python libraryctl.py migrate /path/to/existing/Processed/session /path/to/shared/library
.runtime/venv/bin/python libraryctl.py configure /path/to/shared/library --mount-point /path/to/mount --mount-source //nas-user@nas.local/Apps
./tennisctl start
```

Migrate all completed sessions before `configure`; the command refuses missing publications. Migration updates queue output references while retaining original files and previous outputs. For legacy sessions lacking embedded GPS/chapter metadata, recording fields already in `session.json` are retained; chapters can be measured from existing exports. New recordings receive the enriched metadata automatically. Keep previous outputs until the shared library and partner access are verified.

## Tests and limitations

```sh
.runtime/venv/bin/python -m pytest -q
```

The current activity detector estimates near-player arm motion. It can retain downtime/gestures and miss slow or occluded strokes; preserve originals. Feedback-based detector improvement is a separate future project. Unsupported HDR tags or failed export validation raise an error rather than silently publishing an empty/broken session.

The companion viewing application is [tennis-web](https://github.com/haocs/tennis-web). Cut-quality feedback and algorithm experiments are planned separately.

## Permanent processing timings

Each processing attempt writes `.state/timings/<attempt-id>.json`, including UTC
start/finish, monotonic wall-clock total, per-stage durations, outcome, source
identity and available source duration/resolution/frame rate. Retry attempts are
retained separately. Total includes copying, hashing, preparation, detection,
export, verification, NAS publication and scratch cleanup. It excludes writing
the timing report itself. Repeated progress updates do not reset a stage timer.

Observed-to-start time is separate: it includes upload-settling checks, queue wait
and retry backoff. It is measured since the latest observed source signature,
not from the iPhone recording or upload start. A forced kill/power loss can prevent
the current attempt's final report; older completed attempts remain intact.

Once a digest is known, reports are also mirrored best-effort to
`<library>/metadata/processing/<source-sha256>/<attempt-id>.json`, outside the
checksummed media. A mirror failure retains the Mac report and does not fail an
otherwise successful video job. Reports contain filenames, not original paths,
GPS coordinates or raw exception messages. Legacy non-library deployments retain
reports locally. No historical timings are invented for older recordings.

Export a CSV for analysis with:

```sh
./tennisctl timings > timings.csv
```

The CSV has one row per stage; the total is repeated for context, so do not sum
the total column across stages. These measurements cover sequential processing;
parallelism has not been enabled.

## API worker v2

`api_worker.py` is the portable lease-based executor for
[tennis-processing-server](https://github.com/haocs/tennis-processing-server).
The old `worker.py` folder-watcher is retained for migration/rollback; never run
both automatic schedulers over the same recordings.

Copy `api-worker.example.json` to private `api-worker.json`, configure mounted
storage mappings, server registration token and the existing model runtime. API
worker storage includes `originals` (read only) and `processing` (write staging),
not a library publication root. The server alone activates a result. Runtime
state/checkpoints live in a separate local state directory. A server-owned mirror
of its small heartbeat snapshot lives on NAS; credentials never enter that mirror.

Mac commands:

```sh
./workerctl run           # foreground; Ctrl-C stops the local process
./workerctl start         # login service, keeps the Mac awake while running
./workerctl status
./workerctl logs
./workerctl restart       # refuses active work unless --force is specified
./workerctl stop
```

A Windows machine can use the Python entry point and storage mappings, but Windows
service installation and HDR encoder support are not yet tested/implemented.
This release remains Mac-first; no Windows-ready claim is made.

Execution stages are acquire_source, verify_source, prepare_proxy, analyze,
**process**, package_result and upload_result. The process stage renders clips,
combines and verifies them. Completed stages retain validated local checkpoints;
algorithm/config/source changes invalidate their fingerprint. Process currently
resumes at whole-stage boundaries, not individual clips (follow-up).

Optional hybrid AI filtering (`processing_strategy: "hybrid_ai_vision"`) can be
enabled per task when the worker configures `"ai": {"enabled": true, ...}` in
`api-worker.json`. The worker advertises discovered models from configured local
providers (`antigravity`, `gemini`, `codex`) and honors per-task `ai_model`,
`allow_model_fallback`, and `ai_tuning` overrides, emitting `ai-provenance.json`
without uploading raw frames or prompts. Worker 2.1.0 uses the noninteractive Antigravity print interface and ChatGPT-signed-in Codex with explicit stdin/image inputs; provider schemas are validated before applying decisions. AI remains opt-in per task; existing watchers keep vision-only behavior unless explicitly changed. Raw pose outputs cache locally under
`<state_dir>/cache/pose` while per-task `pose-observations.json` and AI decisions
recompute cleanly when tuning parameters change.

Dryrun creates real local media and sends findings/timing through the API; it
uploads no result/media/checkpoint files to NAS. Experiment tasks isolate the same
executor with parameter changes; training/evaluation executors are future work.
**P2:** local-media cleanup/retention task, protecting active work. No automatic
scratch deletion is enabled in the API worker yet.

Original container/stream tags, GPS/altitude, creation time, frame/codec/color
properties and codec extra data are retained in `original-metadata.json` beside
normal/experiment results, and locally for dryruns. Source SHA-256 and the untouched
original remain authoritative. This snapshot is not a lossless substitute for all
vendor binary metadata. Original Dolby Vision dynamic metadata is not reapplied
to newly encoded HDR clips; original timing/duration tags are not misrepresented
as edited clip durations.

Task events use a durable local outbox and monotonic sequences. The server lease
is authoritative; cached assignments cannot grant ownership. Heartbeats renew the
lease separately from progress. FFmpeg is stopped if ownership/cancellation fails;
NAS uploads check ownership between chunks. Completion is persisted locally and
replayed with an idempotency key after a lost response.

## Standalone Mac installation

Use Homebrew Python 3.12 and Apple's command-line developer tools (the native
launcher is compiled locally):

```sh
brew install python@3.12
git clone https://github.com/haocs/tennis-processing.git
cd tennis-processing
/opt/homebrew/bin/python3.12 install_worker.py --config /path/to/private-worker.json
```

The installer pulls its managed source checkout from GitHub into
`~/Library/Application Support/Tennis Worker/source`, installs an independent
Python environment and pinned model dependency, and registers
`~/Applications/Tennis Worker.app` to run at login. No IDE, Codex runtime or
Documents-folder access is required. The initial checkout can be removed.
Private settings, identity, checkpoints and logs stay under Application Support.

Use `"$HOME/Library/Application Support/Tennis Worker/bin/workerctl"` followed by
`status`, `logs`, `start`, `stop`, or `restart`. To update, pull the managed source
checkout with `git pull --ff-only`, then rerun its installer using the private
`worker.json`. Installation refuses to interrupt an active task.

Storage mappings optionally accept `smb_url`, e.g. `//user@nas/share/path`, plus
an identity marker. macOS reconnects the subfolder with existing saved SMB
credentials. Never put passwords in URLs; mount in Finder first if credentials
are unavailable. No local state, models or dryrun media are published to GitHub.

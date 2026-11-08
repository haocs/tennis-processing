# Published session contract v1

A session is visible only after `metadata/sessions/<id>.json` is atomically committed. Its media and supporting metadata must already be complete. Ignore hidden staging/orphan folders. A corrupt existing publication is retained for inspection rather than silently overwritten.

- `schema_version`: 1.
- `id`: full source SHA-256 hex, stable across storage moves and re-encoding.
- `title`: initial filename stem; website overrides are separate.
- `source`: basename, content hash, original duration, recording timestamp with timezone when available, and optional latitude/longitude. No private directory paths or credentials.
- `media_directory`: safe, single directory name under the configured media root. Canonical publications use the session ID; explicit legacy previews may use an existing processed-session directory name.
- `video`: Practice.mp4 filename, poster name, measured joined-video duration, dimensions, and HDR flag.
- `clips`: chronological clip entries. Each has a stable boundary-derived ID, original clip number, safe relative MP4 filename, original start/end, measured encoded duration, and actual joined-video chapter start/end.
- `chapter_alignment`: `joined-video-keyframe-pts` for the current publisher. Chapter starts are derived from keyframe PTS close to concat offsets measured using exported container durations. Audio priming and rounding are included. The first chapter starts at zero; the last ends at measured joined-video duration.
- `algorithm`: detector version; `encoding`: export profile and limitations.
- `media_checksums` and `metadata_checksums`: SHA-256 values used by the publisher/processor. The website reads committed metadata; it does not rehash multi-gigabyte videos on each page request.

Original intervals are start-inclusive/end-exclusive. A clip's encoded video duration can differ by a frame from its requested original interval. Its chapter span can also include container/audio rounding; do not interchange these three durations.

Clip IDs are the first 24 hex characters of SHA-256 of:

```
<source-sha256>:<original-start-seconds formatted to 6 decimals>:<original-end-seconds formatted to 6 decimals>
```

A boundary change creates a new clip ID. Website state keeps previous IDs for future reconciliation; it does not infer that a differently cut clip is identical. Shared preference keys use the session ID and clip ID. A changed location/title does not alter media identity.

Media and processing metadata are read-only from the website. Likes, overrides, and future feedback belong in the separate website database. Store source and algorithm identities alongside future feedback so it can be traced back to the exact cut.

Supported transport today is a filesystem mounted outside the application, optionally expressed as a local `file://` URL. Storage adapters can add other transports later without coupling the UI to Synology. Raw SMB URLs, HTTP catalogs, and object-storage providers are not currently implemented.

Original iPhone location metadata retains latitude, longitude, and optional
`altitude_m` when present. The website owns saved court names and matching rules;
those do not rename original recordings or rewrite video media.

## Worker heuristic diagnostics (`heuristic-diagnostics.json`)

New processing outputs emit `heuristic-diagnostics.json` alongside session metadata
(listed in `artifacts` and verified in `metadata_checksums` / `output_checksums`, capped at
`<= 512 KiB` and `<= 5` ranked hints per clip or removed gap). Existing publications
without `heuristic-diagnostics.json` remain valid.

Dense frame-by-frame pose observations (`pose-observations.json`) remain in local worker
scratch/cache only and are never published to NAS.

Key invariants and fields:
- `schema_version`: `1`.
- `source_sha256`: 64-hex original content hash; `source_duration_ms` and `source_duration_seconds`.
- `task_id`, `attempt_id`, `video_version_id` (`v_<32-hex>` compact JSON hash over frozen publication identity), `published_at` (bound only after `public_session` generation is frozen).
- `extraction_cache_key`: 64-hex SHA-256 over the complete upstream extraction tuple (`source_sha256`, `proxy_spec` with actual `target_fps: 8.0`, dimensions, rotation, transfer, and `pts_timebase`, `detector_spec`, `extractor_code_digest`, and `roi_and_filter_config`).
- `truth_status`: `"heuristic_only_not_play_or_swing_truth"` and `training_label_status`: `"not_released"`. `generator` explicitly sets `benchmark_truth: false`, `verified_play_or_swing: false`, `verified_player_identity: false`.
- `provenance`: full model, 8 fps proxy, timebase, software/module digest, and ROI/target-selection configuration. Task stage checkpoints are task-local recovery only, not cross-rerun observation cache reuse.
- `publication`: frozen `public_session` linkage (`session_id`, `published_at`, `media_directory`, `algorithm`, `chapter_alignment`, `combined_duration_seconds`, `clip_count`, `clip_ids`, `task_id`, `attempt_id`, `video_version_id`, `version_payload_sha256`).
- `limits` and `omitted_hints_count`: explicit cap metadata (`max_hints_per_clip: 5`, `max_hints_per_gap: 5`, `max_artifact_bytes: 524288`, `omitted_clip_hints_count`, `omitted_gap_hints_count`, `omitted_hints_count`, `hints_truncated`).
- `clips`: ordered kept-clip summaries (`clip_id`, `clip_index`, `file`, `original_start_ms`, `original_end_ms`, `heuristic_status`, `identity_status`, `person_presence_ratio`, `active_motion_ratio`, `ambiguous_multi_person_ratio`, `observed_frame_count`, `hints`, `total_candidate_hints`, `omitted_hints_count`, `hints_truncated`).
- `removed_gaps`: ordered removed-gap summaries (`gap_id`, `gap_index`, `original_start_ms`, `original_end_ms`, `heuristic_status`, `identity_status`, ratios, `hints`, `total_candidate_hints`, `omitted_hints_count`, `hints_truncated`).
- Allowed `heuristic_status`: `"no_detector_warnings"`, `"check_boundary_activity"`, `"extended_low_motion"`, `"person_track_interrupted"`, `"ambiguous_multiple_persons"`, `"unknown"`.
- Allowed `identity_status`: `"unverified_single_primary_track"`, `"ambiguous_multiple_persons"`, `"no_person_detected"`, `"unknown"`.
- Every hint satisfies `parent.original_start_ms <= hint.original_start_ms < hint.original_end_ms <= parent.original_end_ms <= source_duration_ms`.


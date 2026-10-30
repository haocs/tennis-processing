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

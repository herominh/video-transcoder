# Transcode contract v2 (draft)

Protocol `2.0.0-draft`, generation manifest `1.0.0-draft`. **This file is normative.** The Hub
validator (PHP, `App\Protocol\Transcode`) and the worker validator (Python, `tests/contract/` in the
worker repository) both implement exactly the rules below. When a fixture and this file disagree,
this file wins and the fixture is a bug. Nothing live uses this contract yet: `versions.json` lists
no live version, and every fixture is synthetic.

The tree is byte-identical in both repositories (Hub: `laravel/resources/contracts/transcode/v2/`,
worker: `tests/contracts/transcode/v2/`). Its files must never be normalized (line endings,
encoding, trailing newline): both repositories mark it `-text` in `.gitattributes`.

## 1. Files

| Path | Role |
|---|---|
| `README.md` | this rule text |
| `SHA256SUMS` | `<sha256>  <relative path>\n` for every other file, sorted by path bytes, LF endings |
| `versions.json` | accepted versions per line (`protocol`, `manifest`) and channel (`draft`, `live`) |
| `limits.json` | byte limits per kind and carrier, `max_json_depth`, `max_tokens`, `clock_skew_seconds`, `max_segments_per_rendition` |
| `reason-codes.json` | every reason code: layer, HTTP status (B07), retryable, reserved, first unit |
| `artifact-kinds.json` | active and reserved artifact kinds per manifest version |
| `schemas/common.schema.json` | shared `$defs` |
| `schemas/<kind>.schema.json` | one JSON Schema 2020-12 document per kind (section 7) |
| `schemas/trusted-context.schema.json` | the trusted context (section 9) |
| `schemas/fixture-catalog.schema.json` | validates `fixtures/cases.json` |
| `carriers/runpod-input.schema.json`, `carriers/runpod-output.schema.json` | RunPod carrier values (section 12) |
| `signing/vectors.json`, `signing/messages/*.bin` | known-answer signing vectors (section 12) |
| `fixtures/cases.json`, `fixtures/contexts/*.json`, `fixtures/<kind dir>/*.json` | the paired fixture catalog (section 14) |

* Every path written inside a contract JSON file is relative to the tree root, with `/` separators.
* Every schema's `$id` is `https://schemas.video-hub.invalid/transcode/v2/` followed by the file's
  relative path (for example `.../transcode/v2/schemas/common.schema.json`). Relative `$ref`s
  resolve against that `$id`.
* **Aggregate digest** = SHA-256 of the exact bytes of `SHA256SUMS`, lowercase hex. It is pinned as
  `ContractFiles::EXPECTED_DIGEST` (PHP) and `EXPECTED_CONTRACT_DIGEST` (Python). Any edit to any file
  of the tree changes `SHA256SUMS`, and both constants move in the same change.
* No real bucket, endpoint, provider id, secret or key appears anywhere in the tree.

## 2. The pipeline

A validator takes the **exact received bytes** and a **trusted context** (section 9) and returns one
verdict `{accepted, layer, reason, instance_path}`. Layers run in this order; the first failure wins:

| Order | Layer | Checks |
|---|---|---|
| 1 | `size` (L0 a) | raw byte count against the largest limit the context accepts |
| 2 | `parse` (L1) | ASCII bytes, the token budget (a `size` refusal), JSON judged over the whole text, integer literals, top-level object |
| 3 | `version` (L2) | version, then kind; selects the schema |
| 4 | `size` (L0 b) | raw byte count against the selected kind's limit |
| 5 | `schema` (L3) | JSON Schema 2020-12 of the selected kind |
| 6 | `semantic` (L4) | checks S1-S28 against the trusted context |

* An accepted verdict has `layer`, `reason` and `instance_path` all null.
* L1 yields `parse` / `malformed_json`, except its token budget, which yields `size` /
  `payload_too_large` (section 4, step 2).
* `instance_path` is an RFC 6901 JSON Pointer (`""` is the root, `~` is written `~0`, `/` is written
  `~1`, array indexes in decimal). It is null for L0 and L1 rejections and never null for L2, L3, L4.
* A context that is invalid or incomplete (section 9) is a programming error: construction throws.
  It is never turned into a verdict.

### 2.1 Message or manifest

If `context.channel` is `storage`, the payload is a generation manifest: its version field is
`manifest_version`, its kind field is `document_kind`, and the accepted versions are
`context.accepted_manifest_versions`. On every other channel the payload is a message: version field
`protocol_version`, kind field `message_kind`, accepted versions `context.accepted_protocol_versions`.
The accepted kinds are always `context.accepted_message_kinds`.

## 3. L0: size

* **L0 a**, before parsing: the raw byte count (insignificant whitespace included) must be at most the
  largest `limits.json` `max_bytes` value among the kinds in `accepted_message_kinds` (for a `storage`
  context this is the manifest limit). Otherwise: `size` / `payload_too_large`, pointer null.
* **L0 b**, right after L2 selected the kind: the raw byte count must be at most that kind's
  `max_bytes`. Otherwise: `size` / `payload_too_large`, pointer null.

## 4. L1: parse

Every L1 failure has pointer null. Its reason is `malformed_json` (layer `parse`), except step 2, which
is `payload_too_large` (layer `size`). The steps run in this order:

1. **ASCII.** Every byte of the payload is 0x09 (tab), 0x0A (LF), 0x0D (CR) or 0x20 to 0x7E.
   Otherwise `malformed_json`. This refuses a UTF-8 byte order mark, invalid UTF-8 and every non-ASCII
   character, anywhere in the text, members a later duplicate overwrites included: every accepted
   message is ASCII, so its characters are its bytes (section 12).
2. **Token budget, before decoding.** A scan counts tokens byte by byte. It is defined on any ASCII
   input, JSON or not, and is identical on both sides:
   * outside a string literal: `"` starts a string literal and counts 1; `{` counts 1; `[` counts 1;
     a byte of `-0123456789` starts a number literal, counts 1, and the scan skips every following
     byte of `0123456789+-.eE`; a byte of `tfn` starts a literal, counts 1, and the scan skips every
     following byte of `a-z`; every other byte counts 0;
   * inside a string literal: `\` skips the next byte; `"` ends the literal (an unterminated literal
     runs to the end of the text).

   When the count exceeds `limits.json` `max_tokens`, the verdict is `size` / `payload_too_large`,
   pointer null, and nothing is decoded. `max_tokens` is the token count of the maximal valid
   generation manifest (every array full, every optional member present); both suites build that
   manifest and assert `its tokens <= max_tokens <= its tokens + 1024`. The budget bounds the
   decoder's work and memory on hostile input far below the 8 MiB byte limit.
3. **Decode** as RFC 8259 JSON; a duplicate member name keeps the last value. Decoding fails when the
   text is not JSON (this includes the literals `NaN`, `Infinity` and `-Infinity`, comments, trailing
   commas, single quotes, unescaped control characters inside strings and trailing garbage;
   insignificant whitespace is space, tab, LF and CR). It also fails when any of the following holds
   anywhere in the **whole** text, members a later duplicate overwrites included:
   * the container nesting exceeds `limits.json` `max_json_depth` (32);
   * a `\u` escape forms an unpaired UTF-16 surrogate: a `\uD800`-`\uDBFF` escape not immediately
     followed by a `\uDC00`-`\uDFFF` escape, or a `\uDC00`-`\uDFFF` escape not immediately preceded
     by a high one (hex digits in either letter case);
   * an object member name's first character is U+0000 (its string literal begins with `\u0000`).
4. **Integer literals only.** Every number literal in the text matches `-?(0|[1-9][0-9]*)`: no
   fraction and no exponent, so `50.0`, `1e2` and `1.9999999999999999` are refused. The contract
   defines no non-integer number.
5. The top-level value is an object.

Steps 3 to 5 share one verdict (`malformed_json`), so their relative order does not matter.

**Depth counting.** The root container counts 1; each object or array nested inside a container adds
1; scalars add nothing. `{"a":[[1]]}` has nesting 3. Nesting 32 is accepted, 33 is refused.

**How each side judges the whole text.** PHP's `json_decode($bytes, false, 33, JSON_THROW_ON_ERROR)`
already judges the whole text for depth, surrogates and U+0000-led names while it parses (its depth
argument is nesting + 1: `json_decode('[1]', false, 1)` fails, `json_decode('[1]', false, 2)`
succeeds); PHP checks steps 1, 2 and 4 on the raw bytes. Python must not judge the decoded object,
where a duplicate has already replaced the member it overwrote: it scans the raw text (string state
as in step 2; depth = `{` and `[` nesting outside strings; surrogate pairing inside string literals;
a member name = a string literal followed by optional whitespace and `:`) and refuses fractions and
exponents through the decoder's float hook.

**Other parse rules.** PHP decodes to objects (`$associative = false`), never to arrays, so `{}` and
`[]` stay distinct. A number's magnitude never fails L1: an integer literal too large for the
platform becomes a float (PHP's behaviour; Python mirrors it for literals it cannot turn into an int),
and L3 bounds then reject it.

## 5. L2: version, then kind

In this order:

1. The version field is absent, not a string, or not in the accepted versions:
   `version` / `unsupported_version`, pointer `/protocol_version` (manifest: `/manifest_version`).
2. The kind field is absent, not a string, or not one of the six known kinds (`transcode.request`,
   `transcode.progress`, `transcode.result.completed`, `transcode.result.failed`, `hub.error`,
   `generation.manifest`): `version` / `unknown_message_kind`, pointer `/message_kind`
   (manifest: `/document_kind`).
3. The kind is known but not in `accepted_message_kinds`: `version` / `unexpected_message_kind`, same
   pointer as 2.

Versions match by exact string: `2.0.0` is not `2.0.0-draft`. There is no version negotiation.

## 6. L3: schema

| Kind | Schema |
|---|---|
| `transcode.request` | `schemas/transcode-request.schema.json` |
| `transcode.progress` | `schemas/transcode-progress.schema.json` |
| `transcode.result.completed` | `schemas/transcode-result-completed.schema.json` |
| `transcode.result.failed` | `schemas/transcode-result-failed.schema.json` |
| `hub.error` | `schemas/hub-error.schema.json` |
| `generation.manifest` | `schemas/generation-manifest.schema.json` |

* JSON Schema draft 2020-12. Every schema file of `schemas/` and `carriers/` is registered locally
  under its `$id` before validation. A `$ref` that names anything else must fail; nothing is ever
  retrieved over the network.
* No schema uses `format`. Patterns use only portable syntax (`[0-9]` rather than `\d`, no lookaround,
  no `\w`, `\s`, `\p`).
* There is no coercion: `"7"` is not an integer, `null` is not a value. (A fraction or an exponent
  never reaches L3: section 4, step 4 refuses it.)
* Every string that is not an `enum` or `const` carries `maxLength` and a character ban in
  `not.pattern`: control characters `[\x00-\x1f\x7f]`; anything but printable ASCII `[^\x20-\x7e]`
  for free text; anything but printable ASCII and JSON whitespace `[^\x09\x0a\x0d\x20-\x7e]` for a
  carrier's `vh_message`. Free text additionally refuses `://` (section 7). Without the ban, a
  trailing newline would pass `^...$` in Python, whose `$` matches before a final newline under
  `re.search` (opis runs PCRE with the `D` flag, where `$` does not); with the ban both refuse it.
  Every array carries `maxItems`; every integer carries `minimum` and `maximum` within
  +-(2^53 - 1).
* The reason is always `schema_violation`. Each implementation stops at the **first** error its
  library reports (PHP: opis with max errors 1; Python: the first item of jsonschema's
  `iter_errors`), which bounds the work on hostile input, and that error's instance pointer is the
  verdict's `instance_path`. A wrapper error (`properties`, `items`, `additionalProperties` with a
  schema, `$ref`, `allOf`, `if`/`then`/`else`) is followed down to the error it wraps; `anyOf`, `oneOf`, `not`, `contains` and
  `propertyNames` are reported at their own instance (jsonschema reports them that way natively).
* Parity at L3 is on `accepted`, `layer` and `reason`. The pointer is **diagnostic**: on a document
  with several violations the two libraries may name different instances, so no decision may use it
  (B07's `hub.error.pointer` is informative). Each catalog L3 negative holds exactly one violation,
  and its pinned pointer is the one BOTH libraries report first: both suites assert equality.

## 7. Message kinds (summary; the schemas are exact)

| Kind | Direction | Limit (bytes) | Carries |
|---|---|---:|---|
| `transcode.request` | Hub to worker | 4,096 | envelope, identity without execution fields, source locator, output location, profile, renditions, encryption mode and key id, limits, claim bootstrap |
| `transcode.progress` | worker to Hub | 4,096 | envelope, full identity, `event_seq`, stage, stage percentage, optional message, counters, `ext` |
| `transcode.result.completed` | worker to Hub | 16,384 | envelope, full identity, `generation_id`, manifest locator, source read facts, profile, media, encoder, optional diagnostics, `ext` |
| `transcode.result.failed` | worker to Hub | 16,384 | envelope, full identity, typed error, optional source, encoder, diagnostics, `ext` |
| `hub.error` | Hub to worker (HTTP body) | 2,048 | version, kind, id, `sent_at`, error; **no** audience, key id or identity; unsigned in B03 |
| `generation.manifest` | stored by the worker, read by the Hub | 8,388,608 | manifest version, kind, generation, identity (no cell, epoch, revision), profile, encryption, media, renditions, paths, inventory |

Envelope = `protocol_version`, `message_kind`, `message_id`, `sent_at`, `audience`, `key_id`.
Identity = `org_uuid`, `video_uuid`, `owning_cell_id`, `placement_epoch`, `lifecycle_revision`,
`attempt_id`, `dispatch_id`, plus `execution_id` and `execution_fence` on progress and results. The
manifest identity is `org_uuid`, `video_uuid`, `source_id`, `attempt_id`, `dispatch_id`,
`execution_id`.

**Secrets and free text.** No field of any kind is defined to carry a media key, a storage credential
or a presigned URL. Free text (`message`, `error.detail`, `error.message`, `diagnostics[].detail`,
`encoder.ffmpeg_version`) and `ext` string values are printable ASCII and refuse the substring `://`
(the `ascii_text_*` definitions: `not` `[^\x20-\x7e]` and `allOf` `not` `://`). Free text is
producer-sanitized and never holds a URL (`://` is refused); the contract cannot prove that opaque
text holds no key material, so receivers treat free text as untrusted: bounded, never interpreted,
never logged beyond its bounded form.

Bounds of note: `source.size_bytes`, `limits.max_source_bytes` 1 to 32,212,254,720 (30 GiB);
`source.bytes_read` 0 to 32,212,254,720; `limits.max_source_duration_ms` and `media.duration_ms` 1 to
21,600,000 (6 h); `limits.max_wall_time_ms` 1 to 86,400,000; output bytes up to 1,099,511,627,776;
fences and revisions up to 9,007,199,254,740,991 (2^53 - 1); at most 25,209 artifacts and 3,600
segments per rendition. The HLS segment duration is fixed platform-wide (6 s, carried by the media
profile) and is not a request field.

## 8. L4: semantic checks

### 8.1 Timestamps, freshness, comparisons

* A timestamp is `YYYY-MM-DDTHH:MM:SS.mmmZ` (UTC, milliseconds). The schema bounds month 01-12,
  day 01-31, hour 00-23, minute and second 00-59. **S1** additionally requires a real instant of the
  proleptic Gregorian calendar: year 0000 is invalid, and the day must exist in its month
  (February has 29 days in years divisible by 4, except years divisible by 100 and not by 400).
* Freshness arithmetic is in integer milliseconds since 1970-01-01T00:00:00.000Z:
  `now_ms` from `context.now`, `skew_ms = context.clock_skew_s * 1000`. Bounds are inclusive.
* Identifiers and other strings compare by exact equality (case-sensitive; the schema already forces
  lowercase UUIDs). Numbers compare **numerically** (`7.0` equals `7`; PHP must not use `===` between
  an int and a float).

### 8.2 The checks

The checks run in the order S1 to S28, restricted to the ones that apply (8.3). The first failing check
is the verdict: layer `semantic`, its reason and its pointer. Each check reports at most one pointer,
chosen by the rule in the table.

| # | Fails when | Reason | Pointer |
|---|---|---|---|
| S1 | a timestamp of the kind is not a real instant (8.1) | `invalid_timestamp` | the first invalid of `/sent_at`, `/claim/expires_at` (request), `/created_at` (manifest) |
| S2 | `audience != context.expected_audience` | `audience_mismatch` | `/audience` |
| S3 | `key_id` not in `context.known_key_ids` | `unknown_key_id` | `/key_id` |
| S4 | `sent_at_ms > now_ms + skew_ms` | `timestamp_in_future` | `/sent_at` |
| S5 | `sent_at_ms < now_ms - skew_ms` (channel `push` only) | `timestamp_out_of_tolerance` | `/sent_at` |
| S6 | `claim.expires_at_ms <= sent_at_ms`, or `now_ms > claim.expires_at_ms + skew_ms` | `dispatch_expired` | `/claim/expires_at` |
| S7 | one of these ids equals `identity.video_uuid`, checked in this order: `identity.attempt_id`, `identity.dispatch_id`, `identity.execution_id` (when present), the source id (`/source/source_id` on request and results when the `source` block is present; `/identity/source_id` on the manifest), `identity.org_uuid` | `identity_aliases_video_uuid` | the pointer of the first aliasing id |
| S8 | `identity.org_uuid != expect.org_uuid` | `org_mismatch` | `/identity/org_uuid` |
| S9 | `identity.video_uuid != expect.video_uuid` | `video_mismatch` | `/identity/video_uuid` |
| S10 | `identity.owning_cell_id != expect.owning_cell_id` | `cell_mismatch` | `/identity/owning_cell_id` |
| S11 | `identity.placement_epoch` lower / higher than `expect.placement_epoch` | `stale_placement_epoch` / `placement_epoch_ahead` | `/identity/placement_epoch` |
| S12 | `identity.lifecycle_revision` lower / higher than `expect.lifecycle_revision` | `stale_lifecycle_revision` / `lifecycle_revision_ahead` | `/identity/lifecycle_revision` |
| S13 | `identity.attempt_id != expect.attempt_id` | `attempt_mismatch` | `/identity/attempt_id` |
| S14 | `identity.dispatch_id != expect.dispatch_id` | `dispatch_mismatch` | `/identity/dispatch_id` |
| S15 | `identity.execution_id != expect.execution_id` | `execution_mismatch` | `/identity/execution_id` |
| S16 | `identity.execution_fence` lower / higher than `expect.execution_fence` | `stale_execution_fence` / `execution_mismatch` | `/identity/execution_fence` |
| S17 | the source id differs from `expect.source_id`; skipped when an optional `source` block is absent (failed result) | `source_mismatch` | `/source/source_id` (request, results), `/identity/source_id` (manifest) |
| S18 | request: `source.location_id != expect.source_location_id`, then `output.location_id != expect.output_location_id`; completed result: `manifest.location_id != expect.output_location_id` | `storage_location_mismatch` | `/source/location_id`, then `/output/location_id`; `/manifest/location_id` |
| S19 | any of `profile.id`, `profile.version`, `profile.sha256` differs from `expect.profile` | `profile_mismatch` | `/profile` |
| S20 | `encryption.mode != expect.encryption.mode`, or `media_key_id` differs (absent equals only absent) | `encryption_mismatch` | `/encryption` |
| S21 | `generation_id != identity.execution_id` | `generation_execution_mismatch` | `/generation_id` |
| S22 | `event_seq <= context.last_event_seq` | `stale_event_sequence` | `/event_seq` |
| S23 | an artifact's `path` equals the path of an earlier artifact | `duplicate_artifact_path` | `/artifacts/<j>/path` of the first such later occurrence |
| S24 | the first failing of these, in order: (a) `artifact_count` differs from the number of artifacts; (b) `total_bytes` differs from the sum of `size_bytes`; (c) `media.rendition_count` differs from the number of renditions; (d) a rendition `name` repeats an earlier one; (e) the number of `hls_master_playlist` artifacts is not exactly 1; (f) a rendition (in array order) does not have exactly one `hls_media_playlist` artifact and at least one `hls_segment` artifact whose `rendition` is its name | `artifact_inventory_mismatch` | (a) `/artifact_count`, (b) `/total_bytes`, (c) `/media/rendition_count`, (d) `/renditions/<j>/name` of the repeat, (e) `/artifacts`, (f) `/renditions/<i>` |
| S25 | the first failing of these, in order: (a) no artifact has path `master_playlist_path` with kind `hls_master_playlist`; (b) for rendition `i` in array order, no artifact has path `playlist_path` with kind `hls_media_playlist` and `rendition` equal to the rendition's name; (c) `thumbnail_path` is not null and no artifact has that path with kind `thumbnail`; (d) an artifact (in array order) has a `rendition` that names no entry of `renditions` | `dangling_reference` | (a) `/master_playlist_path`, (b) `/renditions/<i>/playlist_path`, (c) `/thumbnail_path`, (d) `/artifacts/<j>/rendition` |
| S26 | with `T` = the number of `thumbnail` artifacts and `D` = whether `diagnostics` holds a code `thumbnail_unavailable`: when `thumbnail_path` is null the check requires `D` and `T == 0`; when it is not null it requires `T == 1` | `thumbnail_diagnostic_missing` | `/thumbnail_path` |
| S27 | a rendition name (in array order) is not in `expect.renditions` | `rendition_not_requested` | `/renditions/<i>/name` of the first such rendition |
| S28 | `artifact_count > expect.max_artifact_count`; else a rendition (in array order) has more than `limits.json` `max_segments_per_rendition` (3,600) `hls_segment` artifacts | `limit_exceeded` | `/artifact_count`; else `/renditions/<i>` |

### 8.3 Which checks run

Checks depend on the context's `role`, the selected kind and the channel. A (role, kind) pair
without a row is not part of the contract (section 9.2). `worker_sender` is the worker's self-check
before sending and runs exactly what `hub_receiver` runs for the same kind; `hub_sender` is the Hub's
self-check of its own dispatch.

| Role | Kind | Allowed channels | Checks, in order |
|---|---|---|---|
| `worker_receiver` | `transcode.request` | `dispatch` | S1 S2 S3 S4 S6 S7 |
| `worker_receiver` | `hub.error` | `push` | S1 |
| `hub_sender` | `transcode.request` | `dispatch` | S1 S2 S3 S4 S6 S7 S8 S9 S10 S11 S12 S13 S14 S17 S18 S19 S20 |
| `hub_receiver`, `worker_sender` | `transcode.progress` | `push`, `poll` | S1 S2 S3 S4 S5 S7 S8 S9 S10 S11 S12 S13 S14 S15 S16 S22 |
| `hub_receiver`, `worker_sender` | `transcode.result.completed` | `push`, `poll` | S1 S2 S3 S4 S5 S7 S8 S9 S10 S11 S12 S13 S14 S15 S16 S17 S18 S19 S21 |
| `hub_receiver`, `worker_sender` | `transcode.result.failed` | `push`, `poll` | S1 S2 S3 S4 S5 S7 S8 S9 S10 S11 S12 S13 S14 S15 S16 S17 |
| `hub_receiver`, `worker_sender` | `generation.manifest` | `storage` | S1 S7 S8 S9 S13 S14 S15 S17 S19 S20 S21 S23 S24 S25 S26 S27 S28 |

S5 runs only when the channel is `push`: the `poll` channel (provider output read later) waives the
past bound, and the `dispatch` channel judges the claim expiry instead (S6). The manifest carries no
cell, epoch, revision or freshness: those are rechecked when the generation is published (B10).

## 9. The trusted context

The context carries what the validating side knows from its own records; identity is compared with
it, never taken from the message. `schemas/trusted-context.schema.json` is exact.

### 9.1 Fields

| Field | Type | Meaning |
|---|---|---|
| `context_version` | integer 1 | required |
| `role` | `hub_receiver`, `hub_sender`, `worker_receiver`, `worker_sender` | required |
| `channel` | `push`, `poll`, `dispatch`, `storage` | required; selects message or manifest (2.1) and the freshness rule |
| `accepted_message_kinds` | 1 to 6 distinct kinds | required |
| `now` | timestamp or null | injected clock |
| `clock_skew_s` | integer 0 to 600 or null | `limits.json` `clock_skew_seconds` (300) in production |
| `accepted_protocol_versions`, `accepted_manifest_versions` | up to 8 distinct versions, or null | from `versions.json` (`live`; `draft` in tests only); may be empty |
| `expected_audience` | audience or null | the receiver's own audience (a sender self-check: the target's) |
| `known_key_ids` | up to 8 distinct key ids, or null | the keys valid for this direction |
| `expect` | object or null | recorded identity: `org_uuid`, `video_uuid`, `owning_cell_id`, `placement_epoch`, `lifecycle_revision`, `attempt_id`, `dispatch_id`, `execution_id`, `execution_fence`, `source_id`, `source_location_id`, `output_location_id`, `profile {id, version, sha256}`, `encryption {mode, media_key_id}`, `renditions`, `max_artifact_count`; each optional and nullable |
| `last_event_seq` | integer 0 to 2,147,483,647 or null | highest accepted progress sequence of the execution |

### 9.2 Construction

Building a context (PHP `TrustedContext::fromArray()`, Python `TrustedContext.from_dict()`) checks, in
this order:

1. The data satisfies `trusted-context.schema.json`; `now`, when present, is a real instant (S1's
   rule); every entry of `accepted_protocol_versions` is a `protocol` version of `versions.json`
   (draft or live) and every entry of `accepted_manifest_versions` a `manifest` version (a context
   cannot create protocol support). Otherwise: an invalid-context error.
2. For every kind in `accepted_message_kinds`, the table of 8.3 has a row for (`role`, kind), and
   `channel` is one of that row's channels. Otherwise: an invalid-context error.
3. For every kind in `accepted_message_kinds`, in list order, every field that 9.3 requires for
   (`role`, kind), in table order, is present and not null. Otherwise: **`ContextIncomplete`** (PHP
   exception, Python exception class of the same name), naming the role, the kind and the first
   missing field.

Both errors are programming errors raised at construction, never a verdict and never a skipped check.
The invalid-context error is a separate type from `ContextIncomplete` (Python: `ContextInvalid`, a
`ValueError`).

The schema judges the context as a JSON document: the root, `expect`, `expect.profile` and
`expect.encryption` are objects (an empty one is `{}`), and `accepted_message_kinds`, both version
lists, `known_key_ids` and `expect.renditions` are arrays. An implementation whose native map cannot
tell the two apart (a PHP array) builds that document explicitly, never by a JSON round trip: an empty
PHP array at an object position is `{}`, and a PHP array that is not a list (`array_is_list`) where an
array is required is an invalid context. PHP also builds a context from a decoded object
(`TrustedContext::fromObject()`, context files decoded with `$associative = false`).

### 9.3 Required fields per (role, kind)

Every context requires `context_version`, `role`, `channel` and `accepted_message_kinds` (the schema
enforces them). In addition, a field is required exactly when L2 or a check that runs for (role, kind)
reads it, whatever the channel:

| Role | Kind | Required fields, in order |
|---|---|---|
| `worker_receiver` | `transcode.request` | `accepted_protocol_versions`, `expected_audience`, `known_key_ids`, `now`, `clock_skew_s` |
| `worker_receiver` | `hub.error` | `accepted_protocol_versions` |
| `hub_sender` | `transcode.request` | `accepted_protocol_versions`, `expected_audience`, `known_key_ids`, `now`, `clock_skew_s`, `expect.org_uuid`, `expect.video_uuid`, `expect.owning_cell_id`, `expect.placement_epoch`, `expect.lifecycle_revision`, `expect.attempt_id`, `expect.dispatch_id`, `expect.source_id`, `expect.source_location_id`, `expect.output_location_id`, `expect.profile`, `expect.encryption` |
| `hub_receiver`, `worker_sender` | `transcode.progress` | `accepted_protocol_versions`, `expected_audience`, `known_key_ids`, `now`, `clock_skew_s`, `expect.org_uuid`, `expect.video_uuid`, `expect.owning_cell_id`, `expect.placement_epoch`, `expect.lifecycle_revision`, `expect.attempt_id`, `expect.dispatch_id`, `expect.execution_id`, `expect.execution_fence`, `last_event_seq` |
| `hub_receiver`, `worker_sender` | `transcode.result.completed` | `accepted_protocol_versions`, `expected_audience`, `known_key_ids`, `now`, `clock_skew_s`, `expect.org_uuid`, `expect.video_uuid`, `expect.owning_cell_id`, `expect.placement_epoch`, `expect.lifecycle_revision`, `expect.attempt_id`, `expect.dispatch_id`, `expect.execution_id`, `expect.execution_fence`, `expect.source_id`, `expect.output_location_id`, `expect.profile` |
| `hub_receiver`, `worker_sender` | `transcode.result.failed` | `accepted_protocol_versions`, `expected_audience`, `known_key_ids`, `now`, `clock_skew_s`, `expect.org_uuid`, `expect.video_uuid`, `expect.owning_cell_id`, `expect.placement_epoch`, `expect.lifecycle_revision`, `expect.attempt_id`, `expect.dispatch_id`, `expect.execution_id`, `expect.execution_fence`, `expect.source_id` |
| `hub_receiver`, `worker_sender` | `generation.manifest` | `accepted_manifest_versions`, `expect.org_uuid`, `expect.video_uuid`, `expect.attempt_id`, `expect.dispatch_id`, `expect.execution_id`, `expect.source_id`, `expect.profile`, `expect.encryption`, `expect.renditions`, `expect.max_artifact_count` |

`expect.<name>` is missing when `expect` itself is missing or null. An empty version list is present
(not missing): it makes every message `unsupported_version`.

## 10. Compatibility rules

| Id | Rule |
|---|---|
| C1 | A version field that is absent, not a string, or not accepted is rejected at L2 (`unsupported_version`) before any schema runs. |
| C2 | No protocol major other than 2 and no manifest major other than 1 is accepted by a B03 validator. |
| C3 | Drafts match by exact string; no draft is compatible with another version. |
| C4 | `versions.json` `live` is empty for both lines in B03; any payload validated under a live context is `unsupported_version`. |
| C5 | A receiver accepts a minor version only when it lists it. Readers upgrade first (accept N and N+1), writers second. There is no "accept a newer minor and ignore what is unknown" path. |
| C6 | Every object of every schema has `additionalProperties: false` (the six kinds, the context, the carriers, the catalog), except `ext`. |
| C7 | `ext` is the only extension point: optional on progress and results, at most 8 members named `x_[a-z0-9_]{1,30}`, values a string of at most 64 printable ASCII characters that does not contain `://`, an integer within +-(2^53 - 1) or a boolean. Receivers ignore `ext` in every decision. |
| C8 | Every identity field is required and non-null, with no default; an identity field equal to `video_uuid` where it names another entity is S7. |
| C9 | A key id outside the context's set for that direction is `unknown_key_id`; there is no fallback key. |
| C10 | Number literals are integer literals (4, step 4). No coercion (6): a number sent as a string, or `null` where a value is required, fails L3. |
| C11 | Enums are closed: an unknown rendition, stage, artifact kind, error class or encoder is rejected, including values reserved for later versions. |
| C12 | Size limits apply to the raw bytes before parsing, whitespace included. |
| C13 | Channel `push`: `now - skew <= sent_at <= now + skew` (S4, S5). |
| C14 | Channel `poll`: only `sent_at <= now + skew` (S4). |
| C15 | Channel `dispatch`: `sent_at <= now + skew` (S4), `claim.expires_at > sent_at` and `now <= claim.expires_at + skew` (S6). |

## 11. Artifact kinds (manifest `1.0.0-draft`)

| Kind | Status | `rendition` | Count rule (L4) |
|---|---|---|---|
| `hls_master_playlist` | active | forbidden | exactly 1, named by `master_playlist_path` |
| `hls_media_playlist` | active | required | exactly 1 per rendition, named by its `playlist_path` |
| `hls_segment` | active | required | 1 to 3,600 per rendition |
| `thumbnail` | active | forbidden | 1 when `thumbnail_path` is set, else 0 (with a `thumbnail_unavailable` diagnostic) |
| `captions`, `transcript`, `chapters`, `storyboard` | reserved | - | rejected in 1.0 (L3) |

The schema enum holds only the active kinds of the declared manifest version, so a reserved or
unknown kind fails L3 and the whole manifest is rejected; it is never skipped. Adding a kind means a
new manifest minor that the Hub reader accepts before any worker writes it. `artifact-kinds.json` lists
active and reserved kinds per manifest version and equals the schema enum. The encryption key and the
manifest itself are never artifacts.

## 12. Signing and carriers

* **Signing input:** the 23 ASCII bytes `video-hub.transcode.v2\n` followed by the exact message
  bytes. No canonicalization: a re-serialized message does not verify.
* **Algorithm:** HMAC-SHA256 with the key that the message's `key_id` names; keys are per direction
  (a Hub-to-worker key never verifies a worker-to-Hub message). The value is `hmac-sha256=` followed
  by 64 **lowercase** hex characters.
* **`sign(bytes, keyBytes) -> string`** returns that value. **`verify(bytes, keyBytes, value) -> bool`**
  is true only when `value` is exactly `hmac-sha256=` plus 64 lowercase hex characters and equals the
  signature under a constant-time comparison (`hash_equals` / `hmac.compare_digest`); any malformed
  value (wrong prefix, uppercase, wrong length, not a string) is `false`, never an exception. The key
  must be non-empty bytes (a programming error otherwise). There is no key storage, key lookup or
  header parsing beyond the `hmac-sha256=` prefix.
* **HTTP carrier** (worker-to-Hub callback, and the `hub.error` response): the body is the message
  bytes; the header `X-VH-Signature: hmac-sha256=<hex>`. `sent_at` is in the body; there is no
  timestamp header. `hub.error` is unsigned in B03.
* **RunPod input** (`carriers/runpod-input.schema.json`) validates the value of RunPod's `input`
  object: `{"vh_message": <message text>, "vh_signature": "hmac-sha256=<hex>"}`, nothing else;
  `vh_message` at most 4,096 characters. **RunPod output** (`carriers/runpod-output.schema.json`) is
  the same shape, returned by the worker handler for both outcomes, `vh_message` at most 16,384
  characters. `vh_message` holds only printable ASCII and JSON whitespace (tab, LF, CR); every
  accepted message is ASCII in all its bytes (section 4, step 1, overwritten duplicates included),
  so characters equal bytes, and decoding the JSON string yields the exact message bytes. The raw limits of `limits.json` `carrier_max_bytes` (`runpod.input` 8,448,
  `runpod.output` 33,024) apply to the serialized carrier value.
* **Live receiver order (B07, not B03):** L0, L1, L2, then key lookup by `key_id` (`unknown_key_id`),
  then signature verification over the exact received bytes (`invalid_signature`), then L3, L4 and
  dedupe (`duplicate_message`, `conflicting_terminal_result`). B03 fixtures carry no signatures, so B03
  checks key id membership in L4 (S3).
* **`signing/vectors.json`**: `keys` (`key_id`, `direction`, `key_hex`; test-only keys),
  `vectors` (`id`, `category`, `purpose`, `message_file`, `key_id` = the key to verify with,
  `v1_timestamp` on the `v1_signing_input` vector, `value`, `valid`), and `carrier_examples` (`carrier`, `schema`, `vector`, `value`: a carrier value whose
  `vh_message` decodes to the vector's message bytes and whose `vh_signature` is the vector's value).
  For a valid vector, `sign(message, key) == value` and `verify` is true; for an invalid one `verify`
  is false. Categories: `valid`, `body_byte_changed`, `reserialized`, `v1_signing_input` (HMAC over the
  v1 input `v1_timestamp + "." + body`), `other_direction_key` (the body signed with the Hub-to-worker
  key, verified with a worker-to-Hub key), `malformed_value`. Both suites recompute the
  `v1_signing_input` and `other_direction_key` values and compare them with the vector.

## 13. Limits and reason codes

`limits.json`: `max_bytes` per kind (request 4,096; progress 4,096; completed 16,384; failed 16,384;
`hub.error` 2,048; manifest 8,388,608), `carrier_max_bytes` (`runpod.input` 8,448, `runpod.output`
33,024), `max_json_depth` 32, `max_tokens` (section 4, step 2), `clock_skew_seconds` 300,
`max_segments_per_rendition` 3,600. The worst case of every kind (every string at its maximum length
in the costliest characters, every integer at its maximum, every array full, every optional member
present, `/` written as `\/`) passes its schema and fits its limit; both suites build it, prove it
maximal against the schema's bounds, and measure it.

**Writer convention.** Senders serialize compactly: no insignificant whitespace, only the escapes
JSON requires (optionally `/` as `\/`), integers as integer literals. The byte limits are guaranteed
for such writers; receivers accept any JSON text within the byte limit.

`reason-codes.json` lists every code in a fixed order with `layer` (`size`, `parse`, `version`,
`schema`, `semantic`; reserved codes: `auth`, `dedupe`, `fetch` or null), `http_status` (what B07
returns; null when the code never becomes an HTTP response), `retryable`, `reserved` (not produced by
the B03 pipeline: `invalid_signature`, `duplicate_message`, `conflicting_terminal_result`,
`manifest_digest_mismatch`, `internal_error`, `service_unavailable`) and `first_unit`. The PHP and
Python enums equal this list, and `hub.error`'s `error.code` enum equals it too.

## 14. The fixture catalog

`fixtures/cases.json` (`catalog_version` 1) lists every case: `id`, `payload` (a file of exact
bytes), `context` (a context file), `rule`, `purpose`, `context_needed` (informative: whether the
verdict depends on the context's content) and `expect {outcome, layer, reason, instance_path}`.
A payload may be reused by several cases with different contexts.

To run a case: read the payload file's bytes exactly as stored, build the context from the context
file (9.2), validate, and assert:

* `accept`: the verdict is accepted.
* `reject`: `layer`, `reason` and `instance_path` equal the pinned values at every layer
  (`instance_path` is null for L0 and L1; for `schema` it is the first error both libraries report,
  section 6).

Catalog rules: 21 positive and 114 negative cases, ids `P01`-`P22` and `N01`-`N114` (two or three
digits). `P17` is retired: it pinned an integer-valued float as accepted, and its payload became
`N76` when number literals had to be integer literals. Large payloads (`P21`, `N84`, `N112`, `N113`)
are stored compact; every file stays under 1 MB; every non-reserved reason code is produced by at
least one negative; every fixture file is used by a case. Each negative starts from a valid positive
and changes one thing, so it holds one violation at its layer: an L3 negative produces errors only
at its pinned pointer (in jsonschema's flattened report), and an L4 negative fails only its pinned
check, except that an S7 case also fails the equality check of the aliased id. A negative may still
break a later layer as a consequence (N27 renames the master playlist, so `master_playlist_path`
would dangle at L4); the pipeline stops at the first failing layer. N03 (nesting 33, L1) and N70
(nesting 32, accepted by L1, refused by L3) pin the depth boundary.

Synthetic values: now `2030-01-01T00:00:00.000Z`; org `11111111-1111-7111-8111-111111111111`,
video `22222222-2222-7222-8222-222222222222`, attempt `33333333-...`, dispatch `44444444-...`,
execution and generation `55555555-...`, source `66666666-...`, media key id `77777777-...`, source
location `aaaaaaaa-...`, output location `bbbbbbbb-...`; cell `cell-test-a`, epoch 7, lifecycle
revision 3, execution fence 2, last event sequence 4; audiences `hub:cell-test-a` and
`worker:runpod-test`; key ids `test-wk2hub-k1`, `test-wk2hub-k2` (worker to Hub) and
`test-hub2wk-k1` (Hub to worker); profile `test-h264-sdr` version 1.

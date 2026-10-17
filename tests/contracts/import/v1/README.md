# Import contract v1

The contract between the Hub and the Google Drive import's worker (issue #50 item 3b; the import's design, section 6a
and decision G-14). **This file is normative.** The Hub validator (PHP, `App\Protocol\Import`) and the worker's
validator (Python, in the worker repository) both implement exactly the rules below; when a fixture and this file
disagree, this file wins and the fixture is a bug. Every fixture is synthetic: no real bucket, endpoint, file id,
provider id, secret or key appears anywhere in the tree.

The tree is byte-identical in both repositories (Hub: `laravel/resources/contracts/import/v1/`, worker:
`tests/contracts/import/v1/`). Its files are never normalized (line endings, encoding, trailing newline): both
repositories mark the tree `-text` in `.gitattributes`. It is separate from the transcode contract v2, which it does
not change, and it has no slot in the release manifest during the pilot.

## 1. Files

| Path | Role |
|---|---|
| `README.md` | this rule text |
| `SHA256SUMS` | `<sha256>  <relative path>\n` for every other file, sorted by path bytes, LF endings |
| `input.schema.json` | the job's input, which the Hub sends to RunPod (section 3) |
| `output.schema.json` | the job's output, which RunPod keeps and the Hub reads for its code (section 4) |
| `receipt.schema.json` | the receipt the worker writes beside the original after its last part (section 5) |
| `fixtures/cases.json` | every fixture of `fixtures/input/`, `fixtures/output/` and `fixtures/receipt/` with the schema it is checked against and its verdict, `accept` or `reject` |
| `fixtures/<document>/*.json` | the fixtures: each `valid-*` one is accepted; each `invalid-*` one breaks exactly one rule |
| `fixtures/input-mac-vector.json` | the known answer of the input's MAC (section 3.1) |
| `fixtures/receipt-mac-vector.json` | the known answer of the receipt's MAC (section 5.2) |

* **Aggregate digest** = SHA-256 of the exact bytes of `SHA256SUMS`, lowercase hex. It is pinned as
  `App\Protocol\Import\ImportContractFiles::EXPECTED_DIGEST` on the Hub and by the worker's own constant. Any edit to
  any file of the tree changes `SHA256SUMS`, and both pins move in the same change, in both repositories.
* Every schema's `$id` is `https://schemas.video-hub.invalid/import/v1/` followed by the file's name. A `$ref` names a
  `$defs` member of the same file only: nothing is ever fetched.

## 2. Validation

* JSON Schema 2020-12. `format` is not used. Every object of every document is closed (`additionalProperties:
  false`): a member the contract does not name is refused, so neither a key nor any text of Google's or RunPod's can
  ride in a document.
* Every member a schema names is required; a member that may be empty is `null`, never absent.
* Every string with a pattern also refuses a control character (`not: {pattern: "[\\x00-\\x1f\\x7f]"}`): a pattern's
  `$` matches before a final line break in some regular-expression engines, and this keeps both validators equal.
* An integer member is a JSON number with an integer value. The bounds below are inclusive.

## 3. The input

What the Hub sends to the import endpoint for one job, built after the job's claim: `{contract, job, file_id,
resource_key, expected, plan, parts, receipt_url, writes_end_at, mac}`.

| Member | Rule |
|---|---|
| `contract` | exactly `"import/v1"` |
| `job` | the job's UUID, lowercase, 8-4-4-4-12 hexadecimal |
| `file_id` | the Drive file id: `^[A-Za-z0-9_-]{10,128}$` |
| `resource_key` | `null`, or the link's resource key: `^[A-Za-z0-9_-]{1,128}$` |
| `expected.size` | the file's size in bytes as Google gave it at the paste: 1 to 10,737,418,240 (10 GiB) |
| `expected.md5`, `expected.sha1`, `expected.sha256` | `null`, or Google's checksum in lowercase hexadecimal of 32, 40 and 64 characters |
| `expected.head_revision_id` | `null`, or Google's head revision: 1 to 255 printable ASCII characters, no space |
| `plan.part_size` | exactly 67,108,864 (64 MiB) |
| `plan.parts_total` | 1 to 160 |
| `parts` | 1 to 160 items `{n, url}`: `n` from 1 to 160, `url` an `https://` URL of printable ASCII, no space, at most 4,096 characters |
| `receipt_url` | an `https://` URL, as `parts[].url` |
| `writes_end_at` | the writer deadline, RFC 3339 in UTC: `YYYY-MM-DDThh:mm:ssZ`, with up to six digits of fraction before the `Z` |
| `mac` | the input's MAC (section 3.1): 64 lowercase hexadecimal characters |

Beyond the schema, the worker refuses the input as `input_refused`, before any request, when: its MAC does not verify
(section 3.1, checked first); `plan.parts_total` is not ⌈`expected.size` ÷ `plan.part_size`⌉; the part numbers are not exactly 1 to `plan.parts_total`, each once; a URL is
not `https://{IMPORT_STORAGE_HOST}/{bucket}/original-videos/{video uuid}/…` with no user information, no port and no
other host, or its bucket and video uuid are not the same in every URL; a part URL is not `original.{extension}?…`
with `partNumber` its own `n` and the one same `uploadId` throughout; the receipt URL is not
`import-receipt-{job}.json?…` with no `uploadId`; or `writes_end_at` has already passed. Every request the worker makes
refuses a redirect. The input is signed (section 3.1): whoever can send a job without the import key spends the
endpoint's time and can replay a job's input byte for byte, which is a second start of the same import (its receipt
must still agree with the store's listing), but cannot change any member of it; whoever can read a job's input can
write its parts and a receipt until `writes_end_at`, but not a receipt the Hub accepts (section 5).

### 3.1 The input's MAC

`mac` = HMAC-SHA256, written as 64 lowercase hexadecimal characters, keyed with the **import key**, which is the
receipt key of section 5.1 (the text of the Hub's secret setting `drive_import_receipt_key` and of the import
endpoint's RunPod Secret `IMPORT_RECEIPT_KEY`, taken as its UTF-8 bytes), over the bytes of `import/v1 input` and one
line feed (`import/v1 input\n`, 16 bytes) followed by the UTF-8 bytes of the input's **canonical form**: the input
object without its `mac` member, written as section 5.1 writes a receipt (the members of every object in the byte
order of their names, arrays in their own order, no whitespace, every number an integer, the same string escapes,
`null` as `null`). The prefix keeps an input's MAC from ever being a receipt's, whose bytes have none.

The worker checks the MAC after the schema and before every other rule of section 3 and before any request,
comparing in constant time: a worker without the key answers `not_configured`, and a MAC that does not verify is
`input_refused`. So the worker acts only on an input the Hub wrote, member for member: another file id, other
checksums, other URLs or another deadline beside a real job's URLs do not verify.

`fixtures/input-mac-vector.json` holds a key (synthetic, the receipt vector's), the prefix, an input, its canonical form
and its MAC. Each side computes the canonical form of `input` and the MAC under `key`, and must find exactly
`canonical` and `mac`, which is also the input's own `mac`. Every `valid-*` input fixture carries the MAC that
verifies under that key; every `invalid-*` one breaks exactly one rule of the schema and carries a well-formed MAC,
but the two `invalid-mac-*`, whose one broken rule is the MAC's.

## 4. The output

What the worker returns to RunPod, which the Hub reads for the job's code only: `{contract, status, code}`.

| Member | Rule |
|---|---|
| `contract` | exactly `"import/v1"` |
| `status` | `imported` or `failed` |
| `code` | `null` when `imported`; when `failed`, one of `file_removed`, `link_not_shared`, `provider_quota`, `provider_refused`, `not_configured`, `size_exceeded`, `size_short`, `integrity_failed`, `source_changed_at_provider`, `deadline`, `download_failed`, `upload_failed`, `input_refused` |

The import's result is never read from the output: it comes from the store and the receipt.

## 5. The receipt

The worker that downloaded the whole file, matched Google's checksum and wrote every part writes one receipt, with one
presigned PUT, to `original-videos/{video uuid}/import-receipt-{job}.json` (at most 64 KiB): `{contract, job, bytes,
checksum_verified, parts, mac}`.

| Member | Rule |
|---|---|
| `contract` | exactly `"import/v1"` |
| `job` | the job's UUID, as in the input |
| `bytes` | the bytes downloaded and written: 1 to 10,737,418,240 |
| `checksum_verified` | `sha256`, `md5` or `none`: the strongest checksum of Google's the whole stream matched |
| `parts` | 1 to 160 items `{n, etag}`: `n` from 1 to 160, `etag` the store's ETag of that part's PUT without its quotes, 1 to 128 printable ASCII characters with no space and no `"` |
| `mac` | the receipt's MAC (section 5.1): 64 lowercase hexadecimal characters |

The Hub refuses a receipt that is not this schema's, of another job, of another size than the paste's, with a
checksum weaker than the input expected, or whose MAC does not verify; it then counts as no receipt at all. A receipt
can only refuse an object: the Hub accepts no part the store does not list.

### 5.1 The MAC

`mac` = HMAC-SHA256, written as 64 lowercase hexadecimal characters, keyed with the **receipt key** (the text of the
Hub's secret setting `drive_import_receipt_key`, which is also the import endpoint's RunPod Secret, taken as its UTF-8
bytes, never decoded from hexadecimal or Base64), over the UTF-8 bytes of the receipt's **canonical form**:

* the receipt object without its `mac` member;
* the members of every object in the byte order of their names, at every level (the parts' `etag` before `n`);
* arrays in their own order;
* no whitespace anywhere outside strings: `,` between items and members, `:` between a name and its value;
* every number written as an integer in decimal, with no sign, no leading zero, no fraction and no exponent (a number
  whose value is not an integer has no canonical form, and its receipt does not verify);
* every string as JSON writes it with its characters as themselves, but for these escapes: `"` and `\` as `\"` and
  `\\`; the control characters U+0000 to U+001F as `\b`, `\t`, `\n`, `\f` and `\r` for those five and otherwise as `\u00`
  and two lowercase hexadecimal digits. Nothing else is escaped: not `/`, not U+007F, not U+2028 or U+2029, nor any
  other character outside ASCII. Every string of a valid receipt is printable ASCII, so no control character arises.

In Python this is `json.dumps(receipt_without_mac, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`; in PHP
the members sorted with `ksort(…, SORT_STRING)` at every level and `json_encode(…, JSON_UNESCAPED_SLASHES |
JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_LINE_TERMINATORS)` (without the last flag PHP escapes U+2028 and U+2029, which
Python does not). The Hub compares MACs in constant time, before it reads any ETag of the receipt.

### 5.2 The known answer

`fixtures/receipt-mac-vector.json` holds a key (synthetic), a receipt, its canonical form and its MAC. Each side
computes the canonical form of `receipt` and the MAC under `key`, and must find exactly `canonical` and `mac`, which is
also the receipt's own `mac`.

## 6. The fixture catalog

`fixtures/cases.json` is `{catalog_version: 1, cases: [{id, schema, fixture, purpose, verdict, instance_path}]}`. `id`
is `I`, `O` or `R` and two digits, `schema` the schema file the fixture is checked against, `fixture` its path in the
tree, `verdict` `accept` or `reject`, and `instance_path` null for an accepted fixture and, for a refused one, the RFC
6901 pointer of its one violation: the member that breaks its rule, or the object itself for a member missing or one
the contract does not name (`""` is the document). Every fixture of `fixtures/input/`, `fixtures/output/` and
`fixtures/receipt/` is named by one case, and each side checks every case against its validator.

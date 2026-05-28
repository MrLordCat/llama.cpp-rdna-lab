# D075 - P003 Q4 lossless pack v1 (quality-safe storage prototype)

Date: 2026-05-28  
Owner: Copilot/perf workspace  
Status: keep (storage prototype, no runtime decode route yet)

## Scope

Define and validate a first lossless payload format for selected Q4 tensors.

Target for this stage:

- zero quality loss by construction (exact byte restoration),
- fail-closed artifact contract,
- reproducible pack+verify tooling for next runtime integration stage.

## Implementation

New script:

- `scripts/research/q4_metacomp_lossless_pack.py`

Format v1:

- Sidecar-selected tensor names are used as input scope.
- Each selected Q4 tensor payload is compressed independently (zlib).
- Packed chunks are concatenated into one blob file.
- JSON manifest stores per-entry metadata:
  - tensor name/type,
  - blob offset,
  - raw/packed sizes,
  - CRC32 and SHA256 of raw payload.
- Verification step decompresses every entry and checks exact raw-byte identity.

Fail-closed behavior:

- missing/non-Q4 entries are skipped and reported,
- no default runtime behavior is changed,
- runtime side remains opt-in and can reject unsupported rows.

## Smoke Result (artifact proof)

Run label:

- `q4metacomp-losslesspack-smoke-qwen36-27b-q4ks-r1`

Measured summary (8 selected tensors):

- total raw bytes: `1,066,106,880`
- total packed bytes: `1,045,885,219`
- packed/raw ratio: `0.981032`
- bytes saved: `20,221,661` (`19.28 MiB`)
- verify: PASS (per-entry decompress + crc32 + sha256)

Interpretation:

- This is quality-safe (lossless) storage proof.
- This is not yet a VRAM speed/fit win claim because backend decode route is not implemented.

## Decision

Keep as the correct quality-safe direction.

Next mandatory stage:

1. runtime reader for lossless blob/manifest,
2. exact decode-to-original-Q4 payload at load/compute boundary,
3. measure whether residency can improve without quality loss.

## Artifacts

- `scripts/research/q4_metacomp_lossless_pack.py`
- `build_logs/agent-workload/q4metacomp-losslesspack-smoke-qwen36-27b-q4ks-r1.q4_metacomp_lossless_pack.bin`
- `build_logs/agent-workload/q4metacomp-losslesspack-smoke-qwen36-27b-q4ks-r1.q4_metacomp_lossless_pack.json`
- `build_logs/agent-workload/q4metacomp-losslesspack-smoke-qwen36-27b-q4ks-r1.q4_metacomp_lossless_pack.md`

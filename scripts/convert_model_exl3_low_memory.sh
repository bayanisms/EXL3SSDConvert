#!/usr/bin/env bash
set -Eeuo pipefail

EXPECTED_COMMIT="be57335b087e4f001c5caae061544df3c06ba01e"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
EXLLAMAV3_ROOT="${EXLLAMAV3_ROOT:-$(cd -- "${BUNDLE_DIR}/.." && pwd)}"
LOW_MEMORY_PATCH="${BUNDLE_DIR}/patches/exllamav3-low-memory-quantization.patch"
PATCH_TARGET="exllamav3/modules/quant/exl3_lib/quantize.py"

usage() {
    cat <<'USAGE'
Usage:
  convert_model_exl3_low_memory.sh SOURCE_MODEL_DIR WORK_DIR OUTPUT_DIR RECIPE

Converts a model to EXL3 with disk-backed large temporary buffers. The source,
work, and output paths must not overlap; WORK_DIR and OUTPUT_DIR must not exist.

The ExLlamaV3 checkout defaults to this repository's parent, or set it with
EXLLAMAV3_ROOT. Optional environment variables:
  PYTHON_BIN, DEVICES, LOG_PATH, EXITCODE_PATH
  HEAD_BITS, MTP_BITS, VISION_BITS (optional converter overrides)
  CODEBOOK (default mul1)
  EXL3_QUANT_MMAP_DIR, EXL3_QUANT_MMAP_THRESHOLD
  ALLOW_DIFFERENT_EXLLAMAV3=1
USAGE
}

fail() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

validate_bits() {
    local name="$1"
    local value="$2"
    [[ "$value" =~ ^[0-9]+$ ]] || fail "$name must be an integer"
    (( (value >= 1 && value <= 8) || value == 16 )) || \
        fail "$name must be between 1 and 8, or 16"
}

paths_overlap() {
    local first="${1%/}"
    local second="${2%/}"
    [[ "$first" == "$second" || "$first" == "$second/"* || "$second" == "$first/"* ]]
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi
[[ $# -eq 4 ]] || { usage >&2; exit 2; }

SOURCE_MODEL_DIR="$(realpath -e -- "$1")" || fail "source model directory does not exist: $1"
WORK_DIR="$(realpath -m -- "$2")"
OUTPUT_DIR="$(realpath -m -- "$3")"
RECIPE="$(realpath -e -- "$4")" || fail "recipe does not exist: $4"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICES="${DEVICES:-0}"
HEAD_BITS="${HEAD_BITS:-}"
MTP_BITS="${MTP_BITS:-}"
VISION_BITS="${VISION_BITS:-}"
CODEBOOK="${CODEBOOK:-mul1}"
LOG_PATH="${LOG_PATH:-${OUTPUT_DIR}.conversion.log}"
EXITCODE_PATH="${EXITCODE_PATH:-${OUTPUT_DIR}.conversion.exitcode}"
export EXL3_QUANT_MMAP_DIR="${EXL3_QUANT_MMAP_DIR:-${WORK_DIR}/mmap}"
export EXL3_QUANT_MMAP_THRESHOLD="${EXL3_QUANT_MMAP_THRESHOLD:-1073741824}"

[[ -z "$HEAD_BITS" ]] || validate_bits HEAD_BITS "$HEAD_BITS"
[[ -z "$MTP_BITS" ]] || validate_bits MTP_BITS "$MTP_BITS"
[[ -z "$VISION_BITS" ]] || validate_bits VISION_BITS "$VISION_BITS"
[[ "$CODEBOOK" == "mcg" || "$CODEBOOK" == "mul1" || "$CODEBOOK" == "3inst" ]] || \
    fail "CODEBOOK must be mcg, mul1, or 3inst"
[[ -f "${SOURCE_MODEL_DIR}/config.json" ]] || fail "source config.json is missing"
[[ -f "${SOURCE_MODEL_DIR}/model.safetensors.index.json" || \
   -f "${SOURCE_MODEL_DIR}/model.safetensors" ]] || \
    fail "source model.safetensors or model.safetensors.index.json is missing"
[[ -f "${EXLLAMAV3_ROOT}/convert.py" ]] || fail "convert.py not found at ${EXLLAMAV3_ROOT}"
[[ ! -e "$OUTPUT_DIR" ]] || fail "output path already exists: ${OUTPUT_DIR}"
[[ ! -e "$WORK_DIR" ]] || fail "work path already exists: ${WORK_DIR}"
if paths_overlap "$SOURCE_MODEL_DIR" "$WORK_DIR" || \
   paths_overlap "$SOURCE_MODEL_DIR" "$OUTPUT_DIR" || \
   paths_overlap "$WORK_DIR" "$OUTPUT_DIR"; then
    fail "source, work, and output paths must not overlap"
fi
command -v "$PYTHON_BIN" >/dev/null || fail "Python executable not found: ${PYTHON_BIN}"
[[ -x /usr/bin/time ]] || fail "/usr/bin/time is required for resource reporting"

if command -v git >/dev/null && git -C "$EXLLAMAV3_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    actual_commit="$(git -C "$EXLLAMAV3_ROOT" rev-parse HEAD)"
    if [[ "$actual_commit" != "$EXPECTED_COMMIT" && "${ALLOW_DIFFERENT_EXLLAMAV3:-0}" != "1" ]]; then
        fail "expected ExLlamaV3 ${EXPECTED_COMMIT}, found ${actual_commit}; set ALLOW_DIFFERENT_EXLLAMAV3=1 to override"
    fi
    if ! (cd -- "$EXLLAMAV3_ROOT" && git apply --reverse --check "$LOW_MEMORY_PATCH" >/dev/null 2>&1); then
        fail "the bundled low-memory patch is not applied to this ExLlamaV3 checkout"
    fi
    if [[ "$actual_commit" == "$EXPECTED_COMMIT" && "${ALLOW_DIFFERENT_EXLLAMAV3:-0}" != "1" ]]; then
        if ! cmp -s \
            <(cd -- "$EXLLAMAV3_ROOT" && git diff --no-ext-diff --binary HEAD -- "$PATCH_TARGET") \
            "$LOW_MEMORY_PATCH"; then
            fail "the tracked diff for ${PATCH_TARGET} does not exactly match the bundled patch"
        fi
        while IFS= read -r changed_path; do
            [[ -z "$changed_path" || "$changed_path" == "$PATCH_TARGET" ]] || \
                fail "unexpected tracked converter change: ${changed_path}"
        done < <(cd -- "$EXLLAMAV3_ROOT" && git diff --name-only HEAD --)
        mapfile -t untracked_runtime_files < <(
            cd -- "$EXLLAMAV3_ROOT" && \
                git ls-files --others --exclude-standard -- convert.py exllamav3
        )
        if (( ${#untracked_runtime_files[@]} )); then
            fail "unexpected untracked converter file: ${untracked_runtime_files[0]}"
        fi
    fi
elif [[ "${ALLOW_DIFFERENT_EXLLAMAV3:-0}" != "1" ]]; then
    fail "cannot verify the ExLlamaV3 revision or low-memory patch outside a Git checkout; set ALLOW_DIFFERENT_EXLLAMAV3=1 to override"
fi

mkdir -p -- "$WORK_DIR" "$EXL3_QUANT_MMAP_DIR" "$(dirname -- "$OUTPUT_DIR")" \
    "$(dirname -- "$LOG_PATH")" "$(dirname -- "$EXITCODE_PATH")"

printf 'ExLlamaV3: %s\nSource:     %s\nRecipe:     %s\nWork:       %s\nOutput:     %s\n' \
    "$EXLLAMAV3_ROOT" "$SOURCE_MODEL_DIR" "$RECIPE" "$WORK_DIR" "$OUTPUT_DIR"

CONVERT_ARGS=(
    --in_dir "$SOURCE_MODEL_DIR"
    --work_dir "$WORK_DIR"
    --out_dir "$OUTPUT_DIR"
    --recipe "$RECIPE"
    --codebook "$CODEBOOK"
    --devices "$DEVICES"
)
[[ -z "$HEAD_BITS" ]] || CONVERT_ARGS+=(--head_bits "$HEAD_BITS")
[[ -z "$MTP_BITS" ]] || CONVERT_ARGS+=(--mtp_bits "$MTP_BITS")
[[ -z "$VISION_BITS" ]] || CONVERT_ARGS+=(--vision_bits "$VISION_BITS")

set +e
(
    cd -- "$EXLLAMAV3_ROOT"
    /usr/bin/time -v "$PYTHON_BIN" convert.py "${CONVERT_ARGS[@]}"
) 2>&1 | tee -- "$LOG_PATH"
rc=${PIPESTATUS[0]}
set -e
if [[ "$rc" -eq 0 ]]; then
    if ! validation_output="$("$PYTHON_BIN" - "$OUTPUT_DIR" <<'PY'
import json
from pathlib import Path
import sys

output = Path(sys.argv[1])
try:
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    index = json.loads(
        (output / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    if not isinstance(config, dict):
        raise ValueError("config.json root is not an object")
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model index has no non-empty weight_map")
    shards = set(weight_map.values())
    if not all(isinstance(shard, str) for shard in shards):
        raise ValueError("model index contains a non-string shard name")
    for shard in shards:
        relative = Path(shard)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe shard path in model index: {shard}")
        shard_path = output / relative
        if not shard_path.is_file() or shard_path.stat().st_size == 0:
            raise ValueError(f"missing or empty indexed shard: {shard}")
except (OSError, ValueError, json.JSONDecodeError) as exc:
    print(exc)
    raise SystemExit(1)
PY
)"; then
        printf 'error: converter exited 0 but output validation failed: %s\n' \
            "$validation_output" | tee -a -- "$LOG_PATH" >&2
        rc=1
    fi
fi
printf '%s\n' "$rc" > "$EXITCODE_PATH"
exit "$rc"

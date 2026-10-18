# Low-memory EXL3 model conversion

Convert models to EXL3 when the source model or conversion intermediates do not
fit in GPU memory. This repository provides:

- A patch for disk-backed large CPU buffers and streamed proxy-error measurement.
- A conversion wrapper with explicit quantization settings, resource logging,
  input validation, and output checks.
- A fresh-process probe for model loading, cache allocation, prefill, decode,
  and GPU memory use.
- Two example per-tensor recipes for a custom Qwen 3.8 27B model.

The low-memory path changes temporary storage, not the requested quantization
precision. Recipes are specific to the source model's tensors and weights;
generate a compatible recipe for other models.

## Setup

Conversion requires Linux, Bash 4 or newer, GNU coreutils and GNU time, an NVIDIA
CUDA environment supported by ExLlamaV3, and fast storage with enough space for
the work directory and mmap buffers. Install ExLlamaV3 from commit
`be57335b087e4f001c5caae061544df3c06ba01e`, including its `pyyaml` dependency.

Run these commands from this repository's root, setting the checkout location
for your machine:

```sh
export EXLLAMAV3_ROOT=/path/to/exllamav3
EXL3_CONVERT_ROOT="$PWD"
git -C "$EXLLAMAV3_ROOT" checkout be57335b087e4f001c5caae061544df3c06ba01e
git -C "$EXLLAMAV3_ROOT" apply \
  "$EXL3_CONVERT_ROOT/patches/exllamav3-low-memory-quantization.patch"
```

To check an already applied patch:

```sh
git -C "$EXLLAMAV3_ROOT" apply --reverse --check \
  "$EXL3_CONVERT_ROOT/patches/exllamav3-low-memory-quantization.patch"
```

The wrapper verifies the pinned revision, requires the tracked diff to match the
bundled patch exactly, and rejects other tracked changes and untracked converter
files. `ALLOW_DIFFERENT_EXLLAMAV3=1` bypasses revision and tree verification for
experimental use. In a Git checkout, the wrapper still checks that the patch is
applied. Outside Git, that override also bypasses patch verification.

## Convert a model

Choose a local source model, a compatible recipe, and new work and output paths.
Source, work, and output must not overlap, and work and output must not exist:

```sh
SOURCE=/models/source-model
WORK=/fast-disk/model-exl3-work
OUT=/models/model-exl3
RECIPE=/path/to/model-recipe.yaml

bash scripts/convert_model_exl3_low_memory.sh \
  "$SOURCE" "$WORK" "$OUT" "$RECIPE"
```

To enforce host-memory limits on a systemd machine, use limits suited to your
available RAM. For example:

```sh
systemd-run --user --wait --collect \
  --unit=exl3-low-memory-convert \
  --property=MemoryHigh=22G \
  --property=MemoryMax=24G \
  --property=MemorySwapMax=4G \
  --setenv="EXLLAMAV3_ROOT=$EXLLAMAV3_ROOT" \
  /bin/bash "$PWD/scripts/convert_model_exl3_low_memory.sh" \
  "$SOURCE" "$WORK" "$OUT" "$RECIPE"
```

Direct execution does not enforce those memory limits. The wrapper writes
`<OUT>.conversion.log` and `<OUT>.conversion.exitcode`. Success requires valid
output config and index files, with every indexed shard present and non-empty.

Optional environment settings:

| Variable | Default |
|---|---|
| `PYTHON_BIN` | `python` |
| `DEVICES` | `0` |
| `CODEBOOK` | `mul1` |
| `HEAD_BITS`, `MTP_BITS`, `VISION_BITS` | Converter or recipe defaults |
| `EXL3_QUANT_MMAP_DIR` | `<WORK>/mmap` |
| `EXL3_QUANT_MMAP_THRESHOLD` | `1073741824` bytes |
| `LOG_PATH`, `EXITCODE_PATH` | Paths beside the output directory |

If `EXLLAMAV3_ROOT` is unset, the wrapper looks in this repository's parent
directory. Use `--help` for the command signature.

## Validate inference

A completed conversion does not prove inference works. Run a short probe in the
same Python environment as ExLlamaV3:

```sh
EXL3_QC_STAGING=1 python tools/exl3_fit_probe.py \
  --model "$OUT" --context 8192 --k-bits 4 --v-bits 4 \
  --prefill-tokens 256 --decode-tokens 32 --prefill-chunk 2048 --no-mtp \
  --output-json reports/short-8192.json
```

After each successful probe, repeat in a fresh process with progressively larger
context and prefill sizes. Read the native context from the source `config.json`.
At that capacity, validate a near-full prefill followed by decode. Stop after a
failed gate. `--allocation-only` checks cache initialization without running the
requested inference workload.

## Development checks

```sh
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

Helper, recipe, and shell syntax checks run without a GPU. Conversion-wrapper
tests require Linux. Patch-application checks require `EXLLAMAV3_ROOT` to point
to a compatible checkout; quantization tests require PyTorch and an installed,
patched ExLlamaV3. Tests report skips when these environments are unavailable.

Keep models, conversion work, logs, and probe reports local. The `.gitignore`
covers common generated files, credentials, and the `models/`, `work/`, `output/`,
`reports/`, and `local/` directories. Logs and reports can include absolute paths,
process details, and generated text; review them before sharing.

# EXL3SSDConvert

Convert Hugging Face models to **EXL3** with lower host-memory pressure during
quantization. EXL3SSDConvert combines a small
[ExLlamaV3](https://github.com/turboderp-org/exllamav3) patch, a conversion
wrapper, and an inference check.

Large CPU work buffers can use fast disk-backed storage, and quantization-error
measurement runs in chunks. This trades disk space and I/O for lower host RAM
pressure. You still need enough GPU memory for the converter's active workload;
a successful conversion does not guarantee the finished model will fit for
inference.

[Setup](#setup) · [Convert](#convert-a-model) · [Validate](#validate-the-result) ·
[Settings](#conversion-settings) · [Recipes](#example-recipes) ·
[Troubleshooting](#troubleshooting) · [Contributing](#development-and-contributing)

## What is included?

| Component | Purpose |
|---|---|
| [Low-memory patch](patches/exllamav3-low-memory-quantization.patch) | Disk-backed large CPU buffers and streamed proxy-error measurement |
| [Conversion wrapper](scripts/convert_model_exl3_low_memory.sh) | Input checks, converter verification, resource logging, and output validation |
| [Fit probe](tools/exl3_fit_probe.py) | Model loading, cache allocation, prefill, decode, and GPU memory reporting |
| [Example recipes](examples/custom-qwen-3.8-27b/recipes/) | Two model-specific per-tensor bitrate configurations |
| [Tests](tests/) | Helper tests and environment-dependent integration checks |

The patch changes temporary storage and error measurement, not the requested
quantization bitrate. It targets one upstream revision; this repository does
not bundle model weights or install ExLlamaV3 automatically.

## Requirements

- **Conversion host:** Linux, NVIDIA GPU, Git, Bash 4+, GNU coreutils
  (`realpath`, `tee`, `cmp`), and GNU time at `/usr/bin/time`.
- **Python environment:** Python 3.10.11+, PyTorch 2.6+ with CUDA support, and
  ExLlamaV3 built from the pinned source below. See the
  [pinned upstream installation instructions](https://github.com/turboderp-org/exllamav3/blob/be57335b087e4f001c5caae061544df3c06ba01e/README.md#installation)
  for CUDA and compiler requirements.
- **Source model:** A local, unquantized Hugging Face model directory with
  `config.json`, tokenizer files, and safetensors weights. The architecture must
  be supported by the pinned ExLlamaV3 revision.
- **Recipe:** A per-tensor EXL3 YAML recipe made for that source model.
- **Storage and RAM:** Fast local storage for the source, output, conversion work,
  and mmap buffers, plus enough host RAM for the remaining conversion workload.
  There is no universal disk, RAM, or VRAM minimum.

Conversion requires Linux. Python helper tests also run on macOS and Windows;
shell checks require Bash. The inference probe uses the first visible CUDA GPU.
This project is a separate helper for ExLlamaV3.

## Setup

Use a dedicated ExLlamaV3 checkout so unrelated changes do not interfere with
the wrapper's revision checks.

### 1. Clone both repositories

```sh
git clone https://github.com/bayanisms/EXL3SSDConvert.git
cd EXL3SSDConvert
export EXL3_CONVERT_ROOT="$PWD"

git clone https://github.com/turboderp-org/exllamav3.git ../exllamav3
export EXLLAMAV3_ROOT="$(cd ../exllamav3 && pwd)"
git -C "$EXLLAMAV3_ROOT" checkout be57335b087e4f001c5caae061544df3c06ba01e
```

Keep `EXLLAMAV3_ROOT` exported in the shell used for conversion. If it is unset,
the wrapper searches this repository's parent directory, which is different
from the sibling checkout shown here.

### 2. Prepare the Python environment

From the EXL3SSDConvert directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip "setuptools>=77" wheel ninja
```

Install a CUDA-enabled PyTorch build using the
[official PyTorch selector](https://pytorch.org/get-started/locally/), choosing
versions compatible with your GPU, driver, CUDA toolkit, and the pinned
ExLlamaV3 requirements. Run the selected installation command inside this
environment before continuing.

### 3. Apply the patch and install ExLlamaV3

```sh
git -C "$EXLLAMAV3_ROOT" apply --check \
  "$EXL3_CONVERT_ROOT/patches/exllamav3-low-memory-quantization.patch"
git -C "$EXLLAMAV3_ROOT" apply \
  "$EXL3_CONVERT_ROOT/patches/exllamav3-low-memory-quantization.patch"

python -m pip install --no-build-isolation -e "$EXLLAMAV3_ROOT"
```

If the patch is already applied, skip the two forward-application commands.
Check the installed environment and patch:

```sh
git -C "$EXLLAMAV3_ROOT" apply --reverse --check \
  "$EXL3_CONVERT_ROOT/patches/exllamav3-low-memory-quantization.patch"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.__version__, torch.version.cuda)'
python -c 'import exllamav3; print(exllamav3.__file__)'
```

The wrapper requires the pinned commit plus exactly the bundled tracked patch.
Other tracked changes and untracked files under the converter's runtime paths
are rejected. Avoid modifying the upstream checkout while a conversion runs.

## Convert a model

Set these paths for your own source model and recipe. **Work and output must
not exist.** Source, work, and output cannot be equal or nested within one
another.

```sh
SOURCE=/absolute/path/to/source-model
WORK="$EXL3_CONVERT_ROOT/work/model-exl3"
OUT="$EXL3_CONVERT_ROOT/output/model-exl3"
RECIPE=/absolute/path/to/your-recipe.yaml

PYTHON_BIN="$(command -v python)" \
  bash "$EXL3_CONVERT_ROOT/scripts/convert_model_exl3_low_memory.sh" \
  "$SOURCE" "$WORK" "$OUT" "$RECIPE"
```

For a new shell, reactivate the virtual environment and export the two checkout
paths again. The wrapper prints the selected paths before starting.

A zero exit status means the converter completed and the wrapper verified that
the output config and index parse and every indexed shard is present and
non-empty. It does not verify model quality or inference.

| Output | Contents |
|---|---|
| `$OUT/` | Converted model package |
| `$OUT.conversion.log` | Converter output and GNU time resource statistics |
| `$OUT.conversion.exitcode` | Recorded result of the conversion/output-validation stage |
| `$WORK/` | Conversion intermediates and default mmap directory |

Early input or environment failures can occur before log and exit-code files
are created. Retain the work directory when diagnosing a failure. The wrapper
does not expose a resume option; upstream checkpoint/resume behavior is
documented in the
[pinned conversion guide](https://github.com/turboderp-org/exllamav3/blob/be57335b087e4f001c5caae061544df3c06ba01e/doc/convert.md).

### Optional: enforce host-memory limits

On a machine with user systemd services, this example bounds host RAM and swap.
Choose limits for your hardware; these values are starting examples, not model
requirements. Passing the Python path explicitly keeps the service in the
same environment as the interactive shell.

```sh
systemd-run --user --wait --collect \
  --unit=exl3-low-memory-convert \
  --property=MemoryHigh=22G \
  --property=MemoryMax=24G \
  --property=MemorySwapMax=4G \
  --setenv="EXLLAMAV3_ROOT=$EXLLAMAV3_ROOT" \
  --setenv="PYTHON_BIN=$(command -v python)" \
  /bin/bash "$EXL3_CONVERT_ROOT/scripts/convert_model_exl3_low_memory.sh" \
  "$SOURCE" "$WORK" "$OUT" "$RECIPE"
```

Direct execution does not impose these limits. For additional settings under
`systemd-run`, pass each variable with `--setenv=NAME=value`.

## Validate the result

Run a short prefill/decode probe in a fresh Python process:

```sh
EXL3_QC_STAGING=1 python "$EXL3_CONVERT_ROOT/tools/exl3_fit_probe.py" \
  --model "$OUT" --context 8192 --k-bits 4 --v-bits 4 \
  --prefill-tokens 256 --decode-tokens 32 --prefill-chunk 2048 --no-mtp \
  --output-json "$EXL3_CONVERT_ROOT/reports/short-8192.json"
```

The probe writes a JSON report and returns nonzero on failure. Inspect its
`status`, `error`, `prefill_success`, `decode_success`, and memory measurements.

- `--context` is cache capacity and must be a positive multiple of 256.
- `--k-bits` and `--v-bits` accept integers from 2 through 8.
- Prefill, decode, and chunk counts must be positive; prefill plus decode must
  not exceed context capacity.
- Adding `--allocation-only` checks cache initialization without the inference
  workload. The other required arguments must still be supplied.
- The probe uses logical `cuda:0`. Use `CUDA_VISIBLE_DEVICES` to select a
  physical GPU when needed. `DEVICES` configures conversion, not this probe.
- Runtime is trunk-only; `--no-mtp` makes that explicit. It does not load a
  speculative MTP component.

After each successful run, increase context and prefill sizes in fresh
processes. Read the native context from the source `config.json`; do not infer
it from the examples. At that capacity, test a near-full prefill followed by
decode. Stop at the first failed stage.

This deterministic synthetic workload checks execution and memory fit. It is
not a language-quality benchmark or a guarantee for every application workload.

## Conversion settings

Set these environment variables before invoking the wrapper:

| Variable | Default / behavior |
|---|---|
| `EXLLAMAV3_ROOT` | Upstream checkout; set explicitly |
| `PYTHON_BIN` | `python`; use the executable from your environment |
| `DEVICES` | `0`; converter GPU selection |
| `CODEBOOK` | `mul1`; also accepts `mcg` and `3inst` |
| `HEAD_BITS` | Omitted unless set; uses converter/recipe behavior |
| `MTP_BITS` | Omitted unless set; upstream default is 4 |
| `VISION_BITS` | Omitted unless set; upstream architecture default |
| `EXL3_QUANT_MMAP_DIR` | `<WORK>/mmap`; choose fast storage |
| `EXL3_QUANT_MMAP_THRESHOLD` | `1073741824` bytes (1 GiB) |
| `LOG_PATH` | `<OUT>.conversion.log` |
| `EXITCODE_PATH` | `<OUT>.conversion.exitcode` |
| `ALLOW_DIFFERENT_EXLLAMAV3` | `0`; see override behavior below |

For example, to request explicit component bitrates:

```sh
HEAD_BITS=6 MTP_BITS=4 VISION_BITS=6 \
  bash "$EXL3_CONVERT_ROOT/scripts/convert_model_exl3_low_memory.sh" \
  "$SOURCE" "$WORK" "$OUT" "$RECIPE"
```

The wrapper accepts bitrate overrides of 1–8 or 16; upstream component
restrictions still apply.
`ALLOW_DIFFERENT_EXLLAMAV3=1` bypasses exact revision and tree verification for
experiments. In a Git checkout, patch verification still runs; outside Git,
that override also bypasses patch verification. Prefer the pinned setup when
reproducing a result or reporting a bug.

## Example recipes

**BPW** means bits per weight. The included recipes assign bitrates to 400
tensors in a custom Qwen 3.8 27B source model:

| Recipe | Target decoder BPW | Head bits |
|---|---:|---:|
| [5.0bpw-hb6.yaml](examples/custom-qwen-3.8-27b/recipes/5.0bpw-hb6.yaml) | 5.0 | 6 |
| [5.125bpw-hb6.yaml](examples/custom-qwen-3.8-27b/recipes/5.125bpw-hb6.yaml) | 5.125 | 6 |

These are configuration examples, not universal presets. The exact custom
source weights and sensitivity measurements are not distributed here. Matching
a model-family name alone does not establish recipe compatibility. Generate
a recipe for your own model using the upstream
[self-calibration workflow](https://github.com/turboderp-org/exllamav3/blob/be57335b087e4f001c5caae061544df3c06ba01e/doc/optimize.md).
Recipe generation has separate calibration and memory requirements; this patch
targets conversion. Predicted KLD in a recipe is an optimizer estimate, not a
measured quality score.

## Troubleshooting

| Symptom | What to check |
|---|---|
| `convert.py not found` | Export `EXLLAMAV3_ROOT` to the upstream checkout, not this repository |
| Revision mismatch, patch failure, or unexpected changes | Use the pinned commit in a dedicated checkout; verify the reverse patch check |
| `No module named exllamav3` or CUDA unavailable | Activate the intended environment and repeat the setup checks; confirm `PYTHON_BIN` |
| Existing or overlapping paths | Select new, separate work/output locations; the wrapper does not overwrite or resume them |
| Host process killed / RAM exhausted | Inspect the log and memory limit; disk-backed buffers do not remove all RAM requirements |
| CUDA out of memory | Distinguish conversion from inference; reduce the relevant workload/context or use more suitable hardware |
| Disk full despite few visible mmap files | Mmap files are unlinked while open and can still occupy space; inspect filesystem free space |
| Converter exits but wrapper fails | Check the reported config, index, or missing/empty shard error before attempting inference |

A slower run can be the cost of storage I/O. Use local fast storage and monitor
free space throughout conversion. For inference failures, preserve the probe
JSON and reduce context to find the first failing stage.

## Development and contributing

Install the lightweight test dependency and run discovery from this
repository's root:

```sh
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

Helper and recipe tests do not require a GPU; shell syntax checks require Bash.
Wrapper integration tests require Linux and GNU tools. Patch-application
checks use `EXLLAMAV3_ROOT`; quantization tests require PyTorch and an installed,
patched ExLlamaV3. Unavailable environments are reported as skips.

[Open an issue](https://github.com/bayanisms/EXL3SSDConvert/issues) with the model
architecture, software revisions, RAM/VRAM and free disk space, the command
with private paths replaced, and the first relevant error. For changes, explain
the affected conversion or probe behavior and include the applicable test
results. Report whether you used the revision override.

Keep weights, work files, credentials, and reports local. The
[ignore rules](.gitignore) cover common artifacts and the `models/`, `work/`,
`output/`, `reports/`, and `local/` directories. Review logs and reports before
sharing: they can contain absolute paths, process details, and generated text.

## License

[MIT](LICENSE). ExLlamaV3 is developed upstream; see its repository for its
documentation and licensing.

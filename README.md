# Low-memory EXL3 model conversion

Convert models to EXL3 when the source model or conversion intermediates do not fit in GPU memory. The wrapper combines ExLlamaV3 conversion with disk-backed large temporary buffers, host-memory limits, explicit quantization settings, conversion-result checks, and a fresh-process runtime probe.

This is a general workflow. Recipes remain model-specific because they map exact tensor names to bitrates. The included custom Qwen 3.8 27B model is an example, not the project identity.

## What this adds

- `patches/exllamav3-low-memory-quantization.patch` moves unusually large CPU conversion buffers to unlinked, disk-backed mmap storage and streams proxy-error measurement instead of materializing another full error tensor.
- `scripts/convert_model_exl3_low_memory.sh` validates inputs, exposes conversion settings through environment variables, logs resource use, and rejects false-success or incomplete output.
- `tools/exl3_fit_probe.py` tests model loading, cache allocation, prefill, decode, and GPU memory in a fresh process.
- `examples/` keeps model-specific recipes, checksums, and measured results separate from the general tooling.

The low-memory path changes storage, not the requested quantization precision.

## Requirements

- Linux and an NVIDIA CUDA environment supported by ExLlamaV3.
- An ExLlamaV3 checkout at commit `be57335b087e4f001c5caae061544df3c06ba01e`.
- ExLlamaV3 installed from that checkout, including its `pyyaml` dependency.
- A local source model and a per-tensor EXL3 recipe compatible with that model.
- Fast storage with enough space for the temporary work tree and mmap buffers.

Apply the bundled patch from the ExLlamaV3 project root:

```sh
git checkout be57335b087e4f001c5caae061544df3c06ba01e
git apply github/patches/exllamav3-low-memory-quantization.patch
```

If it is already applied, this must pass:

```sh
git apply --reverse --check github/patches/exllamav3-low-memory-quantization.patch
```

Without an explicit expert override, the wrapper also requires that the tracked diff exactly equals the bundled patch and that no other tracked or untracked converter files alter the pinned checkout. `ALLOW_DIFFERENT_EXLLAMAV3=1` disables exact revision/tree verification but still requires the low-memory patch to be present; results from that mode are not controlled reproductions.

## Run a conversion

Choose three non-overlapping model paths plus a recipe. Source, work, and output must not be equal, nested inside one another, or ancestor paths of one another. The work and output paths must not already exist:

```sh
SOURCE=/models/source-model
WORK=/fast-disk/model-exl3-work
OUT=/models/model-exl3
RECIPE=/path/to/model-recipe.yaml
```

Run with bounded host memory. The example limits are not universal; increase them only when the machine has enough RAM:

```sh
systemd-run --user --wait --collect \
  --unit=exl3-low-memory-convert \
  --property=MemoryHigh=22G \
  --property=MemoryMax=24G \
  --property=MemorySwapMax=4G \
  /bin/bash "$PWD/github/scripts/convert_model_exl3_low_memory.sh" \
  "$SOURCE" "$WORK" "$OUT" "$RECIPE"
```

By default, `HEAD_BITS` comes from the recipe (or ExLlamaV3's default), `MTP_BITS` uses ExLlamaV3's default of 4, and `VISION_BITS` uses the architecture default. The wrapper also defaults to:

```text
CODEBOOK=mul1
DEVICES=0
EXL3_QUANT_MMAP_THRESHOLD=1073741824
```

Override a setting through the environment when the target model requires it. For example:

```sh
HEAD_BITS=6 MTP_BITS=4 VISION_BITS=6 DEVICES=0 \
  /bin/bash github/scripts/convert_model_exl3_low_memory.sh \
  "$SOURCE" "$WORK" "$OUT" "$RECIPE"
```

The wrapper verifies the supported Git revision and patch, writes `<OUT>.conversion.log` and `<OUT>.conversion.exitcode`, and returns success only when the output config and index parse and every indexed shard exists and is non-empty. Direct execution works without systemd, but then `MemoryHigh=22G`, `MemoryMax=24G`, and `MemorySwapMax=4G` are not enforced.

## Validate the output

A successful conversion or model load alone does not prove inference works. Stop after a failed gate:

1. Validate the packaged config, index, and shards.
2. Load the model in a fresh process.
3. Run a short-context prefill and decode.
4. Allocate cache at progressively larger contexts.
5. Execute progressively larger prefills.
6. At the source model's actual native context, run a near-full prefill followed by decode.

Short probe example:

```sh
EXL3_QC_STAGING=1 python github/tools/exl3_fit_probe.py \
  --model "$OUT" --context 8192 --k-bits 4 --v-bits 4 \
  --prefill-tokens 256 --decode-tokens 32 --prefill-chunk 2048 --no-mtp \
  --output-json short-8192.json
```

Read the native context from the actual source `config.json`; do not assume 262,144 for another model. Repeat the probe in fresh processes, increasing `--context` and `--prefill-tokens` only after the prior gate passes. With `--allocation-only`, the probe exits successfully only if the cache reports initialized. Cache allocation alone is still not an operational-context proof.

## Example: custom Qwen 3.8 27B model

Only the decoder recipe changed between the two retained conversions. Both kept 6-bit head, 4-bit MTP, 6-bit vision, `mul1`, Q4/Q4 cache, and the same 262,144-token validation workload.

| Measured item | 5.0 BPW | 5.125 BPW | Change |
|---|---:|---:|---:|
| Achieved decoder BPW | 4.9998 | 5.1250 | +0.1252 |
| Predicted KLD | 0.003001 | 0.002621 | -12.662% |
| Package size | 18.236 GiB | 18.591 GiB | +0.355 GiB |
| Operational peak | 22,415 MiB | 22,735 MiB | +320 MiB |
| Minimum free VRAM | 1,710 MiB | 1,390 MiB | -320 MiB |
| Prefill throughput | 410.190 tok/s | 410.027 tok/s | -0.040% |
| Full-context result | PASS | PASS | unchanged |

What was achieved:

- Both conversions completed a 261,632-token prefill followed by eight decoded tokens at 262,144-token capacity.
- The 5.125 BPW conversion completed in 5,129 seconds with 22.089 GiB maximum host RSS under the documented memory guard.
- The 5.125 BPW artifact contains three shards and 3,080 indexed tensors and retained 1,390 MiB minimum free VRAM on the measured RTX 3090.
- Moving from 5.0 to 5.125 improved optimizer-predicted KLD with essentially unchanged prefill speed, at the cost of 0.355 GiB package size and 320 MiB operational VRAM.
- Use 5.125 for the stronger predicted quantization quality; use 5.0 for the extra 320 MiB deployment margin.

Predicted KLD is an optimizer estimate, not a measured model-quality benchmark. The exact example recipes, measured results, and checksum manifests are under `examples/custom-qwen-3.8-27b/`.

The original experimental checkout contained local tracked changes outside the low-memory patch. The manifests identify the retained source and output artifacts, but this bundle does not claim a rerun will be byte-identical. For new conversions, the wrapper uses the stricter controlled state described above: the pinned commit plus exactly the bundled patch.

## Bundle layout

```text
README.md / readme.me
patches/exllamav3-low-memory-quantization.patch
scripts/convert_model_exl3_low_memory.sh
tools/exl3_fit_probe.py
tests/
examples/custom-qwen-3.8-27b/
  recipes/5.0bpw-hb6.yaml
  recipes/5.125bpw-hb6.yaml
  results/measured-results.json
  results/5.125-output-checksums.sha256
  source/source-checksums.sha256
```

Run the offline checks with:

```sh
python -m unittest \
  github/tests/test_reproduction_bundle.py \
  github/tests/test_quantize_low_mem.py \
  github/tests/test_exl3_fit_probe.py
```

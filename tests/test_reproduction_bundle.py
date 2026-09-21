import json
import os
import re
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest

import yaml


BUNDLE = Path(__file__).resolve().parents[1]
PROJECT = BUNDLE.parent
EXAMPLE = BUNDLE / "examples" / "custom-qwen-3.8-27b"


class ReproductionBundleTest(unittest.TestCase):
    def test_required_files_are_present(self):
        required = [
            "README.md",
            "readme.me",
            "patches/exllamav3-low-memory-quantization.patch",
            "scripts/convert_model_exl3_low_memory.sh",
            "tools/exl3_fit_probe.py",
            "tests/test_quantize_low_mem.py",
            "tests/test_exl3_fit_probe.py",
            "examples/custom-qwen-3.8-27b/recipes/5.0bpw-hb6.yaml",
            "examples/custom-qwen-3.8-27b/recipes/5.125bpw-hb6.yaml",
            "examples/custom-qwen-3.8-27b/results/measured-results.json",
            "examples/custom-qwen-3.8-27b/results/5.125-output-checksums.sha256",
            "examples/custom-qwen-3.8-27b/source/source-checksums.sha256",
        ]
        missing = [relative for relative in required if not (BUNDLE / relative).is_file()]
        self.assertEqual(missing, [])

    def test_checksum_manifests_cover_source_and_selected_output(self):
        manifests = {
            "examples/custom-qwen-3.8-27b/source/source-checksums.sha256": 26,
            "examples/custom-qwen-3.8-27b/results/5.125-output-checksums.sha256": 17,
        }
        for relative, expected_count in manifests.items():
            with self.subTest(manifest=relative):
                lines = (BUNDLE / relative).read_text().splitlines()
                self.assertEqual(len(lines), expected_count)
                for line in lines:
                    self.assertRegex(line, r"^[0-9a-f]{64}  [^/].*$")
                    self.assertNotIn("..", line)
        example = BUNDLE / "examples" / "custom-qwen-3.8-27b"
        source_manifest = (example / "source" / "source-checksums.sha256").read_text()
        self.assertIn("  config.json\n", source_manifest)
        self.assertIn("  model.safetensors.index.json\n", source_manifest)
        self.assertIn("  tokenizer.json\n", source_manifest)
        output_manifest = (example / "results" / "5.125-output-checksums.sha256").read_text()
        self.assertIn("  tokenizer.json\n", output_manifest)
        self.assertIn("  quantization_config.json\n", output_manifest)

    def test_retained_models_record_the_identical_full_context_workload(self):
        results_path = (
            BUNDLE
            / "examples"
            / "custom-qwen-3.8-27b"
            / "results"
            / "measured-results.json"
        )
        results = json.loads(results_path.read_text())
        keys = [
            "full_context_result",
            "context_capacity",
            "prefill_tokens",
            "decode_tokens",
            "cache",
            "prefill_chunk",
            "qc_staging",
            "optional_speculative_mtp_loaded",
        ]
        five = results["models"]["5.0"]
        five_125 = results["models"]["5.125"]
        self.assertEqual({key: five[key] for key in keys}, {key: five_125[key] for key in keys})

    def test_conversion_script_is_portable_and_has_help(self):
        script = BUNDLE / "scripts" / "convert_model_exl3_low_memory.sh"
        syntax = subprocess.run(
            ["bash", "-n", str(script)], capture_output=True, text=True, check=False
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        help_run = subprocess.run(
            ["bash", str(script), "--help"], capture_output=True, text=True, check=False
        )
        self.assertEqual(help_run.returncode, 0, help_run.stderr)
        self.assertIn("SOURCE_MODEL_DIR", help_run.stdout)
        self.assertIn("RECIPE", help_run.stdout)
        text = script.read_text()
        self.assertNotIn("/example/home", text)
        self.assertNotIn("/example/storage", text)
        self.assertNotIn("qwen", text.lower())
        self.assertIn("EXL3_QUANT_MMAP_DIR", text)
        self.assertIn('HEAD_BITS="${HEAD_BITS:-}"', text)
        self.assertIn('MTP_BITS="${MTP_BITS:-}"', text)
        self.assertIn('VISION_BITS="${VISION_BITS:-}"', text)
        self.assertIn('CONVERT_ARGS+=(--head_bits "$HEAD_BITS")', text)
        self.assertIn('CONVERT_ARGS+=(--mtp_bits "$MTP_BITS")', text)
        self.assertIn('CONVERT_ARGS+=(--vision_bits "$VISION_BITS")', text)
        self.assertIn('CODEBOOK="${CODEBOOK:-mul1}"', text)
        self.assertIn("git apply --reverse --check", text)
        self.assertIn("git diff --no-ext-diff --binary HEAD", text)
        self.assertIn("cmp -s", text)
        self.assertIn("git ls-files --others --exclude-standard", text)

    def test_conversion_script_rejects_nested_source_work_and_output_paths(self):
        script = BUNDLE / "scripts" / "convert_model_exl3_low_memory.sh"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = root / "exllamav3"
            source = root / "source"
            nested_work = source / "work"
            output = root / "output"
            project.mkdir()
            source.mkdir()
            (source / "config.json").write_text("{}\n")
            (source / "model.safetensors").write_bytes(b"source-shard")
            (project / "convert.py").write_text("raise RuntimeError('must not run')\n")
            environment = os.environ.copy()
            environment.update(
                EXLLAMAV3_ROOT=str(project),
                PYTHON_BIN=sys.executable,
                ALLOW_DIFFERENT_EXLLAMAV3="1",
            )

            result = subprocess.run(
                [
                    "bash",
                    str(script),
                    str(source),
                    str(nested_work),
                    str(output),
                    str(EXAMPLE / "recipes" / "5.125bpw-hb6.yaml"),
                ],
                capture_output=True,
                text=True,
                env=environment,
                check=False,
            )

            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("must not overlap", result.stderr)
            self.assertFalse(nested_work.exists())
            self.assertFalse(output.exists())

    def test_conversion_script_fails_if_converter_produces_no_artifact(self):
        script = BUNDLE / "scripts" / "convert_model_exl3_low_memory.sh"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = root / "exllamav3"
            source = root / "source"
            work = root / "work"
            output = root / "output"
            project.mkdir()
            source.mkdir()
            (source / "config.json").write_text("{}\n")
            (source / "model.safetensors.index.json").write_text('{"weight_map": {}}\n')
            (project / "convert.py").write_text(
                'print(" !! Error: synthetic converter preparation failure")\n'
            )
            environment = os.environ.copy()
            environment.update(
                EXLLAMAV3_ROOT=str(project),
                PYTHON_BIN=sys.executable,
                ALLOW_DIFFERENT_EXLLAMAV3="1",
            )

            result = subprocess.run(
                [
                    "bash",
                    str(script),
                    str(source),
                    str(work),
                    str(output),
                    str(EXAMPLE / "recipes" / "5.125bpw-hb6.yaml"),
                ],
                capture_output=True,
                text=True,
                env=environment,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((root / "output.conversion.exitcode").read_text().strip(), "1")
            self.assertFalse(output.exists())

    def test_conversion_script_fails_if_output_index_is_malformed(self):
        script = BUNDLE / "scripts" / "convert_model_exl3_low_memory.sh"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = root / "exllamav3"
            source = root / "source"
            work = root / "work"
            output = root / "output"
            project.mkdir()
            source.mkdir()
            (source / "config.json").write_text("{}\n")
            (source / "model.safetensors.index.json").write_text('{"weight_map": {}}\n')
            (project / "convert.py").write_text(
                "import sys\n"
                "from pathlib import Path\n"
                "out = Path(sys.argv[sys.argv.index('--out_dir') + 1])\n"
                "out.mkdir()\n"
                "(out / 'model.safetensors.index.json').write_text('not-json\\n')\n"
            )
            environment = os.environ.copy()
            environment.update(
                EXLLAMAV3_ROOT=str(project),
                PYTHON_BIN=sys.executable,
                ALLOW_DIFFERENT_EXLLAMAV3="1",
            )

            result = subprocess.run(
                [
                    "bash",
                    str(script),
                    str(source),
                    str(work),
                    str(output),
                    str(EXAMPLE / "recipes" / "5.125bpw-hb6.yaml"),
                ],
                capture_output=True,
                text=True,
                env=environment,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((root / "output.conversion.exitcode").read_text().strip(), "1")

    def test_conversion_script_accepts_a_minimally_valid_artifact(self):
        script = BUNDLE / "scripts" / "convert_model_exl3_low_memory.sh"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = root / "exllamav3"
            source = root / "source"
            work = root / "work"
            output = root / "output"
            project.mkdir()
            source.mkdir()
            (source / "config.json").write_text("{}\n")
            (source / "model.safetensors").write_bytes(b"source-shard")
            (project / "convert.py").write_text(
                "import json, sys\n"
                "from pathlib import Path\n"
                "out = Path(sys.argv[sys.argv.index('--out_dir') + 1])\n"
                "out.mkdir()\n"
                "(out / 'config.json').write_text('{}\\n')\n"
                "(out / 'model-00001-of-00001.safetensors').write_bytes(b'shard')\n"
                "(out / 'model.safetensors.index.json').write_text("
                "json.dumps({'weight_map': {'tensor': 'model-00001-of-00001.safetensors'}}))\n"
            )
            environment = os.environ.copy()
            environment.update(
                EXLLAMAV3_ROOT=str(project),
                PYTHON_BIN=sys.executable,
                ALLOW_DIFFERENT_EXLLAMAV3="1",
            )

            result = subprocess.run(
                [
                    "bash",
                    str(script),
                    str(source),
                    str(work),
                    str(output),
                    str(EXAMPLE / "recipes" / "5.125bpw-hb6.yaml"),
                ],
                capture_output=True,
                text=True,
                env=environment,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((root / "output.conversion.exitcode").read_text().strip(), "0")

    def test_recipes_are_portable_complete_and_match_recorded_targets(self):
        expected = {
            "5.0bpw-hb6.yaml": (5.0, 4.9998, 0.003001),
            "5.125bpw-hb6.yaml": (5.125, 5.125, 0.002621),
        }
        for name, values in expected.items():
            with self.subTest(recipe=name):
                recipe = yaml.safe_load((EXAMPLE / "recipes" / name).read_text())
                self.assertEqual(
                    (recipe["target_bpw"], recipe["achieved_bpw"], recipe["predicted_kld"]),
                    values,
                )
                self.assertEqual(recipe["head_bits"], 6)
                self.assertEqual(len(recipe["tensors"]), 400)
                self.assertNotIn("/example/home", recipe["model"])
                self.assertNotIn("/example/storage", recipe["model"])
                self.assertNotIn("/example/storage", recipe["measurement"])

    def test_patch_matches_either_clean_or_already_patched_checkout(self):
        patch = BUNDLE / "patches" / "exllamav3-low-memory-quantization.patch"
        normal = subprocess.run(
            ["git", "apply", "--check", str(patch)],
            cwd=PROJECT,
            capture_output=True,
            text=True,
            check=False,
        )
        reverse = subprocess.run(
            ["git", "apply", "--reverse", "--check", str(patch)],
            cwd=PROJECT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertTrue(
            normal.returncode == 0 or reverse.returncode == 0,
            f"forward:\n{normal.stderr}\nreverse:\n{reverse.stderr}",
        )
        text = patch.read_text()
        self.assertIn("EXL3_QUANT_MMAP_DIR", text)
        self.assertIn("block_trace_pair_stream", text)

    def test_readme_documents_goal_method_results_and_reproduction(self):
        readme = (BUNDLE / "README.md").read_text()
        for required_text in [
            "Convert models to EXL3 when",
            "custom Qwen 3.8 27B model",
            "262,144",
            "5.125 BPW",
            "5.0 BPW",
            "MemoryHigh=22G",
            "MemoryMax=24G",
            "MemorySwapMax=4G",
            "261,632",
            "22,735 MiB",
            "1,390 MiB",
            "systemd-run --user",
            "convert_model_exl3_low_memory.sh",
            "exl3_fit_probe.py",
        ]:
            self.assertIn(required_text, readme)
        self.assertNotIn("Fable-Cold-Fusion", readme)
        self.assertNotIn("Qwen3_5ForConditionalGeneration", readme)
        self.assertNotIn("5.25", readme)
        self.assertEqual(readme, (BUNDLE / "readme.me").read_text())


if __name__ == "__main__":
    unittest.main()

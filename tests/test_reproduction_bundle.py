import os
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest

import yaml


BUNDLE = Path(__file__).resolve().parents[1]
EXAMPLE = BUNDLE / "examples" / "custom-qwen-3.8-27b"


class ConversionBundleTest(unittest.TestCase):
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

    @unittest.skipUnless(
        sys.platform == "linux", "conversion wrapper requires Linux and GNU tools"
    )
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

    @unittest.skipUnless(
        sys.platform == "linux", "conversion wrapper requires Linux and GNU tools"
    )
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

    @unittest.skipUnless(
        sys.platform == "linux", "conversion wrapper requires Linux and GNU tools"
    )
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

    @unittest.skipUnless(
        sys.platform == "linux", "conversion wrapper requires Linux and GNU tools"
    )
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
                self.assertFalse(Path(recipe["model"]).is_absolute())
                self.assertFalse(Path(recipe["measurement"]).is_absolute())

    def test_patch_matches_either_clean_or_already_patched_checkout(self):
        checkout = os.environ.get("EXLLAMAV3_ROOT")
        if not checkout:
            self.skipTest("set EXLLAMAV3_ROOT to check patch application")
        project = Path(checkout).expanduser().resolve()
        self.assertTrue(
            (project / "convert.py").is_file(), "invalid ExLlamaV3 checkout"
        )
        patch = BUNDLE / "patches" / "exllamav3-low-memory-quantization.patch"
        normal = subprocess.run(
            ["git", "apply", "--check", str(patch)],
            cwd=project,
            capture_output=True,
            text=True,
            check=False,
        )
        reverse = subprocess.run(
            ["git", "apply", "--reverse", "--check", str(patch)],
            cwd=project,
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


if __name__ == "__main__":
    unittest.main()

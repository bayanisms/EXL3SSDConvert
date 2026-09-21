import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


PROBE_PATH = Path(__file__).parents[1] / "tools" / "exl3_fit_probe.py"


def load_probe():
    spec = importlib.util.spec_from_file_location("exl3_fit_probe", PROBE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load probe module at {PROBE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_argv(output_path):
    return [
        "--model", "/models/example",
        "--context", "1024",
        "--k-bits", "4",
        "--v-bits", "4",
        "--prefill-tokens", "768",
        "--decode-tokens", "32",
        "--prefill-chunk", "256",
        "--output-json", str(output_path),
    ]


class FitProbeHelpersTest(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)

    def test_parse_args_accepts_required_probe_configuration(self):
        probe = load_probe()

        args = probe.parse_args(valid_argv(self.root / "probe.json"))

        self.assertEqual(args.context, 1024)
        self.assertEqual(args.k_bits, 4)
        self.assertEqual(args.v_bits, 4)
        self.assertFalse(args.allocation_only)
        self.assertFalse(args.no_mtp)

    def test_parse_args_rejects_invalid_counts_and_cache_geometry(self):
        probe = load_probe()
        invalid = [
            ("--context", "0"),
            ("--context", "1000"),
            ("--k-bits", "1"),
            ("--v-bits", "9"),
            ("--prefill-tokens", "0"),
            ("--decode-tokens", "0"),
            ("--prefill-chunk", "0"),
        ]
        for option, value in invalid:
            with self.subTest(option=option, value=value):
                argv = valid_argv(self.root / "probe.json")
                argv[argv.index(option) + 1] = value
                with self.assertRaises(SystemExit):
                    probe.parse_args(argv)

    def test_parse_args_rejects_sequence_larger_than_cache(self):
        probe = load_probe()
        argv = valid_argv(self.root / "probe.json")
        argv[argv.index("--prefill-tokens") + 1] = "1000"
        argv[argv.index("--decode-tokens") + 1] = "25"

        with self.assertRaises(SystemExit):
            probe.parse_args(argv)

    def test_nvidia_selector_tracks_logical_cuda_zero_remapping(self):
        probe = load_probe()

        self.assertEqual(probe.nvidia_device_selector({}), "0")
        self.assertEqual(
            probe.nvidia_device_selector({"CUDA_VISIBLE_DEVICES": "3,1"}), "3"
        )
        self.assertEqual(
            probe.nvidia_device_selector({"CUDA_VISIBLE_DEVICES": "GPU-deadbeef,2"}),
            "GPU-deadbeef",
        )

    def test_nvidia_snapshot_queries_the_mapped_physical_device(self):
        probe = load_probe()
        responses = [
            subprocess.CompletedProcess([], 0, "24576, 10, 24566, 0\n", ""),
            subprocess.CompletedProcess([], 0, "1234, 10\n", ""),
        ]
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,0"}):
            with mock.patch.object(probe.subprocess, "run", side_effect=responses) as run:
                snapshot = probe.nvidia_snapshot()

        self.assertTrue(snapshot["available"])
        self.assertEqual(snapshot["device_selector"], "2")
        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            self.assertIn("--id=2", call.args[0])

    def test_tile_token_ids_is_deterministic_and_exact(self):
        probe = load_probe()

        first = probe.tile_token_ids([11, 12, 13], 8)
        second = probe.tile_token_ids([11, 12, 13], 8)

        self.assertEqual(first, second)
        self.assertEqual(first, [11, 12, 13, 11, 12, 13, 11, 12])
        self.assertEqual(probe.tile_token_ids([11, 12], 1), [11])

    def test_tile_token_ids_rejects_empty_seed(self):
        probe = load_probe()

        with self.assertRaisesRegex(ValueError, "at least one token"):
            probe.tile_token_ids([], 4)

    def test_atomic_write_json_replaces_target_without_temp_residue(self):
        probe = load_probe()
        target = self.root / "nested" / "probe.json"

        probe.atomic_write_json(target, {"status": "ok", "value": 7})

        self.assertEqual(json.loads(target.read_text()), {"status": "ok", "value": 7})
        self.assertEqual(list(target.parent.glob(f".{target.name}.*.tmp")), [])

    def test_allocation_only_requires_an_initialized_cache(self):
        probe = load_probe()
        report = {}

        with self.assertRaisesRegex(RuntimeError, "cache allocation did not complete"):
            probe.finalize_allocation_only(report, SimpleNamespace(initialized=False))

        self.assertTrue(report["model_loaded"])
        self.assertFalse(report["cache_allocated"])
        self.assertFalse(report["allocation_success"])
        self.assertNotEqual(report.get("status"), "ok")

    def test_allocation_only_records_success_for_an_initialized_cache(self):
        probe = load_probe()
        report = {}

        probe.finalize_allocation_only(report, SimpleNamespace(initialized=True))

        self.assertTrue(report["model_loaded"])
        self.assertTrue(report["cache_allocated"])
        self.assertTrue(report["allocation_success"])
        self.assertEqual(report["status"], "ok")

    def test_execute_with_report_writes_exception_details(self):
        probe = load_probe()
        output = self.root / "failure.json"
        args = probe.parse_args(valid_argv(output))

        def fail(_args, _report):
            raise RuntimeError("offline synthetic failure")

        exit_code = probe.execute_with_report(args, fail)
        report = json.loads(output.read_text())

        self.assertEqual(exit_code, 1)
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["error"]["type"], "RuntimeError")
        self.assertEqual(report["error"]["message"], "offline synthetic failure")
        self.assertIn("RuntimeError: offline synthetic failure", report["error"]["traceback"])
        self.assertEqual(report["config"]["context"], 1024)


if __name__ == "__main__":
    unittest.main()

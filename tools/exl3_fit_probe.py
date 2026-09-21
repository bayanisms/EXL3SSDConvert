#!/usr/bin/env python3
"""Fresh-process EXL3 load, cache, prefill, and decode fit probe."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

PAGE_SIZE = 256
SEED_TEXT = "The quick brown fox jumps over the lazy dog. EXL3 context validation. "


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def cache_bits(value: str) -> int:
    parsed = int(value)
    if not 2 <= parsed <= 8:
        raise argparse.ArgumentTypeError("must be between 2 and 8")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--context", type=positive_int, required=True)
    parser.add_argument("--k-bits", type=cache_bits, required=True)
    parser.add_argument("--v-bits", type=cache_bits, required=True)
    parser.add_argument("--prefill-tokens", type=positive_int, required=True)
    parser.add_argument("--decode-tokens", type=positive_int, required=True)
    parser.add_argument("--prefill-chunk", type=positive_int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--allocation-only", action="store_true")
    parser.add_argument(
        "--no-mtp",
        action="store_true",
        help="Explicitly use trunk-only runtime (the default; MTP is not attached implicitly)",
    )
    args = parser.parse_args(argv)
    if args.context % PAGE_SIZE:
        parser.error(f"--context must be a multiple of {PAGE_SIZE}")
    if args.prefill_tokens + args.decode_tokens > args.context:
        parser.error("--prefill-tokens + --decode-tokens must not exceed --context")
    return args


def tile_token_ids(seed: list[int], length: int) -> list[int]:
    if not seed:
        raise ValueError("seed must contain at least one token")
    if length < 0:
        raise ValueError("length must not be negative")
    repeats = math.ceil(length / len(seed)) if length else 0
    return (seed * repeats)[:length]


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def nvidia_device_selector(environment: dict[str, str] | None = None) -> str:
    """Map Torch logical cuda:0 to the physical selector used by nvidia-smi."""
    env = os.environ if environment is None else environment
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return "0"
    first = visible.split(",", 1)[0].strip()
    return first or "0"


def nvidia_snapshot() -> dict[str, Any]:
    selector = nvidia_device_selector()
    result: dict[str, Any] = {"available": False, "device_selector": selector}
    try:
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
                f"--id={selector}",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip().split(",")
        processes = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
                f"--id={selector}",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip().splitlines()
        process_memory = {}
        for line in processes:
            if not line.strip():
                continue
            pid, used = (part.strip() for part in line.split(",", 1))
            process_memory[pid] = int(used)
        result = {
            "available": True,
            "device_selector": selector,
            "total_mib": int(gpu[0].strip()),
            "used_mib": int(gpu[1].strip()),
            "free_mib": int(gpu[2].strip()),
            "utilization_percent": int(gpu[3].strip()),
            "process_used_mib": process_memory.get(str(os.getpid())),
            "compute_processes_mib": process_memory,
        }
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


class DriverSampler:
    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self.samples = 0
        self.peak_used_mib: int | None = None
        self.minimum_free_mib: int | None = None
        self.peak_process_used_mib: int | None = None
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="nvidia-smi-sampler", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join(timeout=15)
        return self.as_dict()

    def _run(self) -> None:
        while not self._stop.is_set():
            snap = nvidia_snapshot()
            if snap.get("available"):
                used = snap["used_mib"]
                free = snap["free_mib"]
                process_used = snap.get("process_used_mib")
                self.peak_used_mib = used if self.peak_used_mib is None else max(self.peak_used_mib, used)
                self.minimum_free_mib = free if self.minimum_free_mib is None else min(self.minimum_free_mib, free)
                if process_used is not None:
                    self.peak_process_used_mib = (
                        process_used
                        if self.peak_process_used_mib is None
                        else max(self.peak_process_used_mib, process_used)
                    )
                self.samples += 1
            elif snap.get("error"):
                self.error = snap["error"]
            self._stop.wait(self.interval)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_interval_seconds": self.interval,
            "samples": self.samples,
            "peak_device_used_mib": self.peak_used_mib,
            "minimum_device_free_mib": self.minimum_free_mib,
            "peak_process_used_mib": self.peak_process_used_mib,
            "error": self.error,
        }


def torch_snapshot(torch: Any, label: str) -> dict[str, Any]:
    torch.cuda.synchronize(0)
    free, total = torch.cuda.mem_get_info(0)
    return {
        "label": label,
        "timestamp": time.time(),
        "torch_free_bytes": free,
        "torch_total_bytes": total,
        "torch_allocated_bytes": torch.cuda.memory_allocated(0),
        "torch_reserved_bytes": torch.cuda.memory_reserved(0),
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
        "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
        "driver": nvidia_snapshot(),
    }


def reset_peaks(torch: Any) -> None:
    torch.cuda.synchronize(0)
    torch.cuda.reset_peak_memory_stats(0)


def cache_accounting(cache: Any) -> dict[str, Any]:
    layers = list(cache.layers.values())
    recurrent_layers = list(cache.get_all_recurrent_layers().values())
    return {
        "kv_layer_count": len(layers),
        "kv_storage_bytes": sum(layer.storage_size() for layer in layers),
        "recurrent_layer_count": len(recurrent_layers),
        "recurrent_storage_bytes": sum(layer.storage_size() for layer in recurrent_layers),
        "initialized": bool(cache.initialized),
    }


def serializable_vram_accounting(model: Any, cache: Any, generator: Any) -> list[dict[str, Any]]:
    from exllamav3.util.memory import vram_accounting

    return [item.as_dict() for item in vram_accounting(model, cache, generator)]


def run_job(
    torch: Any,
    generator: Any,
    input_ids: Any,
    decode_tokens: int,
    label: str,
    report: dict[str, Any],
) -> dict[str, Any]:
    from exllamav3 import ArgmaxSampler, Job

    reset_peaks(torch)
    started = time.perf_counter()
    job = Job(
        input_ids=input_ids,
        max_new_tokens=decode_tokens,
        min_new_tokens=decode_tokens,
        sampler=ArgmaxSampler(),
        decode_special_tokens=True,
        identifier=label,
    )
    generator.enqueue(job)
    generated_ids: list[int] = []
    generated_text: list[str] = []
    eos = False
    eos_reason = None
    prefill_progress = 0
    prefill_target = 0
    first_stream_seen = False
    prefill_checkpoint = None
    decode_checkpoint = None
    final_event: dict[str, Any] | None = None
    while generator.num_remaining_jobs():
        events = generator.iterate()
        for event in events:
            stage = event.get("stage")
            if stage == "prefill":
                prefill_progress = int(event.get("curr_progress", prefill_progress))
                prefill_target = int(event.get("max_progress", prefill_target))
            elif stage == "streaming":
                if not first_stream_seen:
                    prefill_checkpoint = torch_snapshot(torch, f"{label}_prefill_complete_first_stream")
                    reset_peaks(torch)
                    first_stream_seen = True
                token_ids = event.get("token_ids")
                if token_ids is not None:
                    generated_ids.extend(int(value) for value in token_ids.view(-1).tolist())
                text = event.get("text")
                if text:
                    generated_text.append(text)
                if event.get("eos"):
                    eos = True
                    eos_reason = event.get("eos_reason")
                    final_event = event
    torch.cuda.synchronize(0)
    decode_checkpoint = torch_snapshot(torch, f"{label}_decode_complete")
    elapsed = time.perf_counter() - started
    if final_event:
        full_completion = final_event.get("full_completion", "".join(generated_text))
        time_prefill = final_event.get("time_prefill")
        time_generate = final_event.get("time_generate")
    else:
        full_completion = "".join(generated_text)
        time_prefill = None
        time_generate = None
    result = {
        "label": label,
        "input_tokens": int(input_ids.shape[-1]),
        "requested_decode_tokens": decode_tokens,
        "generated_token_ids": generated_ids,
        "generated_text": full_completion,
        "generated_tokens": len(generated_ids),
        "eos": eos,
        "eos_reason": eos_reason,
        "prefill_progress": prefill_progress,
        "prefill_target": prefill_target,
        "elapsed_seconds": elapsed,
        "reported_prefill_seconds": time_prefill,
        "reported_generate_seconds": time_generate,
        "prefill_checkpoint": prefill_checkpoint,
        "decode_checkpoint": decode_checkpoint,
    }
    report.setdefault("jobs", []).append(result)
    return result


def finalize_allocation_only(report: dict[str, Any], cache: Any) -> None:
    """Record allocation-only success only after cache initialization."""
    allocated = bool(cache.initialized)
    report["model_loaded"] = True
    report["cache_allocated"] = allocated
    report["allocation_success"] = allocated
    if not allocated:
        raise RuntimeError("cache allocation did not complete")
    report["status"] = "ok"


def run_probe(args: argparse.Namespace, report: dict[str, Any]) -> None:
    report["checkpoints"]["P0_process_start"] = {"driver": nvidia_snapshot(), "timestamp": time.time()}
    sampler = DriverSampler()
    sampler.start()
    report["driver_sampler"] = {"status": "running"}
    try:
        import torch
        from exllamav3 import Cache, CacheLayer_quant, Config, Generator, Model, Tokenizer

        torch.cuda.init()
        report["environment"].update(
            {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torch_cuda_runtime": torch.version.cuda,
                "cuda_device": torch.cuda.get_device_name(0),
                "cuda_capability": list(torch.cuda.get_device_capability(0)),
                "exl3_qc_staging": os.environ.get("EXL3_QC_STAGING", "1 (local default)"),
            }
        )
        report["checkpoints"]["P1_cuda_initialized"] = torch_snapshot(torch, "P1_cuda_initialized")

        config = Config.from_directory(str(args.model))
        model = Model.from_config(config)
        cache = Cache(
            model,
            max_num_tokens=args.context,
            layer_type=CacheLayer_quant,
            k_bits=args.k_bits,
            v_bits=args.v_bits,
            max_batch_size=1,
        )
        report["model"] = {
            "architecture": config.architecture,
            "max_context": getattr(config, "max_seq_len", None),
            "mtp_component_available": "mtp" in config.model_classes,
            "mtp_loaded": False,
            "mtp_note": "MTP is a separately attached draft component; normal trunk runtime does not load it",
        }
        report["checkpoints"]["P2_objects_created"] = torch_snapshot(torch, "P2_objects_created")

        reset_peaks(torch)
        load_started = time.perf_counter()
        model.load(
            device="cuda:0",
            progressbar=True,
            max_chunk_size=args.prefill_chunk,
            autosplit_no_forward=args.allocation_only,
        )
        torch.cuda.synchronize(0)
        report["timings"]["model_load_seconds"] = time.perf_counter() - load_started
        report["checkpoints"]["P3_model_weights_loaded"] = torch_snapshot(torch, "P3_model_weights_loaded")
        report["cache"] = cache_accounting(cache)
        report["cache"].update(
            {
                "capacity_tokens": args.context,
                "page_size": PAGE_SIZE,
                "page_count": args.context // PAGE_SIZE,
                "k_bits": args.k_bits,
                "v_bits": args.v_bits,
                "allocation_note": "Cache is attached before model.load and allocated during load; P3/P4 are the same physical point",
            }
        )
        report["checkpoints"]["P4_cache_allocated"] = torch_snapshot(torch, "P4_cache_allocated")
        report["vram_accounting_after_load"] = serializable_vram_accounting(model, cache, None)

        if args.allocation_only:
            finalize_allocation_only(report, cache)
            return

        tokenizer = Tokenizer.from_config(config)
        generator = Generator(
            model=model,
            cache=cache,
            tokenizer=tokenizer,
            max_batch_size=1,
            max_chunk_size=args.prefill_chunk,
            cpu_cache_size=0,
        )
        report["checkpoints"]["P5_generator_initialized"] = torch_snapshot(torch, "P5_generator_initialized")
        report["model_loaded"] = True
        report["cache_allocated"] = bool(cache.initialized)

        seed_tensor = tokenizer.encode(SEED_TEXT, add_bos=True, encode_special_tokens=True)
        if isinstance(seed_tensor, tuple):
            seed_tensor = seed_tensor[0]
        seed_ids = [int(value) for value in seed_tensor.view(-1).tolist()]
        vocab_size = int(getattr(tokenizer, "actual_vocab_size", 0))
        if not seed_ids or any(value < 0 or value >= vocab_size for value in seed_ids):
            raise ValueError("tokenizer produced an invalid deterministic seed")
        report["input_pattern"] = {
            "text": SEED_TEXT,
            "seed_token_ids": seed_ids,
            "seed_token_count": len(seed_ids),
            "vocab_size": vocab_size,
        }

        warm_tokens = min(PAGE_SIZE, args.prefill_tokens)
        warm_ids = torch.tensor([tile_token_ids(seed_ids, warm_tokens)], dtype=torch.long, device="cpu")
        warm = run_job(torch, generator, warm_ids, 1, "warmup", report)
        report["checkpoints"]["P6_first_short_prefill_completed"] = warm["prefill_checkpoint"]
        report["checkpoints"]["P7_first_decode_completed"] = warm["decode_checkpoint"]

        long_ids = torch.tensor(
            [tile_token_ids(seed_ids, args.prefill_tokens)], dtype=torch.long, device="cpu"
        )
        result = run_job(torch, generator, long_ids, args.decode_tokens, "requested", report)
        report["checkpoints"]["P8_requested_prefill_completed"] = result["prefill_checkpoint"]
        report["checkpoints"]["P9_decode_after_requested_prefill"] = result["decode_checkpoint"]
        report["prefill_success"] = result["prefill_checkpoint"] is not None
        report["decode_success"] = result["generated_tokens"] == args.decode_tokens
        if any(value < 0 or value >= vocab_size for value in result["generated_token_ids"]):
            raise ValueError("generated token ID outside tokenizer vocabulary")
        if not result["generated_text"]:
            raise ValueError("generation returned empty text")
        if not report["prefill_success"] or not report["decode_success"]:
            raise RuntimeError("requested prefill/decode did not complete")
        report["cache_stats_after_requested_job"] = generator.get_cache_stats()
        report["cache_tokens"] = {
            "capacity": args.context,
            "occupied_prompt_tokens": args.prefill_tokens,
            "generated_tokens": result["generated_tokens"],
            "remaining_capacity": args.context - args.prefill_tokens - result["generated_tokens"],
            "page_count": args.context // PAGE_SIZE,
        }
        report["vram_accounting_after_generation"] = serializable_vram_accounting(model, cache, generator)
        report["status"] = "ok"
    finally:
        report["driver_sampler"] = sampler.stop()


def base_report(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "status": "running",
        "started_timestamp": time.time(),
        "config": {
            "model": str(args.model),
            "context": args.context,
            "k_bits": args.k_bits,
            "v_bits": args.v_bits,
            "prefill_tokens": args.prefill_tokens,
            "decode_tokens": args.decode_tokens,
            "prefill_chunk": args.prefill_chunk,
            "allocation_only": args.allocation_only,
            "no_mtp": args.no_mtp,
            "qc_staging": os.environ.get("EXL3_QC_STAGING", "1"),
        },
        "environment": {"platform": platform.platform(), "pid": os.getpid()},
        "checkpoints": {},
        "timings": {},
        "model_loaded": False,
        "cache_allocated": False,
        "allocation_success": False,
        "prefill_success": False,
        "decode_success": False,
        "error": None,
    }


def execute_with_report(
    args: argparse.Namespace,
    runner: Callable[[argparse.Namespace, dict[str, Any]], None] = run_probe,
) -> int:
    report = base_report(args)
    try:
        runner(args, report)
        if report["status"] == "running":
            report["status"] = "ok"
        return_code = 0
    except BaseException as exc:
        report["status"] = "error"
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        return_code = 1
    finally:
        report["completed_timestamp"] = time.time()
        report["elapsed_seconds"] = report["completed_timestamp"] - report["started_timestamp"]
        atomic_write_json(args.output_json, report)
    return return_code


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return execute_with_report(args)


if __name__ == "__main__":
    raise SystemExit(main())

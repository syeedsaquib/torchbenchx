"""Compare one ONNX model on AMD NPU, DirectML GPU, and CPU providers."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROVIDERS = {
    "npu": "VitisAIExecutionProvider",
    "gpu": "DmlExecutionProvider",
    "cpu": "CPUExecutionProvider",
}
ACCELERATOR_PROVIDERS = {
    "VitisAIExecutionProvider",
    "DmlExecutionProvider",
}


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required. Install it with `python -m pip install numpy`.") from exc
    return np


def _onnxruntime():
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "ONNX Runtime is not installed in this Python environment. "
            "Use the AMD Ryzen AI environment for Vitis AI and DirectML providers."
        ) from exc
    return ort


def _input_dtype(type_name: str):
    np = _numpy()
    dtypes = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(int16)": np.int16,
        "tensor(int8)": np.int8,
        "tensor(uint8)": np.uint8,
        "tensor(bool)": np.bool_,
    }
    try:
        return dtypes[type_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported ONNX input type: {type_name}") from exc


def _resolve_input_shape(
    metadata_shape: Sequence[Any],
    batch_size: int,
    image_shape: Optional[Sequence[int]],
) -> Tuple[int, ...]:
    if not metadata_shape:
        raise ValueError("The ONNX input has no shape metadata.")

    shape = list(metadata_shape)
    if image_shape is not None:
        if len(shape) != 4 or len(image_shape) != 3:
            raise ValueError("--input-shape requires a rank-4 image input and C H W values.")
        requested_shape = [batch_size, *image_shape]
        for axis, (model_dimension, requested_dimension) in enumerate(zip(shape, requested_shape)):
            if isinstance(model_dimension, int) and model_dimension > 0 and model_dimension != requested_dimension:
                raise ValueError(
                    f"Requested input shape {tuple(requested_shape)} conflicts with "
                    f"the model's fixed dimension {model_dimension} at axis {axis}."
                )
        shape = requested_shape

    if shape[0] in (None, -1) or not isinstance(shape[0], int):
        shape[0] = batch_size
    elif shape[0] != batch_size:
        raise ValueError(
            f"Model has a fixed batch size of {shape[0]}, but --batch-size is {batch_size}."
        )

    for axis, dimension in enumerate(shape):
        if dimension in (None, -1) or not isinstance(dimension, int):
            shape[axis] = 1
    if any(dimension <= 0 for dimension in shape):
        raise ValueError(f"Could not resolve a positive input shape from {metadata_shape!r}.")
    return tuple(shape)


def make_input_feed(
    session: Any,
    batch_size: int,
    image_shape: Optional[Sequence[int]],
    seed: int,
) -> Dict[str, Any]:
    np = _numpy()
    feed: Dict[str, Any] = {}
    for input_index, input_meta in enumerate(session.get_inputs()):
        shape = _resolve_input_shape(input_meta.shape, batch_size, image_shape if input_index == 0 else None)
        dtype = _input_dtype(input_meta.type)
        rng = np.random.default_rng(seed + input_index)
        if dtype == np.bool_:
            value = rng.integers(0, 2, size=shape).astype(dtype)
        elif np.issubdtype(dtype, np.integer):
            value = rng.integers(0, 10, size=shape, dtype=dtype)
        else:
            value = rng.standard_normal(size=shape).astype(dtype)
        feed[input_meta.name] = value
    if not feed:
        raise ValueError("The ONNX model does not expose any inputs.")
    return feed


def inspect_profile_events(
    events: Iterable[Dict[str, Any]],
    requested_provider: str,
) -> Dict[str, Any]:
    """Summarize node assignment from ORT profiling events, not provider registration."""
    assigned_nodes: Dict[str, set] = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        args = event.get("args") or {}
        provider = args.get("provider") or args.get("execution_provider") or event.get("provider")
        if not provider:
            continue

        category = str(event.get("cat", "")).lower()
        name = str(event.get("name", ""))
        if category and category != "node" and not name.lower().endswith("_kernel_time"):
            continue

        node_id = args.get("node_index", args.get("node_name", name))
        assigned_nodes.setdefault(str(provider), set()).add(str(node_id))

    accelerator_nodes = (
        0
        if requested_provider == "CPUExecutionProvider"
        else len(assigned_nodes.get(requested_provider, set()))
    )
    cpu_nodes = len(assigned_nodes.get("CPUExecutionProvider", set()))
    if requested_provider == "CPUExecutionProvider":
        verification = "CPU EXECUTION" if cpu_nodes else "UNKNOWN"
    elif accelerator_nodes and cpu_nodes:
        verification = "PARTIAL"
    elif accelerator_nodes:
        verification = "VERIFIED"
    elif cpu_nodes:
        verification = "CPU FALLBACK"
    else:
        verification = "UNKNOWN"

    return {
        "verification": verification,
        "accelerator_nodes": accelerator_nodes or None,
        "cpu_nodes": cpu_nodes or None,
        "provider_node_counts": {
            provider: len(node_ids) for provider, node_ids in assigned_nodes.items()
        },
    }


def read_profile(profile_path: Optional[str], requested_provider: str) -> Dict[str, Any]:
    if not profile_path or not Path(profile_path).is_file():
        return {
            "verification": "UNKNOWN",
            "accelerator_nodes": None,
            "cpu_nodes": None,
            "provider_node_counts": {},
        }

    with Path(profile_path).open("r", encoding="utf-8") as profile_file:
        payload = json.load(profile_file)
    if isinstance(payload, list):
        events = payload
    elif isinstance(payload, dict):
        events = payload.get("traceEvents", payload.get("events", []))
    else:
        events = []
    return inspect_profile_events(events, requested_provider)


def benchmark_device(
    model_path: Path,
    device: str,
    batch_size: int,
    runs: int,
    warmup: int,
    image_shape: Optional[Sequence[int]],
    seed: int,
    profile_dir: Path,
) -> Dict[str, Any]:
    ort = _onnxruntime()
    available = ort.get_available_providers()
    provider = PROVIDERS[device]
    if provider not in available:
        raise RuntimeError(
            f"{provider} is unavailable. Providers in this environment: {available}"
        )

    profile_dir.mkdir(parents=True, exist_ok=True)
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(profile_dir / f"{device}_profile")
    provider_chain = [provider]
    if device != "cpu":
        provider_chain.append("CPUExecutionProvider")

    init_start = time.perf_counter()
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=provider_chain,
    )
    init_ms = (time.perf_counter() - init_start) * 1000.0
    actual_providers = session.get_providers()
    feed = make_input_feed(session, batch_size, image_shape, seed)

    try:
        warmup_start = time.perf_counter()
        for _ in range(warmup):
            session.run(None, feed)
        warmup_ms = (time.perf_counter() - warmup_start) * 1000.0

        latencies_ms: List[float] = []
        total_start = time.perf_counter()
        for _ in range(runs):
            run_start = time.perf_counter()
            session.run(None, feed)
            latencies_ms.append((time.perf_counter() - run_start) * 1000.0)
        elapsed = time.perf_counter() - total_start
    finally:
        profile_path = session.end_profiling()

    np = _numpy()
    profile = read_profile(profile_path, provider)
    return {
        "device": device.upper(),
        "provider": provider,
        "session_providers": actual_providers,
        "status": "OK",
        "session_init_ms": init_ms,
        "warmup_ms": warmup_ms,
        "runs": runs,
        "batch_size": batch_size,
        "total_inferences": runs * batch_size,
        "avg_latency_ms": float(np.mean(latencies_ms)),
        "median_latency_ms": float(np.median(latencies_ms)),
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
        "p99_latency_ms": float(np.percentile(latencies_ms, 99)),
        "throughput_images_per_sec": (runs * batch_size) / max(elapsed, 1e-12),
        "profile_path": profile_path,
        **profile,
    }


def print_summary(results: Sequence[Dict[str, Any]]) -> None:
    print("\n" + "=" * 92)
    print("ONNX RUNTIME PROVIDER COMPARISON")
    print("=" * 92)
    for result in results:
        print(f"\n{result['device']} ({result.get('provider', 'N/A')})")
        if result.get("status") != "OK":
            print(f"  FAILED: {result.get('error', 'Unknown error')}")
            continue
        print(f"  Session providers: {result['session_providers']}")
        print(f"  Initialization:   {result['session_init_ms']:.2f} ms")
        print(f"  Warmup:            {result['warmup_ms']:.2f} ms")
        print(f"  Average latency:   {result['avg_latency_ms']:.3f} ms/request")
        print(f"  Median latency:    {result['median_latency_ms']:.3f} ms/request")
        print(f"  P95 / P99:         {result['p95_latency_ms']:.3f} / {result['p99_latency_ms']:.3f} ms")
        print(f"  Throughput:        {result['throughput_images_per_sec']:.2f} images/sec")
        print(f"  Verification:      {result['verification']}")
        print(f"  Assigned nodes:    accelerator={result['accelerator_nodes']}, CPU={result['cpu_nodes']}")
        print(f"  Profile:           {result['profile_path']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark an ONNX model on available NPU, GPU, and CPU providers."
    )
    parser.add_argument("model", type=Path, help="Path to an ONNX model")
    parser.add_argument("--devices", nargs="+", choices=PROVIDERS, default=["npu", "gpu", "cpu"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--input-shape",
        nargs=3,
        type=int,
        metavar=("C", "H", "W"),
        help="Override the first rank-4 image input shape; otherwise use ONNX metadata.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    args = parser.parse_args()

    if not args.model.is_file():
        parser.error(f"ONNX model not found: {args.model}")
    if args.batch_size <= 0 or args.runs <= 0 or args.warmup < 0:
        parser.error("batch-size and runs must be positive; warmup cannot be negative.")

    try:
        ort = _onnxruntime()
    except RuntimeError as exc:
        parser.error(str(exc))
    print(f"Available providers: {ort.get_available_providers()}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    for device in args.devices:
        try:
            results.append(
                benchmark_device(
                    args.model,
                    device,
                    args.batch_size,
                    args.runs,
                    args.warmup,
                    args.input_shape,
                    args.seed,
                    args.output_dir / "profiles",
                )
            )
        except Exception as exc:
            results.append({
                "device": device.upper(),
                "provider": PROVIDERS[device],
                "status": "FAILED",
                "error": str(exc),
            })

    print_summary(results)
    json_path = args.output_dir / "results.json"
    csv_path = args.output_dir / "results.csv"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    csv_columns = [
        "device", "provider", "status", "error", "session_providers",
        "session_init_ms", "warmup_ms", "runs", "batch_size", "total_inferences",
        "avg_latency_ms", "median_latency_ms", "p95_latency_ms", "p99_latency_ms",
        "throughput_images_per_sec", "verification", "accelerator_nodes", "cpu_nodes",
        "provider_node_counts", "profile_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_columns, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            row = dict(result)
            if isinstance(row.get("session_providers"), list):
                row["session_providers"] = json.dumps(row["session_providers"])
            if isinstance(row.get("provider_node_counts"), dict):
                row["provider_node_counts"] = json.dumps(row["provider_node_counts"])
            writer.writerow(row)

    print(f"\nSaved results:\n  {json_path}\n  {csv_path}")
    return 0 if any(result.get("status") == "OK" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import psutil
import pandas as pd

try:
    import plotly.express as px
except Exception:  # pragma: no cover - optional at runtime
    px = None  # type: ignore


@dataclass
class _Result:
    Model: str
    Framework: str
    Device: str
    Latency_ms: float
    Throughput_sps: float
    Memory_MB: float
    Memory_Bytes: int
    Params_M: float


class BenchX:
    def __init__(
        self,
        device: Optional[str] = None,
        batch_size: int = 16,
        runs: int = 30,
        warmup: int = 10,
    ) -> None:
        self._device_pref = device
        self.batch_size = int(batch_size)
        self.runs = int(runs)
        self.warmup = int(warmup)
        self._results: List[_Result] = []

    # ----------------------------- Public API ----------------------------- #
    def benchmark(
        self,
        model: Any,
        name: str,
        input_shape: Sequence[int] = (3, 224, 224),
        framework: str = "pytorch",
    ) -> Dict[str, Any]:
        framework_lc = framework.lower()
        if framework_lc not in ("pytorch", "tensorflow"):
            raise ValueError("framework must be 'pytorch' or 'tensorflow'")

        if framework_lc == "pytorch":
            result = self._benchmark_torch(model, name, tuple(input_shape))
        else:
            result = self._benchmark_tf(model, name, tuple(input_shape))

        self._results.append(_Result(**result))
        return result

    def summary(self) -> pd.io.formats.style.Styler:
        df = pd.DataFrame([r.__dict__ for r in self._results])
        if df.empty:
            return pd.DataFrame(columns=[
                "Model", "Framework", "Device", "Latency_ms",
                "Throughput_sps", "Memory_MB", "Memory_Bytes", "Params_M",
            ]).style

        # Add humanized memory column for readability
        df["Memory"] = df["Memory_Bytes"].map(self._humanize_bytes)
        df_sorted = df.sort_values(by=["Throughput_sps"], ascending=False, ignore_index=True)

        styler = df_sorted.style.format({
            "Latency_ms": "{:.2f}",
            "Throughput_sps": "{:.0f}",
            "Memory_MB": "{:.1f}",
            "Params_M": "{:.2f}",
        })

        # Highlight best throughput
        if len(df_sorted) > 0:
            max_idx = df_sorted["Throughput_sps"].idxmax()
            def _highlight_best(row: pd.Series) -> List[str]:
                return [
                    "background-color: #e6ffed; font-weight: 600;" if row.name == max_idx else ""
                ] * len(row)
            styler = styler.apply(_highlight_best, axis=1)

        # Stronger, more visible color accents on white
        styler = styler.background_gradient(subset=["Latency_ms"], cmap="Reds_r")
        styler = styler.background_gradient(subset=["Throughput_sps"], cmap="Greens")
        styler = styler.background_gradient(subset=["Memory_MB"], cmap="Purples")

        # Subtle borders for a neat table look
        styler = styler.set_table_styles([
            {"selector": "table", "props": [
                ("border-collapse", "separate"), ("border-spacing", "0 8px"), ("background", "#ffffff")
            ]},
            {"selector": "th", "props": [("background", "#f0f4ff"), ("font-weight", "700"), ("color", "#222")]} ,
            {"selector": "td", "props": [("border", "1px solid #ddd"), ("padding", "8px 12px"), ("color", "#111")]},
        ])
        try:
            styler = styler.hide_columns(["Memory_Bytes"])  # keep the raw bytes hidden but available
        except Exception:
            pass
        return styler

    def visualize(
        self,
        metric_x: str = "Latency_ms",
        metric_y: str = "Throughput_sps",
    ) -> Any:
        if px is None:
            raise RuntimeError("plotly is not installed. Please `pip install plotly`. ")

        df = pd.DataFrame([r.__dict__ for r in self._results])
        if df.empty:
            raise RuntimeError("No results to visualize. Run benchmark() first.")

        fig = px.scatter(
            df,
            x=metric_x,
            y=metric_y,
            color="Framework",
            size="Params_M",
            hover_data=["Model", "Device", "Memory_MB"],
            title="torchbenchx — Performance Comparison",
        )
        fig.update_layout(legend_title_text="Framework")
        fig.update_traces(marker=dict(opacity=0.85, line=dict(width=1, color="#ffffff")))
        return fig

    # ----------------------------- Backends ------------------------------ #
    def _benchmark_torch(self, model: Any, name: str, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        import math
        import numpy as np
        try:
            import torch
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("PyTorch is required for framework='pytorch'") from exc

        device = self._resolve_torch_device()
        model = model.to(device)
        model.eval()

        batch = self.batch_size
        c, h, w = input_shape
        dummy = torch.randn(batch, c, h, w, device=device)

        # Parameter count (millions) and bytes
        param_elems = sum(p.numel() for p in model.parameters())
        params_m = param_elems / 1_000_000.0
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

        # Memory tracking (GPU peak if CUDA, else process RSS delta)
        process = psutil.Process()
        start_rss = process.memory_info().rss
        peak_mb_gpu = 0.0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup):
                _ = model(dummy)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)

        # Timed runs
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(self.runs):
                _ = model(dummy)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
        total = time.perf_counter() - start

        if device.type == "cuda":
            peak_bytes = torch.cuda.max_memory_allocated(device)
            peak_mb_gpu = peak_bytes / (1024 ** 2)
        end_rss = process.memory_info().rss
        delta_mb_cpu = max(0.0, (end_rss - start_rss) / (1024 ** 2))
        # Estimated activation memory from input (rough, but non-zero and comparable)
        activation_bytes = self.batch_size * c * h * w * 4  # float32
        est_bytes = max(int(param_bytes + activation_bytes), 0)
        # Prefer GPU peak if available, else use estimate (avoid zeroes)
        mem_bytes = int(peak_mb_gpu * (1024 ** 2)) if device.type == "cuda" and peak_mb_gpu > 0 else max(int(delta_mb_cpu * (1024 ** 2)), est_bytes)
        mem_mb = mem_bytes / (1024 ** 2)

        latency_ms = (total / self.runs) * 1000.0
        throughput_sps = (batch * self.runs) / max(total, 1e-9)

        return {
            "Model": name,
            "Framework": "PyTorch",
            "Device": str(device),
            "Latency_ms": float(latency_ms),
            "Throughput_sps": float(throughput_sps),
            "Memory_MB": float(mem_mb),
            "Memory_Bytes": int(mem_bytes),
            "Params_M": float(params_m),
        }

    def _benchmark_tf(self, model: Any, name: str, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        import numpy as np
        try:
            import tensorflow as tf
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("TensorFlow is required for framework='tensorflow'") from exc

        # TensorFlow typically uses channels-last
        if len(input_shape) != 3:
            raise ValueError("input_shape must be a 3-tuple for images")
        c, h, w = input_shape
        hwc = (h, w, c)

        batch = self.batch_size
        dummy_np = np.random.randn(batch, *hwc).astype("float32")
        dummy = tf.convert_to_tensor(dummy_np)

        # Ensure model is built
        _ = model(dummy, training=False)

        # Parameter count (millions) and bytes
        try:
            params = model.count_params()
        except Exception:
            params = sum(int(v.shape.num_elements() or 0) for v in model.trainable_variables)
        params_m = params / 1_000_000.0
        param_bytes = params * 4  # assume float32

        # Device string
        device = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"

        process = psutil.Process()
        start_rss = process.memory_info().rss

        # Warmup
        for _ in range(self.warmup):
            _ = model(dummy, training=False)
            # Best-effort sync using .numpy() materialization
            _ = _.numpy() if hasattr(_, "numpy") else _

        # Timed runs
        start = time.perf_counter()
        for _ in range(self.runs):
            out = model(dummy, training=False)
            _ = out.numpy() if hasattr(out, "numpy") else out
        total = time.perf_counter() - start

        end_rss = process.memory_info().rss
        delta_bytes = max(0, end_rss - start_rss)
        activation_bytes = self.batch_size * h * w * c * 4  # float32
        est_bytes = max(int(param_bytes + activation_bytes), 0)
        mem_bytes = max(delta_bytes, est_bytes)
        mem_mb = mem_bytes / (1024 ** 2)

        latency_ms = (total / self.runs) * 1000.0
        throughput_sps = (batch * self.runs) / max(total, 1e-9)

        return {
            "Model": name,
            "Framework": "TensorFlow",
            "Device": device.lower(),
            "Latency_ms": float(latency_ms),
            "Throughput_sps": float(throughput_sps),
            "Memory_MB": float(mem_mb),
            "Memory_Bytes": int(mem_bytes),
            "Params_M": float(params_m),
        }

    # ----------------------------- Utils --------------------------------- #
    def _resolve_torch_device(self):
        try:
            import torch
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("PyTorch is required for framework='pytorch'") from exc

        if self._device_pref is not None:
            return torch.device(self._device_pref)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _humanize_bytes(self, n: int) -> str:
        units = ["B", "KB", "MB", "GB", "TB"]
        x = float(n)
        idx = 0
        while x >= 1024.0 and idx < len(units) - 1:
            x /= 1024.0
            idx += 1
        if idx == 0:
            return f"{int(x)} {units[idx]}"
        return f"{x:.2f} {units[idx]}"

    # -------------------------- Enhanced Visuals ------------------------- #
    def visualize(
        self,
        metric_x: str = "Latency_ms",
        metric_y: str = "Throughput_sps",
    ) -> Any:
        if px is None:
            raise RuntimeError("plotly is not installed. Please `pip install plotly`. ")

        df = pd.DataFrame([r.__dict__ for r in self._results])
        if df.empty:
            raise RuntimeError("No results to visualize. Run benchmark() first.")

        fig = px.scatter(
            df,
            x=metric_x,
            y=metric_y,
            color="Framework",
            size="Params_M",
            hover_data=["Model", "Device", "Memory_MB"],
            title="torchbenchx — Performance Comparison",
        )
        fig.update_layout(
            legend_title_text="Framework",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            title=dict(font=dict(size=20, color="#111")),
            xaxis_title=dict(text=metric_x.replace("_", " "), font=dict(size=14)),
            yaxis_title=dict(text=metric_y.replace("_", " "), font=dict(size=14)),
            plot_bgcolor="#fff",
            paper_bgcolor="#fff",
        )
        fig.update_xaxes(showgrid=True, gridcolor="#eee")
        fig.update_yaxes(showgrid=True, gridcolor="#eee")
        fig.update_traces(marker=dict(opacity=0.9, line=dict(width=1, color="#ffffff")))
        return fig

    def visualize_all(self) -> Dict[str, Any]:
        if px is None:
            raise RuntimeError("plotly is not installed. Please `pip install plotly`. ")
        df = pd.DataFrame([r.__dict__ for r in self._results])
        if df.empty:
            raise RuntimeError("No results to visualize. Run benchmark() first.")

        figs: Dict[str, Any] = {}
        # Scatter throughput vs latency
        figs["scatter"] = self.visualize("Latency_ms", "Throughput_sps")

        # Bar: Throughput by model
        figs["throughput_bar"] = px.bar(
            df.sort_values("Throughput_sps", ascending=False),
            x="Model",
            y="Throughput_sps",
            color="Framework",
            title="Throughput (samples/sec) by Model",
        )
        figs["throughput_bar"].update_layout(legend=dict(orientation="h", y=1.02, x=1, xanchor="right"))
        figs["throughput_bar"].update_yaxes(gridcolor="#eee")

        # Bar: Latency by model (lower is better)
        figs["latency_bar"] = px.bar(
            df.sort_values("Latency_ms", ascending=True),
            x="Model",
            y="Latency_ms",
            color="Framework",
            title="Latency (ms per inference) by Model",
        )
        figs["latency_bar"].update_layout(legend=dict(orientation="h", y=1.02, x=1, xanchor="right"))
        figs["latency_bar"].update_yaxes(gridcolor="#eee")

        # Bar: Memory by model (MB)
        figs["memory_bar"] = px.bar(
            df.sort_values("Memory_MB", ascending=True),
            x="Model",
            y="Memory_MB",
            color="Framework",
            title="Estimated Memory Usage (MB) by Model",
        )
        figs["memory_bar"].update_layout(legend=dict(orientation="h", y=1.02, x=1, xanchor="right"))
        figs["memory_bar"].update_yaxes(gridcolor="#eee")
        return figs



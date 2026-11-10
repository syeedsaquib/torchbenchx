# torchbenchx — Unified Deep Learning Benchmark Library

Benchmark. Compare. Visualize.

`torchbenchx` is a framework-agnostic deep learning benchmarking library that measures latency, throughput, memory usage, and parameter count of models from both PyTorch and TensorFlow — all inside your Jupyter Notebook, with no servers or dashboards required.

It’s designed for researchers, developers, and AI enthusiasts who want quick, reliable model comparisons with interactive visualizations.

## Key Features

- **Dual Framework Support**: Benchmark models from both PyTorch and TensorFlow using one unified API.
- **Latency & Throughput Measurement**: Calculates per-inference latency and samples-per-second throughput.
- **Model Parameters & Memory Usage**: Automatically estimates memory footprint and parameter count.
- **Device Awareness**: Detects and runs on CPU or GPU automatically.
- **Notebook-native Visualization**: Generates Plotly scatter plots of performance metrics directly in Jupyter/Colab.
- **Minimal Dependencies**: Lightweight and ready to use.

## Installation

```bash
pip install torchbenchx
# Backends and viz libs
pip install torch torchvision tensorflow plotly pandas psutil
```

Optional: install `notebook` or `jupyterlab` if using outside Colab.

## Quick Start

```python
from torchbenchx import BenchX
import torchvision.models as tv_models
import tensorflow as tf

bench = BenchX(batch_size=16, runs=30)

bench.benchmark(tv_models.resnet18(weights=None), "ResNet18", framework="pytorch")
bench.benchmark(tf.keras.applications.MobileNetV3Small(), "MobileNetV3", framework="tensorflow")

bench.summary()       # Styled table
bench.visualize()     # Plotly scatter
```

## Compare ONNX Runtime providers (AMD NPU, GPU, and CPU)

The runnable example at `example/benchmark_onnx_providers.py` compares one ONNX model using the same deterministic inputs and workload on the Vitis AI NPU, DirectML GPU, and CPU execution providers. It reports session initialization and warmup separately from inference, saves per-request latency statistics and throughput, and preserves ONNX Runtime profiling files and CSV/JSON results.

Run it from the Ryzen AI Conda environment that contains the AMD ONNX Runtime installation:

```bat
conda activate ryzen-ai-1.7.1
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
python C:\path\to\torchbenchx\example\benchmark_onnx_providers.py C:\path\to\mobilenetv2.onnx
```

For NPU and GPU runs, the provider list should include `VitisAIExecutionProvider` and `DmlExecutionProvider`, as well as `CPUExecutionProvider`. A generic `pip install onnxruntime` installation may expose only CPU (or other non-AMD providers); it does not add the AMD providers. Use AMD's documented Ryzen AI 1.7.1 runtime/environment setup rather than replacing that runtime with a generic wheel.

Options include `--batch-size`, `--runs`, `--warmup`, and `--input-shape C H W` for a rank-4 image input. The model metadata is used by default; unresolved dynamic dimensions default to `1`, so specify `--input-shape` when the model needs a particular image size. Failed or unavailable backends are reported as failed rows while other devices continue. Verification is based on provider assignments in profile node events, not merely on provider registration; `UNKNOWN` means the profile did not provide enough assignment information to confirm execution.

Results are written to `benchmark_results\results.csv`, `benchmark_results\results.json`, and `benchmark_results\profiles\`. For example:

```bat
python example\benchmark_onnx_providers.py C:\Users\syeed\Desktop\mobilenetv2.onnx --runs 200 --warmup 20
```

## Core Class: BenchX

```python
BenchX(device=None, batch_size=16, runs=30, warmup=10)
```

- **device**: "cuda" or "cpu" — auto-detected if not set (for PyTorch)
- **batch_size**: Number of samples per inference
- **runs**: Number of repeated runs for averaging
- **warmup**: Initial runs ignored to stabilize measurements

### Methods

```python
benchmark(model, name, input_shape=(3, 224, 224), framework="pytorch")
```
Returns a dict with:
`{Model, Framework, Device, Latency_ms, Throughput_sps, Memory_MB, Params_M}`

```python
summary()  # Styled pandas DataFrame (stronger colors on white bg)
visualize(metric_x="Latency_ms", metric_y="Throughput_sps")  # Scatter with improved legend/layout
visualize_all()  # Returns dict of figures: scatter, throughput_bar, latency_bar, memory_bar
```

## Dependencies

- **torch**: PyTorch backend
- **tensorflow**: TensorFlow backend
- **plotly**: Interactive plotting
- **pandas**: Data handling and summary display
- **psutil**: Memory usage monitoring

## Internal Workflow

1. Warmup phase to stabilize performance
2. Timed inference loop across `runs`
3. Metrics: latency (ms/inference), throughput (samples/sec)
4. Memory estimation (GPU peak or process RSS)
5. Results aggregated into a DataFrame and visualized
6. Memory estimation: reports `Memory_Bytes`, `Memory_MB`, and a humanized `Memory`

## Use Cases

- **Research**: Compare architectures across frameworks
- **Development**: Optimize inference pipelines
- **Education**: Demonstrate benchmarking concepts
- **Model Selection**: Speed vs efficiency trade-offs

## Roadmap

- FLOPs estimation
- Power consumption metrics
- JAX and ONNX Runtime support
- Export results to CSV/JSON
- Leaderboard UI for reproducibility

## License

MIT


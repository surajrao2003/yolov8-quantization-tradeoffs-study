# Quantization Tradeoffs in YOLOv8 Across Frameworks and Hardware

This project measures how YOLOv8 quantization and input resolution affect model size, person mAP at IoU 0.5, and FPS when the same workflows are run through PyTorch, ONNX Runtime, TensorFlow Lite, and TensorRT on fixed benchmark machines.

This repository benchmarks YOLOv8 model variants across four deployment frameworks:

- **PyTorch**
- **ONNX Runtime**
- **TensorFlow Lite**
- **TensorRT**

We benchmark the same targets across different model sizes and input resolutions to understand the tradeoffs. We also identify the optimal edge deployment configuration (model size, precision/quantization, input resolution, hardware target) under resource constraints, then validate it through testing on a Jetson edge device.

### What are we trying to understand

1. How does quantization impact:
  - Model size
  - Accuracy
  - Speed
2. How does performance compare across hardware options.

### Metrics

We report:

- **Model size (MB)** (file size on disk)
- **Accuracy** as **person detection mAP at IoU 0.5**
- **Inference speed** as **FPS** (end to end over the dataset loop)

### Model variants and input sizes

- **Models**: YOLOv8n, YOLOv8s, YOLOv8m
- **Input sizes**: 320, 640, 1280

### Hardware (benchmark targets)

**Machine A (laptop)**

- **CPU1:** 13th Gen Intel(R) Core(TM) i5-13420H, 8 cores, 12 logical processors  
- **GPU1:** NVIDIA GeForce RTX 4050, 6GB VRAM  
- **CUDA version:** 12.8

**Machine B (workstation)**

- **CPU2:** Intel(R) Xeon(R) w5-3435X, 16 cores, 32 logical processors  
- **GPU2:** NVIDIA RTX 4000 Ada Generation, 20GB VRAM  
- **CUDA version:** 12.8

**Edge Device (NVIDIA Jetson Orin Nano 8 GB)**

- We use the NVIDIA Jetson Orin Nano, an Arm-based system-on-chip with an integrated NVIDIA GPU and 8 GB of unified memory, to evaluate edge deployment. The optimal YOLOv8 configuration identified by the optimization framework is deployed on the device and validated under resource-constrained conditions.

### Quantization and precision methods (by framework)

**PyTorch**

1. FP32 (baseline)
2. FP16 (half precision)
3. INT8 Dynamic Quantization (post training, weights only)
4. INT8 Static Post Training Quantization (PTQ, FX graph mode, calibration required, CPU oriented)

**ONNX Runtime**

1. FP32 ONNX export
2. FP16 ONNX export
3. INT8 Dynamic Quantization (`quantize_dynamic`)
4. INT8 Static Quantization (PTQ with calibration, QDQ format)

**TensorFlow Lite**

1. FP32 TFLite (baseline)
2. Float16 Quantization (weights stored as float16)
3. Dynamic Range Quantization (weights int8, activations quantized dynamically)
4. Full Integer Quantization (INT8 weights and activations, representative dataset required)

**TensorRT**

1. FP32 engine
2. FP16 engine
3. INT8 engine (calibration required)

### Repo layout


| Path                                | Role                                                                                                                                                                               |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `quantization_scripts/`             | Export and quantization scripts for all frameworks                                                                                                                                 |
| `optimization/`                     | `benchmark_results.csv` (manual snapshot), utility scoring (`run_optimization.py`, default resolution 640), outputs under `output_csv_results/` (CSV, report, `tradeoff_plot.png`) |
| `onnx_benchmarking/`                | ONNX Runtime benchmarking (FPS, size, mAP)                                                                                                                                         |
| `pytorch_benchmarking/`             | PyTorch benchmarking (FPS, size, mAP)                                                                                                                                              |
| `tflite_benchmarking/`              | TFLite benchmarking (FPS, size, mAP)                                                                                                                                               |
| `tensorrt_benchmarking/`            | TensorRT benchmarking (FPS, size, mAP); `trt_inference_jetson.py` for Jetson Orin Nano validation (separate default output folder)                                                 |
| `code_files/`                       | Shared preprocessing, postprocessing, and mAP utilities used by benchmarking folders                                                                                               |
| `models/yolo_models/`               | Ultralytics `.pt` input weights                                                                                                                                                    |
| `models/onnx_quantized_models/`     | ONNX outputs grouped by precision and input size                                                                                                                                   |
| `models/pytorch_quantized_models/`  | PyTorch outputs grouped by precision and input size                                                                                                                                |
| `models/tflite_quantized_models/`   | TFLite outputs grouped by precision and input size                                                                                                                                 |
| `models/tensorrt_quantized_models/` | TensorRT engines grouped by precision and input size                                                                                                                               |
| `imagedata/`                        | Example dataset folder with `images/` and `labels/`                                                                                                                                |
| `outputfolder/`                     | Default annotated output folder recreated on each run                                                                                                                              |


### Project structure

```
yolov8-quantization-tradeoffs-study/
├── code_files/                                 # contains shared files used by all frameworks
├── onnx_benchmarking/                          # used for ONNX inference
│   ├── main_execution.py
│   └── onnx_inference.py
├── pytorch_benchmarking/                       # used for PyTorch inference
│   ├── main_execution.py
│   └── pytorch_inference.py
├── tflite_benchmarking/                        # used for tflite inference 
│   ├── main_execution.py
│   ├── tflite_inference.py
│   └── tflite_runtime.py
├── tensorrt_benchmarking/                      # used for tensorrt inference (works only on NVIDIA GPU's)
│   ├── main_execution.py
│   ├── trt_inference_jetson.py                 # Jetson Orin Nano: same metrics, default outputfolder_trt_jetson/
│   ├── trt_inference.py                        
│   └── trt_runtime.py
├── quantization_scripts/                       # scripts to export, quantize models
│   ├── calibration_dataset/
│   ├── onnx_quantization_files/                # onnx export and quantization scripts
│   ├── pytorch_quantization_files/             # pytorch export and quantization scripts
│   ├── tflite_quantization_files/              # tflite export and quantization scripts
│   └── tensorrt_quantization_files/            # tensorrt engine build scripts
├── optimization/                               
│   ├── benchmark_results.csv                   # hardcoded benchmark results
│   ├── run_optimization.py                     # to find utility scores of different configurations for edge deployment
│   └── output_csv_results/                     # full_results_with_utility.csv, optimization_report.md
├── models/
│   ├── yolo_models/                            # original yolov8 pytorch models (actual fp32 size)
│   ├── onnx_quantized_models/
│   │   ├── FP32_onnx_models/
│   │   │   ├── 320/
│   │   │   ├── 640/
│   │   │   └── 1280/
│   │   ├── FP16_onnx_models/
│   │   ├── INT8_onnx_dynamic_quantized_models/
│   │   └── INT8_onnx_static_quantized_models/
│   ├── pytorch_quantized_models/
│   │   ├── FP16_pytorch_models/
│   │   ├── INT8_pytorch_dynamic_quantized_models/
│   │   └── INT8_pytorch_static_quantized_models/
│   ├── tflite_quantized_models/                 # only works on CPU
│   │   ├── FP32_tflite_models/
│   │   ├── FP16_tflite_models/
│   │   ├── dynamic_range_tflite_models/
│   │   └── full_integer_INT8_tflite_models/
│   ├── tensorrt_quantized_models/               # only run on NVIDIA GPU's
│   │   ├── FP32_trt_models/
│   │   ├── FP16_trt_models/
│   │   └── INT8_trt_models/
│   └── config_custom_data.yaml                 
├── imagedata/                                   # example: images/ + labels/
├── outputfolder/                                # inference outputs (created/overwritten on run)
├── requirements.txt
└── README.md
```

### Environment setup

Use a **conda** environment. Install **PyTorch** and the GPU stack **before** `requirements.txt` so versions stay consistent.

1. **Create and activate the environment**

```bash
conda create -n yolo_env python=3.11 -y
conda activate yolo_env
```

1. **PyTorch**
  Install from [pytorch.org](https://pytorch.org/get-started/locally/) using the command that matches your OS and CUDA (or CPU) build.
2. **ONNX Runtime (CUDA)**

```bash
pip install "onnxruntime-gpu[cuda,cudnn]==1.23.2"
```

1. **TensorRT and PyCUDA (conda)**
  These conda installs are **required** for the TensorRT build and benchmark scripts to work reliably with the rest of the stack.

```bash
conda install tensorrt=10.16.1.11
conda install -c conda-forge "pycuda=2025.1.2"
```

1. **TensorFlow** (TFLite export and benchmarking)

```bash
pip install tensorflow
```

1. **Remaining Python packages**

```bash
pip install -r requirements.txt
```

If you will not use TensorRT, you can skip steps 3–4 and install ONNX Runtime for CPU or GPU from the [ONNX Runtime docs](https://onnxruntime.ai/docs/install/) instead of step 2.

### Dataset format

All benchmarking entrypoints expect `--data-root` to contain:

```
<data-root>/
  images/
  labels/
```

The labels are YOLO `.txt` format, and we compute mAP for **person class id 0**.

### How to export and quantize models

Each quantization script has a short run block at the top of the file. The common entrypoints are:

- **ONNX**: `quantization_scripts/onnx_quantization_files/`
- **PyTorch**: `quantization_scripts/pytorch_quantization_files/`
- **TFLite**: `quantization_scripts/tflite_quantization_files/`
- **TensorRT**: `quantization_scripts/tensorrt_quantization_files/`

### How to run benchmarking

All benchmark entrypoints print:

- FPS
- device=cpu or device=gpu (when applicable)
- Model size (on disk)
- mAP@IoU=0.50

**ONNX Runtime**

```powershell
python .\onnx_benchmarking\main_execution.py --model "models\onnx_quantized_models\FP32_onnx_models\640\yolov8n_640.onnx" --device cpu --input-size 640 --data-root "imagedata"
```

**PyTorch**

```powershell
python .\pytorch_benchmarking\main_execution.py --model "models\yolo_models\yolov8n.pt" --device cpu --input-size 640 --data-root "imagedata"
```

**TFLite**

```powershell
python .\tflite_benchmarking\main_execution.py --model "models\tflite_quantized_models\FP32_tflite_models\640\yolov8n_640_fp32.tflite" --device cpu --input-size 640 --data-root "imagedata"
```

**TensorRT**

```powershell
python .\tensorrt_benchmarking\main_execution.py --engine "models\tensorrt_quantized_models\FP16_trt_models\640\yolov8n_640_trt_fp16.engine" --input-size 640 --data-root "imagedata"
```

**TensorRT on Jetson Orin Nano (edge validation)**

Use `tensorrt_benchmarking/trt_inference_jetson.py` on the board after you build a `.engine` **on that Jetson** (same workflow as desktop TensorRT: ONNX export on a dev machine, then engine build on Orin for Orin). It prints the same FPS, size, and mAP lines as `main_execution.py` but labels the device line as Jetson and writes overlays to `**outputfolder_trt_jetson/`** by default (or `--output-dir`).

```bash
python3 tensorrt_benchmarking/trt_inference_jetson.py \
  --engine models/tensorrt_quantized_models/FP16_trt_models/640/yolov8n_640_trt_fp16.engine \
  --input-size 640 \
  --data-root imagedata
```

### Finding the Best Configuration for Edge Deployment (Using Optimization)

After you fill `optimization/benchmark_results.csv`, you can rank configurations with a scalar **utility** score. The optimizer uses **only rows with input resolution 640**, so every framework and model variant is compared at the same letterbox size.

Run:

```powershell
python .\optimization\run_optimization.py
```

This also writes `**optimization/output_csv_results/tradeoff_plot.png**`: two scatter panels (FPS vs [mAP@0.5](mailto:mAP@0.5) and model size vs [mAP@0.5](mailto:mAP@0.5)) for the same filtered rows, colored by framework, with feasible vs not feasible markers. Use `**--no-plot**` to skip the image.

**Utility definition (defaults in the script):** `U = alpha * A_norm + beta * F_norm - gamma * S_norm`

Where `A_norm` uses **[mAP@0.5](mailto:mAP@0.5)**, `F_norm` uses **FPS**, and `S_norm` uses **model size (MB)**. Normalization is **min max over the filtered 640 rows only**.

Default weights: **alpha = 0.50**, **beta = 0.30**, **gamma = 0.20**.

**Feasibility constraints (defaults):**

- **Accuracy:** `map50 >= 0.65`
- **Latency:** `latency_ms <= 100`
- **Size:** `size_mb <= 50`

Override weights and cuts with CLI flags such as `--alpha`, `--beta`, `--gamma`, `--amin`, `--lmax`, `--smax`. Use `--csv` to point at another table, or `--resolution` to change the input size filter (default **640**).

### Notes

- `optimization/benchmark_results.csv` is a fixed table of results copied in from completed benchmark runs. It is not written or updated by the benchmarking scripts.
- `outputfolder/` is recreated on each run when it is used as the default output location.
- **TensorRT:** A TensorRT `.engine` file is optimized for the exact GPU it was built on. So, if you build it on **GPU1** and run it on **GPU2**, TensorRT may produce wrong results, crash, hang, or run slower. Rebuild the `.engine` separately on each GPU/machine.
- **Jetson:** Treat the Orin like another GPU target: build engines on the Jetson, use JetPack’s TensorRT stack, and see `requirements.txt` comments for pip on aarch64.

### Conclusion

This repo provides a repeatable workflow for benchmarking YOLOv8 deployment configurations across PyTorch, ONNX Runtime, TFLite, and TensorRT. It compares model size, FPS, and person [mAP@0.5](mailto:mAP@0.5) under different quantization settings, input resolutions, and hardware targets, making it easier to evaluate the trade-offs between accuracy, speed, and file size. Since no single configuration is best for every deployment scenario, the optimization script ranks results from `benchmark_results.csv` using user-defined weights and constraints, helping select the most suitable setup for a given edge deployment requirement. The selected configuration can then be validated on edge hardware, such as a Jetson Orin Nano or a local NVIDIA GPU, using TensorRT or another suitable deployment framework.
# Utility optimization report
## Scope
- Input resolution filter: **640** (common comparison point across frameworks).
- Source CSV: `optimization/benchmark_results.csv`
- Rows after filter: **126**
- Feasible rows: **79** / 126

## Utility
Normalized terms use **min max** over the filtered rows only.
- **U** = 0.5 * A_norm + 0.3 * F_norm - 0.2 * S_norm
- **A_norm**: from **mAP@0.5** (higher is better)
- **F_norm**: from **FPS** (higher is better)
- **S_norm**: from **model size (MB)** (larger file penalizes utility)

## Feasibility defaults
- map50 >= **0.65**
- latency_ms <= **100.0**
- size_mb <= **50.0**

Override weights and cuts with `--alpha`, `--beta`, `--gamma`, `--amin`, `--lmax`, `--smax`.

## Best feasible configuration
| Field | Value |
| --- | --- |
| framework | TensorRT |
| precision | FP16 |
| hardware | GPU2 |
| model | YOLOv8s |
| fps | 57.5626 |
| latency_ms | 17.3724 |
| size_mb | 24.35 |
| map50 | 0.707869 |
| utility | 0.674796 |

## Top 15 by utility (all rows, feasible and not)
| framework   | precision    |   resolution | hardware   | model   |     fps |   latency_ms |   size_mb |    map50 |   notes |   A_norm |   F_norm |    S_norm |   utility | feasible   |
|:------------|:-------------|-------------:|:-----------|:--------|--------:|-------------:|----------:|---------:|--------:|---------:|---------:|----------:|----------:|:-----------|
| TensorRT    | FP16         |          640 | GPU2       | YOLOv8s | 57.5626 |      17.3724 |     24.35 | 0.707869 |     nan | 0.874732 | 0.888391 | 0.145433  |  0.674796 | True       |
| PyTorch     | FP32         |          640 | GPU2       | YOLOv8s | 62.4497 |      16.0129 |     21.53 | 0.697097 |     nan | 0.808442 | 0.965186 | 0.12595   |  0.668587 | True       |
| TensorRT    | FP16         |          640 | GPU2       | YOLOv8m | 52.5428 |      19.0321 |     52.53 | 0.723565 |     nan | 0.971323 | 0.80951  | 0.340127  |  0.660489 | False      |
| TensorRT    | FP16         |          640 | GPU1       | YOLOv8s | 54.0923 |      18.4869 |     23.34 | 0.707808 |     nan | 0.874356 | 0.833859 | 0.138455  |  0.659645 | True       |
| PyTorch     | FP16         |          640 | GPU2       | YOLOv8s | 58.4418 |      17.111  |     21.54 | 0.697065 |     nan | 0.808245 | 0.902206 | 0.126019  |  0.649581 | True       |
| TensorRT    | FP16         |          640 | GPU1       | YOLOv8m | 49.8339 |      20.0667 |     52.2  | 0.722321 |     nan | 0.963667 | 0.766943 | 0.337847  |  0.644347 | False      |
| PyTorch     | INT8 Dynamic |          640 | GPU2       | YOLOv8s | 61.9458 |      16.1431 |     42.93 | 0.697097 |     nan | 0.808442 | 0.957268 | 0.273801  |  0.636641 | True       |
| TensorRT    | FP16         |          640 | GPU2       | YOLOv8n | 58.4916 |      17.0965 |      8.6  | 0.679939 |     nan | 0.702854 | 0.902989 | 0.0366174 |  0.615    | True       |
| TensorRT    | FP32         |          640 | GPU2       | YOLOv8s | 54.0234 |      18.5105 |     56.08 | 0.70787  |     nan | 0.874738 | 0.832776 | 0.364654  |  0.614271 | False      |
| PyTorch     | FP32         |          640 | GPU2       | YOLOv8m | 57.7331 |      17.3211 |     49.7  | 0.699277 |     nan | 0.821857 | 0.89107  | 0.320575  |  0.614135 | True       |
| TensorRT    | FP16         |          640 | GPU1       | YOLOv8n | 57.9883 |      17.2449 |      7.95 | 0.679762 |     nan | 0.701764 | 0.89508  | 0.0321266 |  0.612981 | True       |
| TensorRT    | FP32         |          640 | GPU2       | YOLOv8n | 57.401  |      17.4213 |     15.68 | 0.67989  |     nan | 0.702552 | 0.885851 | 0.0855327 |  0.599925 | True       |
| PyTorch     | FP16         |          640 | GPU1       | YOLOv8s | 47.6628 |      20.9807 |     21.54 | 0.697073 |     nan | 0.808294 | 0.732826 | 0.126019  |  0.598791 | True       |
| TensorRT    | FP32         |          640 | GPU1       | YOLOv8s | 49.6425 |      20.144  |     61.19 | 0.707869 |     nan | 0.874732 | 0.763935 | 0.399959  |  0.586555 | False      |
| PyTorch     | FP32         |          640 | GPU1       | YOLOv8s | 44.7074 |      22.3677 |     21.53 | 0.697108 |     nan | 0.80851  | 0.686385 | 0.12595   |  0.58498  | True       |

## Top 15 among feasible only
| framework   | precision    |   resolution | hardware   | model   |     fps |   latency_ms |   size_mb |    map50 |   notes |   A_norm |   F_norm |    S_norm |   utility | feasible   |
|:------------|:-------------|-------------:|:-----------|:--------|--------:|-------------:|----------:|---------:|--------:|---------:|---------:|----------:|----------:|:-----------|
| TensorRT    | FP16         |          640 | GPU2       | YOLOv8s | 57.5626 |      17.3724 |     24.35 | 0.707869 |     nan | 0.874732 | 0.888391 | 0.145433  |  0.674796 | True       |
| PyTorch     | FP32         |          640 | GPU2       | YOLOv8s | 62.4497 |      16.0129 |     21.53 | 0.697097 |     nan | 0.808442 | 0.965186 | 0.12595   |  0.668587 | True       |
| TensorRT    | FP16         |          640 | GPU1       | YOLOv8s | 54.0923 |      18.4869 |     23.34 | 0.707808 |     nan | 0.874356 | 0.833859 | 0.138455  |  0.659645 | True       |
| PyTorch     | FP16         |          640 | GPU2       | YOLOv8s | 58.4418 |      17.111  |     21.54 | 0.697065 |     nan | 0.808245 | 0.902206 | 0.126019  |  0.649581 | True       |
| PyTorch     | INT8 Dynamic |          640 | GPU2       | YOLOv8s | 61.9458 |      16.1431 |     42.93 | 0.697097 |     nan | 0.808442 | 0.957268 | 0.273801  |  0.636641 | True       |
| TensorRT    | FP16         |          640 | GPU2       | YOLOv8n | 58.4916 |      17.0965 |      8.6  | 0.679939 |     nan | 0.702854 | 0.902989 | 0.0366174 |  0.615    | True       |
| PyTorch     | FP32         |          640 | GPU2       | YOLOv8m | 57.7331 |      17.3211 |     49.7  | 0.699277 |     nan | 0.821857 | 0.89107  | 0.320575  |  0.614135 | True       |
| TensorRT    | FP16         |          640 | GPU1       | YOLOv8n | 57.9883 |      17.2449 |      7.95 | 0.679762 |     nan | 0.701764 | 0.89508  | 0.0321266 |  0.612981 | True       |
| TensorRT    | FP32         |          640 | GPU2       | YOLOv8n | 57.401  |      17.4213 |     15.68 | 0.67989  |     nan | 0.702552 | 0.885851 | 0.0855327 |  0.599925 | True       |
| PyTorch     | FP16         |          640 | GPU1       | YOLOv8s | 47.6628 |      20.9807 |     21.54 | 0.697073 |     nan | 0.808294 | 0.732826 | 0.126019  |  0.598791 | True       |
| PyTorch     | FP32         |          640 | GPU1       | YOLOv8s | 44.7074 |      22.3677 |     21.53 | 0.697108 |     nan | 0.80851  | 0.686385 | 0.12595   |  0.58498  | True       |
| PyTorch     | FP32         |          640 | GPU2       | YOLOv8n | 63.4439 |      15.762  |      6.23 | 0.660001 |     nan | 0.580157 | 0.980809 | 0.0202432 |  0.580273 | True       |
| PyTorch     | INT8 Dynamic |          640 | GPU2       | YOLOv8n | 64.6652 |      15.4643 |     12.34 | 0.660001 |     nan | 0.580157 | 1        | 0.0624568 |  0.577587 | True       |
| ONNX        | INT8 Dynamic |          640 | GPU2       | YOLOv8s | 41.3695 |      24.1724 |     42.79 | 0.7079   |     nan | 0.874922 | 0.633934 | 0.272834  |  0.573075 | True       |
| TensorRT    | FP32         |          640 | GPU1       | YOLOv8n | 52.1043 |      19.1923 |     17.96 | 0.679877 |     nan | 0.702472 | 0.80262  | 0.101285  |  0.571765 | True       |

## Full table
See `full_results_with_utility.csv` in this folder for every filtered row with `A_norm`, `F_norm`, `S_norm`, `utility`, and `feasible`.

## Tradeoff figure
Scatter summary: **`tradeoff_plot.png`** — FPS vs mAP@0.5 and model size vs mAP@0.5 (markers by framework; circle = feasible under current cuts, x = not; dashed line is `--amin`).

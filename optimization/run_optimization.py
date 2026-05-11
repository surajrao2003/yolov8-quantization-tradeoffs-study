"""
Utility-based ranking for deployment configs (640 input size only).

Run from project root:

  python optimization/run_optimization.py

Reads optimization/benchmark_results.csv, keeps rows with resolution 640, normalizes mAP@0.5, FPS, and size (MB)
over that slice, scores U = alpha*A_norm + beta*F_norm - gamma*S_norm, applies default feasibility cuts,
writes optimization/output_csv_results/full_results_with_utility.csv, optimization_report.md,
and tradeoff_plot.png (FPS vs mAP, size vs mAP for the same filtered rows).

# default input CSV -> optimization/benchmark_results.csv
# Optional: --csv path/to/results.csv, --no-plot to skip the PNG
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = OPT_DIR / "benchmark_results.csv"
DEFAULT_OUT_DIR = OPT_DIR / "output_csv_results"


def _path_relative_to_repo(path: Path, *, repo: Path) -> str:
    """Path for markdown (always relative to repo root, forward slashes, no machine-specific absolutes)."""
    rel = os.path.relpath(path.resolve(), repo.resolve())
    return rel.replace("\\", "/")


DEFAULT_ALPHA = 0.50
DEFAULT_BETA = 0.30
DEFAULT_GAMMA = 0.20
DEFAULT_AMIN = 0.65
DEFAULT_LMAX = 100.0
DEFAULT_SMAX = 50.0
DEFAULT_RESOLUTION = 640

# Distinct colors for scatter (framework names in benchmark CSV).
_FRAMEWORK_COLORS: dict[str, str] = {
    "ONNX": "#1f77b4",
    "PyTorch": "#ff7f0e",
    "TFLite": "#2ca02c",
    "TensorRT": "#d62728",
}


def _min_max_norm(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mn = float(s.min(skipna=True))
    mx = float(s.max(skipna=True))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
        return pd.Series(0.5, index=s.index, dtype="float64")
    return (s - mn) / (mx - mn)


def _write_tradeoff_plot(
    sub: pd.DataFrame,
    out_png: Path,
    *,
    resolution: int,
    amin: float,
) -> bool:
    """Scatter tradeoffs: throughput vs accuracy, footprint vs accuracy (same slice as utility)."""
    plot_df = sub.dropna(subset=["fps", "map50", "size_mb"]).copy()
    if plot_df.empty:
        return False

    frameworks = sorted(plot_df["framework"].astype(str).unique())
    colors = {fw: _FRAMEWORK_COLORS.get(fw, "#7f7f7f") for fw in frameworks}

    fig, (ax_fps, ax_sz) = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    fig.suptitle(
        f"Deployment tradeoffs (input {resolution}, all hardware in table)",
        fontsize=12,
    )

    def _scatter_panel(ax: plt.Axes, xcol: str, xlabel: str, title: str) -> None:
        for fw in frameworks:
            part = plot_df[plot_df["framework"].astype(str) == fw]
            c = colors[fw]
            ok = part["feasible"].astype(bool)
            if ok.any():
                p0 = part.loc[ok]
                ax.scatter(
                    p0[xcol],
                    p0["map50"],
                    c=c,
                    marker="o",
                    s=38,
                    alpha=0.88,
                    edgecolors="black",
                    linewidths=0.35,
                )
            if (~ok).any():
                p1 = part.loc[~ok]
                ax.scatter(
                    p1[xcol],
                    p1["map50"],
                    c=c,
                    marker="x",
                    s=46,
                    alpha=0.65,
                    linewidths=1.0,
                )
        ax.axhline(amin, color="gray", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("mAP@0.5 (person)")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)

    _scatter_panel(ax_fps, "fps", "FPS (higher is better)", "Speed vs accuracy")
    _scatter_panel(ax_sz, "size_mb", "Model size (MB)", "Size vs accuracy")

    from matplotlib.lines import Line2D

    legend_handles: list[Line2D] = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors[fw],
            markeredgecolor="black",
            markersize=8,
            label=fw,
        )
        for fw in frameworks
    ]
    legend_handles.append(
        Line2D([0], [0], marker="o", color="w", markerfacecolor="lightgray", markeredgecolor="k", markersize=8, label="feasible")
    )
    legend_handles.append(Line2D([0], [0], marker="x", color="k", linestyle="None", markersize=9, label="not feasible"))
    legend_handles.append(
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1.5, label=f"map50 = {amin:g} (amin cut)")
    )

    fig.legend(handles=legend_handles, loc="outside lower center", ncol=min(4, len(legend_handles)), fontsize=8)

    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> int:
    p = argparse.ArgumentParser(description="Utility score over optimization/benchmark_results.csv (640 resolution slice).")
    p.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Benchmark CSV path (default: optimization/benchmark_results.csv next to this script).",
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR, help="Where to write CSV + markdown report.")
    p.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION, help="Input size filter (default: 640).")
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Weight on normalized mAP@0.5.")
    p.add_argument("--beta", type=float, default=DEFAULT_BETA, help="Weight on normalized FPS.")
    p.add_argument("--gamma", type=float, default=DEFAULT_GAMMA, help="Weight on normalized model size (subtracted).")
    p.add_argument("--amin", type=float, default=DEFAULT_AMIN, help="Feasibility: minimum map50.")
    p.add_argument("--lmax", type=float, default=DEFAULT_LMAX, help="Feasibility: maximum latency_ms.")
    p.add_argument("--smax", type=float, default=DEFAULT_SMAX, help="Feasibility: maximum size_mb.")
    p.add_argument("--no-plot", action="store_true", help="Skip writing tradeoff_plot.png.")
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        csv_path = REPO / csv_path
    if not csv_path.is_file():
        raise FileNotFoundError(f"Benchmark CSV not found: {args.csv}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    required = {"framework", "precision", "resolution", "hardware", "model", "fps", "latency_ms", "size_mb", "map50"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    df["resolution"] = pd.to_numeric(df["resolution"], errors="coerce")
    sub = df[df["resolution"] == int(args.resolution)].copy()
    if sub.empty:
        raise ValueError(f"No rows with resolution={args.resolution} in {csv_path}")

    for col in ("fps", "latency_ms", "size_mb", "map50"):
        sub[col] = pd.to_numeric(sub[col], errors="coerce")

    sub["A_norm"] = _min_max_norm(sub["map50"])
    sub["F_norm"] = _min_max_norm(sub["fps"])
    sub["S_norm"] = _min_max_norm(sub["size_mb"])

    sub["utility"] = (
        float(args.alpha) * sub["A_norm"]
        + float(args.beta) * sub["F_norm"]
        - float(args.gamma) * sub["S_norm"]
    )

    sub["feasible"] = (
        (sub["map50"] >= float(args.amin))
        & (sub["latency_ms"] <= float(args.lmax))
        & (sub["size_mb"] <= float(args.smax))
    )

    sub = sub.sort_values("utility", ascending=False)
    out_csv = out_dir / "full_results_with_utility.csv"
    sub.to_csv(out_csv, index=False)

    feas = sub[sub["feasible"]]
    best = feas.iloc[0] if len(feas) else None

    report_path = out_dir / "optimization_report.md"
    lines: list[str] = []
    lines.append("# Utility optimization report\n")
    lines.append("## Scope\n")
    lines.append(
        f"- Input resolution filter: **{int(args.resolution)}** (common comparison point across frameworks).\n"
    )
    lines.append(f"- Source CSV: `{_path_relative_to_repo(csv_path, repo=REPO)}`\n")
    lines.append(f"- Rows after filter: **{len(sub)}**\n")
    lines.append(f"- Feasible rows: **{int(sub['feasible'].sum())}** / {len(sub)}\n")
    lines.append("\n## Utility\n")
    lines.append("Normalized terms use **min max** over the filtered rows only.\n")
    lines.append(
        f"- **U** = {args.alpha:g} * A_norm + {args.beta:g} * F_norm - {args.gamma:g} * S_norm\n"
    )
    lines.append("- **A_norm**: from **mAP@0.5** (higher is better)\n")
    lines.append("- **F_norm**: from **FPS** (higher is better)\n")
    lines.append("- **S_norm**: from **model size (MB)** (larger file penalizes utility)\n")
    lines.append("\n## Feasibility defaults\n")
    lines.append(f"- map50 >= **{args.amin}**\n")
    lines.append(f"- latency_ms <= **{args.lmax}**\n")
    lines.append(f"- size_mb <= **{args.smax}**\n")
    lines.append("\nOverride weights and cuts with `--alpha`, `--beta`, `--gamma`, `--amin`, `--lmax`, `--smax`.\n")

    lines.append("\n## Best feasible configuration\n")
    if best is not None:
        lines.append("| Field | Value |\n| --- | --- |\n")
        for k in ("framework", "precision", "hardware", "model", "fps", "latency_ms", "size_mb", "map50", "utility"):
            v = best[k]
            if isinstance(v, float):
                lines.append(f"| {k} | {v:.6g} |\n")
            else:
                lines.append(f"| {k} | {v} |\n")
    else:
        lines.append("_No row satisfies the feasibility constraints. Relax `--amin`, `--lmax`, or `--smax`._\n")

    lines.append("\n## Top 15 by utility (all rows, feasible and not)\n")
    lines.append(sub.head(15).to_markdown(index=False))
    lines.append("\n")

    lines.append("\n## Top 15 among feasible only\n")
    if len(feas):
        lines.append(feas.head(15).to_markdown(index=False))
        lines.append("\n")
    else:
        lines.append("_None._\n")

    lines.append(f"\n## Full table\n")
    lines.append(f"See `{out_csv.name}` in this folder for every filtered row with `A_norm`, `F_norm`, `S_norm`, `utility`, and `feasible`.\n")

    plot_path = out_dir / "tradeoff_plot.png"
    plot_written = False
    if not args.no_plot:
        plot_written = _write_tradeoff_plot(
            sub, plot_path, resolution=int(args.resolution), amin=float(args.amin)
        )
        if plot_written:
            lines.append("\n## Tradeoff figure\n")
            lines.append(
                f"Scatter summary: **`{plot_path.name}`** — FPS vs mAP@0.5 and model size vs mAP@0.5 "
                f"(markers by framework; circle = feasible under current cuts, x = not; dashed line is `--amin`).\n"
            )

    report_path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote: {_path_relative_to_repo(out_csv, repo=REPO)}")
    print(f"Wrote: {_path_relative_to_repo(report_path, repo=REPO)}")
    if plot_written:
        print(f"Wrote: {_path_relative_to_repo(plot_path, repo=REPO)}")
    if best is not None:
        print(f"Best feasible utility: {float(best['utility']):.6f} ({best['framework']} / {best['precision']} / {best['model']} / {best['hardware']})")
    else:
        print("No feasible configuration under current constraints.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

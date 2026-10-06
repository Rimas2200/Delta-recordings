"""Compare two CSP watchdog captures on identical plot scales."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_helpers(robot_root: Path):
    from plot_delta_csv import positions_xyz
    from plot_watchdog_dashboard import read_watchdog

    return positions_xyz, read_watchdog


def prepare(path: Path, robot_root: Path, positions_xyz, read_watchdog) -> dict:
    record = read_watchdog(path)
    t = record["t"]
    target = record["target"]
    changes = np.zeros(len(t), dtype=bool)
    changes[1:] = np.any(np.diff(target, axis=0) != 0, axis=1)
    moving = np.flatnonzero(changes)
    if not len(moving):
        raise ValueError(f"No target changes in {path}")
    start = float(t[moving[0]])
    window = t >= start - 0.25
    elapsed = t[window] - start
    xyz_target = positions_xyz(target[window, :3], robot_root, -1)
    xyz_actual = positions_xyz(record["actual"][window, :3], robot_root, -1)
    if not (np.isfinite(xyz_target).all() and np.isfinite(xyz_actual).all()):
        raise ValueError(f"XYZ conversion failed in {path}")
    xyz_delta = xyz_target - xyz_actual
    xyz_distance = np.linalg.norm(xyz_delta, axis=1)
    commanded = elapsed >= 0
    base = record["base"]
    dt_ms = np.diff(t) * 1000
    report = {
        "file": path.name,
        "motion_start_s": start,
        "captured_motion_s": float(t[-1] - start),
        "xyz_error_rms_mm": float(np.sqrt(np.mean(xyz_distance[commanded] ** 2))),
        "xyz_error_p95_mm": float(np.percentile(xyz_distance[commanded], 95)),
        "xyz_error_max_mm": float(np.max(xyz_distance[commanded])),
        "axis_error_max_abs_counts": np.max(np.abs(record["error"][window][commanded]), axis=0).astype(int).tolist(),
        "cycle_p99_ms": float(np.percentile(dt_ms, 99)),
        "cycle_max_ms": float(np.max(dt_ms)),
        "wake_lateness_p99_us": float(np.percentile(base["wake_lateness_ns"] / 1000, 99)),
        "wake_lateness_max_us": float(np.max(base["wake_lateness_ns"] / 1000)),
        "bad_wkc_cycles": int(np.count_nonzero(base["wkc"] < 0)),
        "quick_stop_cycles": int(np.count_nonzero(base["quick_stop_requested"] != 0)),
    }
    return dict(record=record, elapsed=elapsed, window=window,
                xyz_target=xyz_target, xyz_actual=xyz_actual,
                xyz_delta=xyz_delta, xyz_distance=xyz_distance,
                start=start, report=report)


def padded_limits(values: np.ndarray, padding: float = 0.06) -> tuple[float, float]:
    low, high = float(np.min(values)), float(np.max(values))
    margin = max((high - low) * padding, 0.01)
    return low - margin, high + margin


def draw_main(runs: list[dict], output: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.grid": True, "grid.alpha": 0.22})
    fig, axes = plt.subplots(4, 2, figsize=(16, 16), layout="constrained")
    fig.suptitle("CSP flight watchdog — сравнение с одинаковыми масштабами\n"
                 "Время = 0 при первом изменении задания", fontsize=15)
    names = [r["report"]["file"].removeprefix("csp_flight_watchdog_").removesuffix(".csv") for r in runs]
    all_xy = np.concatenate([
        np.vstack((r["xyz_target"][:, :2], r["xyz_actual"][:, :2]))
        - r["xyz_target"][np.searchsorted(r["elapsed"], 0), :2]
        for r in runs
    ])
    xlim, ylim = padded_limits(all_xy[:, 0]), padded_limits(all_xy[:, 1])
    time_limits = (-0.25, max(float(r["elapsed"][-1]) for r in runs) + 0.05)

    for col, r in enumerate(runs):
        t = r["elapsed"]
        rec = r["record"]
        title = names[col]
        origin = r["xyz_target"][np.searchsorted(t, 0), :2]
        target_xy = r["xyz_target"][:, :2] - origin
        actual_xy = r["xyz_actual"][:, :2] - origin
        ax = axes[0, col]
        ax.plot(target_xy[:, 0], target_xy[:, 1], "k--", lw=1.4, label="Задание")
        ax.plot(actual_xy[:, 0], actual_xy[:, 1], color="tab:blue", lw=1, label="Факт")
        ax.scatter([0], [0], color="black", s=18, zorder=4)
        ax.set(xlim=xlim, ylim=ylim, xlabel="ΔX, мм", ylabel="ΔY, мм",
               title=f"{title} · траектория XY")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="best", fontsize=8)

        ax = axes[1, col]
        for i, color in enumerate(("tab:blue", "tab:orange", "tab:green")):
            ax.plot(t, r["xyz_delta"][:, i], color=color, lw=0.8,
                    label=("ΔX", "ΔY", "ΔZ")[i])
        ax.plot(t, r["xyz_distance"], color="black", lw=1.2, label="|ΔXYZ|")
        ax.set(title=f"{title} · ошибка XYZ", ylabel="мм")
        ax.legend(ncol=4, fontsize=8)

        ax = axes[2, col]
        for i, color in enumerate(("tab:blue", "tab:orange", "tab:green", "tab:purple")):
            ax.plot(t, rec["error"][r["window"], i] / 1000, color=color,
                    lw=0.8, label=f"M{i+1}")
        ax.set(title=f"{title} · ошибка осей", ylabel="тыс. отсчётов")
        ax.legend(ncol=4, fontsize=8)

        ax = axes[3, col]
        for i, color in enumerate(("tab:blue", "tab:orange", "tab:green", "tab:purple")):
            ax.plot(t, rec["velocity"][r["window"], i] / 1000, color=color,
                    lw=0.8, label=f"M{i+1}")
        ax.set(title=f"{title} · фактическая скорость", ylabel="тыс. ед. CSV")
        ax.legend(ncol=4, fontsize=8)

    for row in range(1, 4):
        limits = padded_limits(np.concatenate([line.get_ydata() for ax in axes[row] for line in ax.lines]))
        for ax in axes[row]:
            ax.set_ylim(limits)
            ax.set_xlim(time_limits)
            ax.axvline(0, color="0.4", ls="--", lw=0.8)
            ax.set_xlabel("Время от начала движения, с")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def draw_timing(runs: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(16, 9), sharex=True, layout="constrained")
    fig.suptitle("CSP flight watchdog — тайминг · общие шкалы", fontsize=15)
    names = [r["report"]["file"].removeprefix("csp_flight_watchdog_").removesuffix(".csv") for r in runs]
    for col, r in enumerate(runs):
        rec = r["record"]
        t = rec["t"] - r["start"]
        base = rec["base"]
        axes[0, col].plot(t[1:], np.diff(rec["t"]) * 1000, lw=0.55, color="tab:blue")
        axes[0, col].axhline(1, ls="--", color="black", lw=0.8)
        axes[0, col].set(title=f"{names[col]} · период цикла", ylabel="мс")
        axes[1, col].plot(t, base["wake_lateness_ns"] / 1000, lw=0.55, color="tab:orange")
        axes[1, col].set(title="Опоздание пробуждения", ylabel="мкс")
        axes[2, col].plot(t, base["dc_phase_error_ns"] / 1000, lw=0.55, color="tab:green")
        axes[2, col].set(title="Ошибка фазы DC", ylabel="мкс", xlabel="Время от начала движения, с")
        for row in range(3):
            axes[row, col].axvline(0, ls="--", color="0.35", lw=0.8)
            axes[row, col].grid(alpha=0.22)
    for row in range(3):
        limits = padded_limits(np.concatenate([line.get_ydata() for ax in axes[row] for line in ax.lines if len(line.get_ydata()) > 2]))
        for ax in axes[row]:
            ax.set_ylim(limits)
    axes[0, 0].set_xlim(min(float(r["record"]["t"][0] - r["start"]) for r in runs),
                        max(float(r["record"]["t"][-1] - r["start"]) for r in runs))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs=2, type=Path)
    from plot_delta_csv import default_robot_root

    parser.add_argument("--robot-root", type=Path,
                        default=default_robot_root())
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "plots")
    args = parser.parse_args()
    positions_xyz, read_watchdog = load_helpers(args.robot_root)
    runs = [prepare(p, args.robot_root, positions_xyz, read_watchdog) for p in args.csv]
    suffix = "_vs_".join(p.stem.removeprefix("csp_flight_watchdog_") for p in args.csv)
    main_output = args.output_dir / f"csp_flight_{suffix}_comparison.png"
    timing_output = args.output_dir / f"csp_flight_{suffix}_timing.png"
    stats_output = args.output_dir / f"csp_flight_{suffix}_stats.json"
    draw_main(runs, main_output)
    draw_timing(runs, timing_output)
    stats_output.write_text(json.dumps([r["report"] for r in runs], ensure_ascii=False, indent=2), encoding="utf-8")
    print(main_output)
    print(timing_output)
    print(stats_output)
    print(stats_output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

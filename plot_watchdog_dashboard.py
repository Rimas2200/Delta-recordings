"""Графики записи CSP flight watchdog."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_delta_csv import default_robot_root, positions_xyz


DEFAULT_CSV = "csp_flight_watchdog_20261005_115227.csv"
COLORS = ("tab:blue", "tab:orange", "tab:green", "tab:purple")


def read_watchdog(path: Path) -> dict:
    data = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True,
                                       dtype=None, encoding="utf-8"))
    required = {
        "sequence", "monotonic_ns", "axis_slot", "target", "actual", "error",
        "actual_velocity", "actual_torque", "gcode", "ring_pending", "wkc",
        "dc_phase_error_ns", "wake_lateness_ns", "statusword",
        "quick_stop_requested", "drive_enabled", "motion_ready",
    }
    missing = required - set(data.dtype.names or ())
    if missing:
        raise ValueError(f"В CSV отсутствуют столбцы: {', '.join(sorted(missing))}")
    slots = sorted(int(slot) for slot in np.unique(data["axis_slot"]))
    if slots[:3] != [0, 1, 2]:
        raise ValueError("Для XYZ нужны оси с axis_slot 0, 1 и 2")
    axes = [np.sort(data[data["axis_slot"] == slot], order="sequence") for slot in slots]
    sequence = axes[0]["sequence"]
    if any(len(axis) != len(sequence) or not np.array_equal(axis["sequence"], sequence)
           for axis in axes[1:]):
        raise ValueError("Строки разных осей не совпадают по sequence")
    base = axes[0]
    t = (base["monotonic_ns"].astype(np.int64) - int(base["monotonic_ns"][0])) / 1e9
    target = np.column_stack([axis["target"] for axis in axes]).astype(float)
    actual = np.column_stack([axis["actual"] for axis in axes]).astype(float)
    error = np.column_stack([axis["error"] for axis in axes]).astype(float)
    if not np.array_equal(error, target - actual):
        raise ValueError("Столбец error не совпадает с target - actual")
    velocity = np.column_stack([axis["actual_velocity"] for axis in axes]).astype(float)
    torque = np.column_stack([axis["actual_torque"] for axis in axes]).astype(float)
    return dict(t=t, axes=axes, base=base, target=target, actual=actual,
                error=error, velocity=velocity, torque=torque)


def find_motion_window(record: dict, before: float, after: float) -> tuple[float, float, float | None, float | None]:
    t, target = record["t"], record["target"]
    changes = np.zeros(len(t), dtype=bool)
    changes[1:] = np.any(np.diff(target, axis=0) != 0, axis=1)
    active = np.flatnonzero(changes)
    if len(active) == 0:
        return float(t[0]), float(t[-1]), None, None
    motion_start, motion_end = float(t[active[0]]), float(t[active[-1]])
    start = max(float(t[0]), motion_start - before)
    end = min(float(t[-1]), motion_end + after)
    return start, end, motion_start, motion_end


def shade_command(ax, t: np.ndarray, command: np.ndarray):
    changes = np.diff(np.r_[False, command.astype(bool), False].astype(int))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    for first, last in zip(starts, ends):
        right = t[min(last, len(t) - 1)]
        ax.axvspan(t[first], right, color="tab:green", alpha=0.08, lw=0)


def draw_dashboard(record: dict, source: Path, output: Path, robot_root: Path,
                   before: float, after: float):
    t = record["t"]
    base = record["base"]
    target, actual = record["target"], record["actual"]
    error, velocity, torque = record["error"], record["velocity"], record["torque"]
    start, end, motion_start, motion_end = find_motion_window(record, before, after)
    zoom = (t >= start) & (t <= end)
    if zoom.sum() < 3:
        raise ValueError("В выбранном окне меньше трёх циклов")
    command = base["gcode"] != 0
    xyz_target = positions_xyz(target[:, :3], robot_root, -1)
    xyz_actual = positions_xyz(actual[:, :3], robot_root, -1)
    xyz_error = xyz_target - xyz_actual
    distance = np.linalg.norm(xyz_error, axis=1)
    if not np.all(np.isfinite(xyz_target[zoom])) or not np.all(np.isfinite(xyz_actual[zoom])):
        raise ValueError("Кинематика не рассчитала XYZ на всём выбранном участке")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.grid": True, "grid.alpha": 0.22})
    fig, ax = plt.subplots(4, 2, figsize=(18, 15), layout="constrained")
    stamp = source.stem.removeprefix("csp_flight_watchdog_")
    clock = f"{stamp[9:11]}:{stamp[11:13]}:{stamp[13:15]}" if len(stamp) >= 15 else stamp
    if motion_start is None:
        motion_label = "Изменений задания нет"
    else:
        motion_label = f"Изменение задания {motion_start:.2f}–{motion_end:.2f} с"
    fig.suptitle(f"CSP flight watchdog · {clock}\n"
                 f"{motion_label} · окно графиков {start:.2f}–{end:.2f} с", fontsize=16)

    drive_colors = COLORS[:min(len(record["axes"]), len(COLORS))]
    first_in_window = np.flatnonzero(zoom)[0]
    origin = target[first_in_window, :len(drive_colors)]
    for i, color in enumerate(drive_colors):
        ax[0, 0].plot(t[zoom], (target[zoom, i] - origin[i]) / 1e6,
                      ls="--", lw=1.1, color=color, label=f"M{i+1} задание")
        ax[0, 0].plot(t[zoom], (actual[zoom, i] - origin[i]) / 1e6,
                      lw=0.9, color=color, label=f"M{i+1} факт")
    ax[0, 0].set_title("Положение осей: задание и факт")
    ax[0, 0].set_ylabel("От начала окна, млн отсчётов")
    ax[0, 0].legend(fontsize=7, ncol=2, loc="best")

    ax[0, 1].plot(xyz_target[zoom, 0], xyz_target[zoom, 1], "k--", lw=1.3,
                  label="Задание XY")
    ax[0, 1].plot(xyz_actual[zoom, 0], xyz_actual[zoom, 1], color="tab:blue",
                  lw=1.0, label="Фактически XY")
    ax[0, 1].scatter([xyz_target[first_in_window, 0]], [xyz_target[first_in_window, 1]],
                     s=30, color="black", zorder=4, label="Начало окна")
    ax[0, 1].set_title("Траектория XY · X показан крупнее")
    ax[0, 1].set_xlabel("X, мм")
    ax[0, 1].set_ylabel("Y, мм")
    x_values = np.r_[xyz_target[zoom, 0], xyz_actual[zoom, 0]]
    x_margin = max(1.0, 0.1 * np.ptp(x_values))
    ax[0, 1].set_xlim(float(np.min(x_values) - x_margin),
                     float(np.max(x_values) + x_margin))
    ax[0, 1].text(0.02, 0.03, "Масштабы X и Y различаются",
                  transform=ax[0, 1].transAxes, fontsize=8,
                  bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9})
    ax[0, 1].legend(fontsize=8)

    for i, color in enumerate(drive_colors):
        ax[1, 0].plot(t[zoom], error[zoom, i] / 1000,
                      lw=0.9, color=color, label=f"M{i+1}")
    ax[1, 0].axhline(0, color="0.4", lw=0.8)
    ax[1, 0].set_title("Ошибка слежения: target − actual")
    ax[1, 0].set_ylabel("Тысяч отсчётов")
    ax[1, 0].legend(fontsize=8)

    for i, color in enumerate(COLORS[:3]):
        ax[1, 1].plot(t[zoom], xyz_error[zoom, i], lw=0.9,
                      color=color, label=("ΔX", "ΔY", "ΔZ")[i])
    ax[1, 1].plot(t[zoom], distance[zoom], color="black", lw=1.1,
                  label="|ΔXYZ|")
    ax[1, 1].axhline(0, color="0.4", lw=0.8)
    ax[1, 1].set_title("Расхождение задания и факта в XYZ")
    ax[1, 1].set_ylabel("мм")
    ax[1, 1].legend(fontsize=8, ncol=4)

    for i, color in enumerate(drive_colors):
        ax[2, 0].plot(t[zoom], velocity[zoom, i], lw=0.8,
                      color=color, label=f"M{i+1}")
        ax[2, 1].plot(t[zoom], torque[zoom, i], lw=0.65,
                      color=color, alpha=0.85, label=f"M{i+1}")
    ax[2, 0].set_title("Фактические скорости")
    ax[2, 0].set_ylabel("Единицы CSV")
    ax[2, 1].set_title("Фактические моменты")
    ax[2, 1].set_ylabel("Единицы 0x6077")
    ax[2, 0].legend(fontsize=8)
    ax[2, 1].legend(fontsize=8)

    dt_ms = np.r_[np.nan, np.diff(t) * 1000]
    ax[3, 0].plot(t, dt_ms, color="tab:blue", lw=0.6, label="Период цикла, мс")
    ax[3, 0].axhline(1, color="0.35", lw=0.8, ls="--")
    delay_ax = ax[3, 0].twinx()
    delay_ax.plot(t, base["wake_lateness_ns"] / 1000,
                  color="tab:orange", lw=0.55, alpha=0.8, label="Опоздание, мкс")
    delay_ax.plot(t, base["dc_phase_error_ns"] / 1000,
                  color="tab:green", lw=0.55, alpha=0.8, label="Ошибка DC, мкс")
    delay_ax.set_ylabel("мкс")
    ax[3, 0].set_title("Период цикла и задержки · вся запись")
    ax[3, 0].set_ylabel("мс")
    handles1, labels1 = ax[3, 0].get_legend_handles_labels()
    handles2, labels2 = delay_ax.get_legend_handles_labels()
    ax[3, 0].legend(handles1 + handles2, labels1 + labels2, fontsize=7, ncol=2)

    lanes = [
        ("G-code", base["gcode"] != 0, "tab:blue"),
        ("Буфер > 0", base["ring_pending"] > 0, "tab:orange"),
        ("M1 0x7237", record["axes"][0]["statusword"] == "0x7237", COLORS[0]),
        ("M2 0x7237", record["axes"][1]["statusword"] == "0x7237", COLORS[1]),
        ("M3 0x7237", record["axes"][2]["statusword"] == "0x7237", COLORS[2]),
    ]
    if len(record["axes"]) > 3:
        lanes.append(("M4 0x7237", record["axes"][3]["statusword"] == "0x7237", COLORS[3]))
    for i, (label, mask, color) in enumerate(lanes):
        ax[3, 1].fill_between(t, i, i + 0.7, where=mask,
                              color=color, alpha=0.8, step="post")
    bad_wkc = np.flatnonzero(base["wkc"] < 0)
    for j, index in enumerate(bad_wkc):
        ax[3, 1].axvline(t[index], color="tab:red", lw=1.2,
                         label="WKC < 0" if j == 0 else None)
    qs = np.flatnonzero(base["quick_stop_requested"] != 0)
    for j, index in enumerate(qs):
        ax[3, 1].axvline(t[index], color="black", lw=0.8,
                         label="Quick Stop" if j == 0 else None)
    ax[3, 1].set_yticks(np.arange(len(lanes)) + 0.35,
                        [label for label, _, _ in lanes])
    ax[3, 1].set_ylim(len(lanes), 0)
    ax[3, 1].set_title("События и состояния · вся запись")
    if len(record["axes"]) > 3:
        m4_error = np.max(np.abs(error[:, 3]))
        ax[3, 1].text(0.99, 0.02, f"M4: max |error|={m4_error:.0f} отсчётов",
                      transform=ax[3, 1].transAxes, ha="right", va="bottom",
                      bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"})
    if len(bad_wkc) or len(qs):
        ax[3, 1].legend(fontsize=7, loc="upper left")

    for row in range(3):
        for column in range(2):
            if row != 0 or column != 1:
                ax[row, column].set_xlim(start, end)
                shade_command(ax[row, column], t, command)
            if row > 0 or column == 0:
                ax[row, column].set_xlabel("Время от начала CSV, с")
    for column in range(2):
        ax[3, column].axvspan(start, end, color="tab:blue", alpha=0.06)
        ax[3, column].set_xlim(float(t[0]), float(t[-1]))
        ax[3, column].set_xlabel("Время от начала CSV, с")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {output}")
    if motion_start is None:
        motion_report = "изменений задания нет"
    else:
        motion_report = f"изменение задания {motion_start:.3f}–{motion_end:.3f} с"
    print(f"{motion_report}; максимальная ошибка XYZ в окне: "
          f"{np.max(distance[zoom]):.2f} мм")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent / DEFAULT_CSV)
    parser.add_argument("--output", type=Path,
                        help="Путь выходного PNG (по умолчанию папка plots рядом с CSV)")
    parser.add_argument("--robot-root", type=Path,
                        default=default_robot_root())
    parser.add_argument("--before", type=float, default=0.25,
                        help="Запас времени до движения, с")
    parser.add_argument("--after", type=float, default=1.5,
                        help="Запас времени после движения, с")
    args = parser.parse_args()
    if args.before < 0 or args.after < 0:
        parser.error("--before и --after не могут быть отрицательными")
    output = args.output or args.csv.parent / "plots" / (args.csv.stem + "_dashboard.png")
    draw_dashboard(read_watchdog(args.csv), args.csv, output,
                   args.robot_root, args.before, args.after)


if __name__ == "__main__":
    main()

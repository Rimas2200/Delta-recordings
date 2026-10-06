from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_delta_csv import default_robot_root, positions_xyz, read_recording


REFERENCE = {
    "171908": (500.5, -582.2),
    "172026": (500.5, -502.1),
    "172212": (1424.2, -589.1),
}
DEFAULT_FILES = (
    "delta_encoders_20261005_171908_257540.csv",
    "delta_encoders_20261005_172026_240388.csv",
    "delta_encoders_20261005_172212_932026.csv",
)


def quick_stop_time(path: Path, motion_start: float) -> float | None:
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            t = int(row["elapsed_ns"]) / 1e9
            if t > motion_start and row["M1_servo_on"] == "0":
                return t
    return None


def corrected_circle_z(xyz: np.ndarray, moving: np.ndarray, complete: bool,
                       before_motion: np.ndarray) -> np.ndarray:
    z = xyz[:, 2]
    if complete:
        x, y = xyz[:, 0] / 400, xyz[:, 1] / 400
        features = np.column_stack((np.ones(len(x)), x, y, x * x, x * y, y * y))
        indices = np.flatnonzero(moving & np.isfinite(z))[::10]
        if len(indices) < 100:
            raise ValueError("Недостаточно точек для коррекции Z на полном круге")

        coefficients = np.linalg.lstsq(features[indices], z[indices], rcond=None)[0]
        z = z - features @ coefficients
    else:
        z = z.copy()
    return z - np.median(z[before_motion])


def load_circle(path: Path, robot_root: Path):
    t, pos, vel, torque = read_recording(path)
    xyz = positions_xyz(pos, robot_root, -1)
    speed = np.max(np.abs(vel), axis=1)
    moving = speed > 50
    if not moving.any():
        raise ValueError(f"Не найден участок движения: {path}")
    motion_start = float(t[moving][0])
    motion_end = float(t[moving][-1])
    quick_stop = quick_stop_time(path, motion_start)
    complete = (
        quick_stop is None
        and np.linalg.norm(xyz[moving][-1, :2] - xyz[moving][0, :2]) < 10
    )
    before_motion = t < motion_start - 0.1
    z_delta = corrected_circle_z(xyz, moving & (t < (quick_stop or motion_end)),
                                 complete, before_motion)
    time_zero = max(0.0, motion_start - 0.1)
    display_end = min(float(t[-1]), (quick_stop or motion_end) + 0.7)
    display = (t >= time_zero) & (t <= display_end)
    before_stop = display & (t <= (quick_stop or motion_end))
    span = float(np.ptp(z_delta[before_stop]))
    return dict(path=path, t=t, xyz=xyz, torque=torque, z_delta=z_delta,
                quick_stop=quick_stop, complete=complete, time_zero=time_zero,
                display=display, before_stop=before_stop, span=span,
                motion_start=motion_start, display_end=display_end)


def draw_circle_column(axes, run: dict, radius: float):
    path, t, xyz = run["path"], run["t"], run["xyz"]
    rel_t = t - run["time_zero"]
    display = run["display"]
    pre_stop = run["before_stop"]
    quick_stop = run["quick_stop"]
    parts = path.stem.split("_")
    stamp = parts[3] if len(parts) > 3 else ""
    if len(stamp) >= 6:
        clock = f"{stamp[:2]}:{stamp[2:4]}:{stamp[4:6]}"
    else:
        clock = path.stem
    speed, target_z = REFERENCE.get(stamp, (None, None))
    top, middle, bottom = axes
    if speed is not None:
        top.set_title(f"{clock} · {speed:g} мм/с\nZ задания {target_z:g} мм · R={radius:g} мм", fontsize=12)
    else:
        top.set_title(f"{clock} · R={radius:g} мм", fontsize=12)

    start_idx = np.flatnonzero(pre_stop)[0]
    end_idx = np.flatnonzero(pre_stop)[-1]
    angle = np.unwrap(np.arctan2(xyz[:, 1], xyz[:, 0]))
    arc_end = angle[end_idx] if quick_stop is not None else angle[start_idx] + 2 * np.pi
    phi = np.linspace(angle[start_idx], arc_end, 500)
    top.plot(radius * np.cos(phi), radius * np.sin(phi), "k--", lw=1,
             label="Задание R (по образцу)")
    top.plot(xyz[pre_stop, 0], xyz[pre_stop, 1], color="tab:blue", lw=1.1,
             label="По энкодерам")
    if quick_stop is not None:
        after_stop = display & (t >= quick_stop)
        top.plot(xyz[after_stop, 0], xyz[after_stop, 1], color="0.5", lw=1.1,
                 label="После Quick Stop")
    top.set_aspect("equal", adjustable="box")
    top.set_xlim(-420, 420)
    top.set_ylim(-420, 420)
    top.set_xlabel("X, мм")
    top.set_ylabel("Y, мм")
    top.legend(loc="center", fontsize=8)

    middle.plot(rel_t[display], run["z_delta"][display], color="tab:blue", lw=0.95)
    middle.axhline(0, color="0.3", linestyle="--", lw=0.9)
    middle.set_title(f"Размах до остановки: {run['span']:.3f} мм", fontsize=12)
    middle.set_ylabel("ΔZ от исходного положения, мм")
    middle.set_xlabel("Время от начала движения, с")
    middle.set_xlim(0, run["display_end"] - run["time_zero"])
    if quick_stop is not None:
        qs_rel = quick_stop - run["time_zero"]
        middle.axvspan(qs_rel, run["display_end"] - run["time_zero"],
                       color="0.85", alpha=0.65, label="После Quick Stop")
        middle.legend(loc="upper left", fontsize=8)

    for axis, color in enumerate(("tab:blue", "tab:orange", "tab:green")):
        bottom.plot(rel_t[display], run["torque"][display, axis], color=color,
                    lw=0.55, alpha=0.8, label=f"M{axis+1}")
    if quick_stop is not None:
        bottom.axvline(quick_stop - run["time_zero"], color="k", ls="--",
                       lw=1, label="Quick Stop")
    bottom.set_title("Моменты приводов на круге", fontsize=12)
    bottom.set_ylabel("Момент, единицы 0x6077")
    bottom.set_xlabel("Время от начала движения, с")
    bottom.set_xlim(0, run["display_end"] - run["time_zero"])
    bottom.legend(loc="upper right", fontsize=8, ncol=4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path,
                        help="Три delta_encoders_*.csv")
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--robot-root", type=Path,
                        default=default_robot_root())
    parser.add_argument("--radius", type=float, default=380.0,
                        help="Радиус задания из образца, мм")
    parser.add_argument("--output", type=Path,
                        help="Путь выходного PNG (по умолчанию папка plots внутри --input-dir)")
    args = parser.parse_args()
    files = args.files or [args.input_dir / name for name in DEFAULT_FILES]
    if len(files) != 3:
        parser.error("Нужно указать ровно три CSV")
    if args.radius <= 0:
        parser.error("Радиус должен быть положительным")
    output = args.output or args.input_dir / "plots" / "circular_trajectories_20261005.png"
    runs = [load_circle(path, args.robot_root) for path in files]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.grid": True, "grid.alpha": 0.2})
    fig, axes = plt.subplots(3, 3, figsize=(17, 12), layout="constrained")
    fig.suptitle("Круговые траектории: три скорости и высоты\nXYZ рассчитаны по энкодерам", fontsize=17)
    for column, run in enumerate(runs):
        draw_circle_column(axes[:, column], run, args.radius)
        print(f"{run['path'].name}: dZ {run['span']:.3f} мм; "
              f"Quick Stop: {run['quick_stop'] if run['quick_stop'] is not None else 'нет'}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {output}")


if __name__ == "__main__":
    main()

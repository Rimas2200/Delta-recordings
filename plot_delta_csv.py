from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np


def default_robot_root() -> Path:
    """Find Delta from either a sibling checkout or a submodule checkout."""
    parent = Path(__file__).resolve().parent.parent
    for candidate in (parent, parent / "Delta"):
        if (candidate / "robot.config").is_file() and (candidate / "Delta" / "robot_math_adapter.py").is_file():
            return candidate
    return parent / "Delta"


def read_recording(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    required = {"elapsed_ns", "valid"} | {
        f"M{i}_{field}" for i in (1, 2, 3) for field in ("actual", "velocity", "torque")
    }
    missing = required - set(data.dtype.names or ())
    if missing:
        raise ValueError(f"В CSV отсутствуют столбцы: {', '.join(sorted(missing))}")
    data = np.atleast_1d(data)
    good = data["valid"] == 1
    if good.sum() < 3:
        raise ValueError("В файле меньше трёх корректных записей")
    t = data["elapsed_ns"][good].astype(float) / 1e9
    pos = np.column_stack([data[f"M{i}_actual"][good] for i in (1, 2, 3)]).astype(float)
    vel = np.column_stack([data[f"M{i}_velocity"][good] for i in (1, 2, 3)]).astype(float)
    torque = np.column_stack([data[f"M{i}_torque"][good] for i in (1, 2, 3)]).astype(float)
    order = np.argsort(t)
    return t[order], pos[order], vel[order], torque[order]


def all_servos_off(path: Path) -> bool:
    """Проверяет, были ли все три привода отключены во всех корректных строках."""
    valid_rows = 0
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["valid"] != "1":
                continue
            valid_rows += 1
            if any(row[f"M{i}_servo_on"] != "0" for i in (1, 2, 3)):
                return False
    return valid_rows > 0


def positions_xyz(pos: np.ndarray, robot_root: Path, encoder_direction: int) -> np.ndarray:
    """Переводит отсчёты 6064 в XYZ через RobotMath."""
    adapter_dir = robot_root / "Delta"
    if not (adapter_dir / "robot_math_adapter.py").is_file():
        raise FileNotFoundError(f"Не найден robot_math_adapter.py в {adapter_dir}")
    sys.path.insert(0, str(adapter_dir))
    from robot_math_adapter import RobotMath

    robot = RobotMath()
    config = robot_root / "robot.config"
    zero_line = next(
        (line for line in config.read_text(encoding="utf-8").splitlines()
         if "#zeroes" in line),
        None,
    )
    if zero_line is None:
        raise ValueError(f"Не найдена строка #zeroes в {config}")
    zeros = np.asarray([int(v) for v in zero_line.split("#")[0].split()[:3]])
    if zeros.shape != (3,):
        raise ValueError("В robot.config нужны нули трёх осей")
    # У записей знак осей обратен калибровке адаптера.
    adapted = zeros + encoder_direction * (pos - zeros)
    xyz = np.full((len(pos), 3), np.nan)
    for i, counts in enumerate(adapted):
        try:
            xyz[i] = np.asarray(robot.position_from_units(tuple(map(int, counts)))) * 1000
        except ValueError:
            pass
    # XY в записях повёрнуты относительно координат RobotMath.
    xyz[:, :2] = xyz[:, [1, 0]] * np.array([1, -1])
    return xyz


def auto_zoom(t: np.ndarray, vel: np.ndarray, width: float) -> tuple[float, float]:
    speed = np.max(np.abs(vel), axis=1)
    threshold = max(20.0, 0.08 * np.percentile(speed, 99))
    active = np.flatnonzero(speed > threshold)
    center = float(t[active[-1]]) if len(active) else float(t[np.argmax(speed)])
    return max(float(t[0]), center - width / 2), min(float(t[-1]), center + width / 2)


def colored_path(ax, x: np.ndarray, y: np.ndarray, color: np.ndarray,
                 label: tuple[str, str]):
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(color)
    points = np.column_stack((x[keep], y[keep]))
    if len(points) < 2:
        raise ValueError("Для траектории осталось меньше двух достижимых точек")
    segments = np.stack((points[:-1], points[1:]), axis=1)
    line = LineCollection(segments, cmap="coolwarm", linewidth=1.7, alpha=0.9)
    line.set_array(color[keep][:-1])
    ax.add_collection(line)
    ax.autoscale_view()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(label[0])
    ax.set_ylabel(label[1])
    return line


def spatial_z_residual(xyz: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """Убирает зависимость Z от XY при постоянном задании высоты."""
    x, y, z = xyz.T
    x, y = x / 400, y / 400
    features = np.column_stack((np.ones(len(x)), x, y, x * x, x * y, y * y,
                                x * x * x, x * x * y, x * y * y, y * y * y))
    moving = np.max(np.abs(vel), axis=1) > 50
    keep = moving & np.all(np.isfinite(xyz), axis=1)
    if keep.sum() < 100:
        raise ValueError("Для --flat-z недостаточно точек движения")
    index = np.flatnonzero(keep)[::max(1, keep.sum() // 3000)]
    coefficients = np.linalg.lstsq(features[index], z[index], rcond=None)[0]
    return z - features @ coefficients


def plot(t, pos, vel, torque, source: Path, output: Path, xyz: np.ndarray | None,
         zoom: tuple[float, float] | None, zoom_width: float, z_reference: float | None,
         flat_z: bool, torque_panel: bool, servos_off: bool):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.grid": True,
                         "grid.alpha": 0.22, "figure.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), layout="constrained")
    stamp = source.stem.removeprefix("delta_encoders_").split("_")
    clock = f"{stamp[1][:2]}:{stamp[1][2:4]}:{stamp[1][4:6]}" if len(stamp) > 1 else source.stem
    if servos_off:
        title = f"{clock} - перемещение при выключенных приводах"
    else:
        title = f"{clock} - траектория и колебания по энкодерам"
    fig.suptitle(title, fontsize=17)
    axy, afull, adetail, aspeed = axes.flat
    colors = ["tab:blue", "tab:orange", "tab:green"]
    if xyz is None:
        values = (pos - pos[0]) / 1e6
        line = colored_path(axy, values[:, 0], values[:, 1], t,
                            ("ΔM1, млн отсчётов", "ΔM2, млн отсчётов"))
        axy.set_title("Траектория в координатах энкодеров")
        fig.colorbar(line, ax=axy, label="Время записи, с", shrink=0.85)
        for i, color in enumerate(colors):
            afull.plot(t, values[:, i], lw=0.8, color=color, label=f"M{i+1}")
        afull.set_title("Перемещение трёх осей")
        afull.set_ylabel("От начального положения, млн отсчётов")
        afull.legend(loc="best")
    else:
        if flat_z:
            z_plot = spatial_z_residual(xyz, vel)
            z_label = "ΔZ после пространственной коррекции, мм"
        else:
            z_plot = xyz[:, 2] - z_reference if z_reference is not None else xyz[:, 2]
            z_label = "Z − опорная высота, мм" if z_reference is not None else "Z, мм"
        line = colored_path(axy, xyz[:, 0], xyz[:, 1], z_plot, ("X, мм", "Y, мм"))
        axy.set_title("Расчётный XY по энкодерам" if servos_off else
                      "Траектория XY по энкодерам")
        fig.colorbar(line, ax=axy, label=z_label, shrink=0.85)
        afull.plot(t, z_plot, lw=0.8, color=colors[0])
        afull.set_title("Отклонение высоты" if flat_z else "Высота Z по энкодерам")
        afull.set_ylabel(z_label)
        afull.axhline(0, color="0.4", lw=0.7)
        if servos_off:
            note = "M1–M3: servo_on=0 весь файл"
            if not np.any(torque):
                note += "; моменты=0"
            afull.text(0.02, 0.96, note, transform=afull.transAxes, va="top",
                       bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.9})
        if np.isnan(xyz[:, 2]).any():
            print(f"Недостижимых точек для кинематики проекта: {np.isnan(xyz[:, 2]).sum()}")
    afull.set_xlabel("Время от начала CSV, с")

    start, end = zoom if zoom is not None else auto_zoom(t, vel, zoom_width)
    if not t[0] <= start < end <= t[-1]:
        raise ValueError("Интервал --zoom должен находиться внутри записи")
    window = (t >= start) & (t <= end)
    if window.sum() < 3:
        raise ValueError("В интервале --zoom меньше трёх точек")
    if xyz is None:
        local = (pos[window] - pos[np.flatnonzero(window)[0]]) / 1e3
        for i, color in enumerate(colors):
            adetail.plot(t[window], local[:, i], lw=1.1, color=color, label=f"M{i+1}")
        adetail.set_ylabel("От начала окна, тыс. отсчётов")
        adetail.legend(loc="best")
        adetail.set_title("Детально: перемещение осей")
    else:
        adetail.plot(t[window], z_plot[window], lw=1.1, color=colors[0], label="Z")
        adetail.set_ylabel(z_label)
        adetail.set_title("Последний участок: высота Z" if servos_off else
                          "Детально: высота по энкодерам")
    adetail.set_xlabel("Время от начала CSV, с")
    if torque_panel:
        aspeed.plot(t[window], torque[window, 2], lw=0.8, color="tab:purple", label="M3")
        aspeed.set_title("Момент M3 на том же участке")
        aspeed.set_ylabel("Момент, единицы CSV")
        aspeed.axhline(0, color="0.4", lw=0.7)
    else:
        for i, color in enumerate(colors):
            aspeed.plot(t[window], vel[window, i], lw=0.9, color=color, label=f"M{i+1}")
        aspeed.set_title("Скорости осей на том же участке")
        aspeed.set_ylabel("Скорость, единицы CSV")
    aspeed.set_xlabel("Время от начала CSV, с")
    aspeed.legend(loc="best")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Сохранено: {output}")
    print(f"Детальный участок: {start:.3f}–{end:.3f} с")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Исходный delta_encoders_*.csv")
    parser.add_argument("--output", type=Path,
                        help="Путь выходного PNG (по умолчанию папка plots рядом с CSV)")
    parser.add_argument("--robot-root", type=Path,
                        default=default_robot_root(),
                        help="Папка проекта delta с Delta/robot_math_adapter.py")
    parser.add_argument("--raw", action="store_true", help="Графики только в отсчётах энкодеров")
    parser.add_argument("--encoder-direction", type=int, choices=(-1, 1), default=-1,
                        help="Знак энкодера относительно адаптера robot_math (по умолчанию -1)")
    parser.add_argument("--z-reference", type=float,
                        help="Вычесть эту опорную высоту, мм, из графиков Z")
    parser.add_argument("--flat-z", action="store_true",
                        help="Для постоянного задания Z: вычесть пространственный тренд, найденный по CSV")
    parser.add_argument("--torque-panel", action="store_true",
                        help="Показать момент M3 вместо скоростей на нижнем правом графике")
    parser.add_argument("--zoom", nargs=2, type=float, metavar=("START", "END"),
                        help="Участок для нижних графиков, секунды от начала CSV")
    parser.add_argument("--zoom-width", type=float, default=1.4,
                        help="Ширина автоматически выбранного участка в секундах")
    args = parser.parse_args()
    if args.zoom_width <= 0:
        parser.error("--zoom-width должен быть положительным")
    if args.flat_z and (args.raw or args.z_reference is not None):
        parser.error("--flat-z нельзя сочетать с --raw или --z-reference")
    t, pos, vel, torque = read_recording(args.csv)
    servos_off = all_servos_off(args.csv)
    if args.flat_z and servos_off:
        parser.error("--flat-z нельзя применять к записи с выключенными приводами")
    xyz = None if args.raw else positions_xyz(pos, args.robot_root, args.encoder_direction)
    output = args.output or args.csv.parent / "plots" / (args.csv.stem + "_plots.png")
    plot(t, pos, vel, torque, args.csv, output, xyz,
         tuple(args.zoom) if args.zoom else None, args.zoom_width, args.z_reference,
         args.flat_z, args.torque_panel, servos_off)


if __name__ == "__main__":
    main()

"""Замер быстродействия двух реализаций эрозии (п. 2 задания).

Эксперименты:

1. Время эрозии (ядро 3x3) в зависимости от размера изображения
   256x256 ... 2048x2048.
2. Время эрозии в зависимости от размера ядра 3x3 ... 9x9
   на изображении 512x512 (дополнительный эксперимент: по условию
   примитив фиксирован 3x3, график показывает масштабируемость).

Для каждой точки выполняется несколько прогонов, берётся медиана.
Перед замером проверяется попиксельное совпадение результатов обеих
реализаций, чтобы сравнение времени было корректным.

Результаты: ``results/benchmark.csv``, ``results/benchmark.png``.

Запуск (из каталога ``lab_1``)::

    py src/benchmark.py
"""

import csv
import os
import statistics
import time

import cv2
import numpy as np

from erosion_cv2 import erode_cv2
from erosion_native import erode_native
from generate_images import make_scene


def binary_scene(width, height, seed):
    """Синтетическая сцена, приведённая к бинарному виду методом Отсу."""
    image = make_scene(width, height, seed=seed)
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def measure(func, image, kernel, runs):
    """Медиана времени работы ``func`` на ``image`` за ``runs`` прогонов."""
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        func(image, kernel)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def main():
    os.makedirs("results", exist_ok=True)
    rows = []  # (эксперимент, размер изображения, размер ядра, native, cv2)

    print("Эксперимент 1: зависимость от размера изображения (ядро 3x3)")
    for size in (256, 512, 1024, 2048):
        binary = binary_scene(size, size, seed=42)
        kernel = np.ones((3, 3), dtype=np.uint8)

        assert np.array_equal(
            erode_native(binary, kernel), erode_cv2(binary, kernel)
        ), "Результаты нативной и библиотечной реализаций не совпадают"

        runs = 5 if size <= 512 else 3
        t_native = measure(erode_native, binary, kernel, runs)
        t_cv = measure(erode_cv2, binary, kernel, runs)
        rows.append(("image_size", size, 3, t_native, t_cv))
        print(
            f"  {size}x{size}: native {t_native:.4f} с, cv2 {t_cv:.6f} с, "
            f"ускорение {t_native / t_cv:.0f}x"
        )

    print("Эксперимент 2: зависимость от размера ядра (изображение 512x512)")
    for k in (3, 5, 7, 9):
        binary = binary_scene(512, 512, seed=43)
        kernel = np.ones((k, k), dtype=np.uint8)
        t_native = measure(erode_native, binary, kernel, 3)
        t_cv = measure(erode_cv2, binary, kernel, 5)
        rows.append(("kernel_size", 512, k, t_native, t_cv))
        print(
            f"  ядро {k}x{k}: native {t_native:.4f} с, cv2 {t_cv:.6f} с, "
            f"ускорение {t_native / t_cv:.0f}x"
        )

    csv_path = os.path.join("results", "benchmark.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["experiment", "image_size_px", "kernel_size_px", "native_s", "cv2_s", "speedup"]
        )
        for experiment, size, k, t_native, t_cv in rows:
            writer.writerow(
                [experiment, size, k, f"{t_native:.6f}", f"{t_cv:.6f}", f"{t_native / t_cv:.1f}"]
            )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    size_rows = [r for r in rows if r[0] == "image_size"]
    sizes = [r[1] for r in size_rows]
    axes[0].plot(sizes, [r[3] for r in size_rows], "o-", label="нативная (Python)")
    axes[0].plot(sizes, [r[4] for r in size_rows], "s-", label="cv2.erode")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xticks(sizes, [str(s) for s in sizes])
    axes[0].set_xlabel("Размер изображения, px")
    axes[0].set_ylabel("Время, с (медиана)")
    axes[0].set_title("Эрозия, ядро 3×3")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    kernel_rows = [r for r in rows if r[0] == "kernel_size"]
    ks = [r[2] for r in kernel_rows]
    axes[1].plot(ks, [r[3] for r in kernel_rows], "o-", label="нативная (Python)")
    axes[1].plot(ks, [r[4] for r in kernel_rows], "s-", label="cv2.erode")
    axes[1].set_yscale("log")
    axes[1].set_xticks(ks)
    axes[1].set_xlabel("Размер ядра, px")
    axes[1].set_ylabel("Время, с (медиана)")
    axes[1].set_title("Эрозия, изображение 512×512")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    fig.suptitle("Сравнение быстродействия реализаций эрозии")
    fig.tight_layout()
    plot_path = os.path.join("results", "benchmark.png")
    fig.savefig(plot_path, dpi=150)

    print(f"Сохранено: {csv_path}")
    print(f"Сохранено: {plot_path}")


if __name__ == "__main__":
    main()

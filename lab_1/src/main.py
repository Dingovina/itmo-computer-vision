"""CLI-точка входа ЛР1 (вариант 7 — эрозия).

Загружает изображение, бинаризует его методом Отсу (встроенная функция
OpenCV, по условию реализовывать вручную не требуется), применяет эрозию
двумя способами — нативно на Python и через ``cv2.erode`` — сохраняет
результаты и выводит время работы обеих реализаций.

Запуск (из каталога ``lab_1``)::

    py src/main.py images/sample_shapes.png [-o results] [-k 3]
"""

import argparse
import os
import time

import cv2
import numpy as np

from binarize import binarize_otsu
from erosion_cv2 import erode_cv2
from erosion_native import erode_native


def run(image_path, out_dir="results", kernel_size=3):
    """Обработать одно изображение и вывести тайминги.

    :param image_path: путь к входному изображению
    :param out_dir: каталог для сохранения результатов
    :param kernel_size: размер квадратного структурного элемента
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {image_path}")

    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(image_path))[0]

    binary = binarize_otsu(image)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    start = time.perf_counter()
    eroded_native = erode_native(binary, kernel)
    t_native = time.perf_counter() - start

    start = time.perf_counter()
    eroded_cv = erode_cv2(binary, kernel)
    t_cv = time.perf_counter() - start

    identical = bool(np.array_equal(eroded_native, eroded_cv))

    # Карта расхождений: чёрные точки там, где результаты различаются
    # (ожидается полностью белое изображение)
    diff = np.full(binary.shape, 255, dtype=np.uint8)
    diff[eroded_native != eroded_cv] = 0

    paths = {
        "binary": os.path.join(out_dir, f"{name}_binary.png"),
        "native": os.path.join(out_dir, f"{name}_eroded_native.png"),
        "cv2": os.path.join(out_dir, f"{name}_eroded_cv2.png"),
        "diff": os.path.join(out_dir, f"{name}_diff.png"),
    }
    cv2.imwrite(paths["binary"], binary)
    cv2.imwrite(paths["native"], eroded_native)
    cv2.imwrite(paths["cv2"], eroded_cv)
    cv2.imwrite(paths["diff"], diff)

    print(f"Изображение: {image_path} ({binary.shape[1]}x{binary.shape[0]})")
    print(f"Ядро: {kernel_size}x{kernel_size}")
    print(f"Время erode_native: {t_native:.4f} с")
    print(f"Время erode_cv2:    {t_cv:.6f} с")
    print(f"Ускорение cv2:      {t_native / max(t_cv, 1e-9):.1f}x")
    print(f"Результаты совпадают попиксельно: {identical}")
    for key, path in paths.items():
        print(f"  {key}: {path}")


def main():
    parser = argparse.ArgumentParser(description="ЛР1: эрозия (нативно + OpenCV)")
    parser.add_argument("image", help="путь к входному изображению")
    parser.add_argument("-o", "--out-dir", default="results", help="каталог для результатов")
    parser.add_argument("-k", "--kernel", type=int, default=3, help="размер ядра (по заданию 3)")
    args = parser.parse_args()
    run(args.image, args.out_dir, args.kernel)


if __name__ == "__main__":
    main()

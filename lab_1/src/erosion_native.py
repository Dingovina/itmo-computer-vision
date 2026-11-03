"""Нативная реализация эрозии на Python (вариант 7)"""

import numpy as np


def erode_native(image, kernel=None):
    """Эрозия изображения, реализованная циклами на Python.

    :param image: двумерный numpy.ndarray типа uint8
    :param kernel: структурный элемент — numpy.ndarray из 0/1 с нечётными
                   размерами; по умолчанию примитив 3x3 из единиц
    :return: результат эрозии (numpy.ndarray, uint8, тот же размер)
    """
    if kernel is None:
        kernel = np.ones((3, 3), dtype=np.uint8)
    kernel = np.asarray(kernel)
    if image.ndim != 2:
        raise ValueError("Ожидается двумерное (grayscale) изображение")
    if image.dtype != np.uint8:
        raise ValueError("Ожидается изображение типа uint8")
    if kernel.ndim != 2 or kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("Структурный элемент должен быть 2D с нечётными размерами")

    height, width = image.shape
    k_h, k_w = kernel.shape
    k_cy, k_cx = k_h // 2, k_w // 2

    # Смещения активных (ненулевых) точек структурного элемента
    offsets = [
        (dy - k_cy, dx - k_cx)
        for dy in range(k_h)
        for dx in range(k_w)
        if kernel[dy, dx]
    ]

    # tolist() даёт быстрый доступ к скалярам из чистого Python
    data = image.tolist()
    result = np.empty((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            minimum = 255
            for dy, dx in offsets:
                yy = y + dy
                xx = x + dx
                if 0 <= yy < height and 0 <= xx < width:
                    value = data[yy][xx]
                    if value < minimum:
                        minimum = value
            result[y, x] = minimum
    return result

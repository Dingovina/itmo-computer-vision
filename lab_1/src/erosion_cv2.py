"""Реализация эрозии с использованием встроенной функции OpenCV (вариант 7)."""

import cv2
import numpy as np


def erode_cv2(image, kernel=None):
    """Эрозия изображения через ``cv2.erode``.

    Интерфейс совпадает с ``erode_native`` из модуля erosion_native.

    :param image: двумерный numpy.ndarray типа uint8
    :param kernel: структурный элемент — numpy.ndarray из 0/1;
                   по умолчанию примитив 3x3 из единиц
    :return: результат эрозии (numpy.ndarray, uint8, тот же размер)
    """
    if kernel is None:
        kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.erode(image, kernel.astype(np.uint8))

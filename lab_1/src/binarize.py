"""Бинаризация изображения (предобработка для эрозии, вариант 7).

По условию лабораторной бинаризацию разрешено не реализовывать вручную,
поэтому используется встроенная функция OpenCV ``cv2.threshold`` с
автоматическим порогом по методу Отсу.
"""

import cv2


def binarize_otsu(image):
    """Бинаризовать изображение методом Отсу.

    :param image: изображение (numpy.ndarray), grayscale или BGR
    :return: бинарное изображение (numpy.ndarray, uint8, значения 0/255)
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

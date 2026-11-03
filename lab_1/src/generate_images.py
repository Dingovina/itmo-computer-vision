"""Генерация тестовых изображений для ЛР1.

Создаёт в каталоге ``images/`` синтетические изображения:

- ``sample_shapes.png`` — синтетическая «сцена»: градиент, крупные
  фигуры и мелкие точки с шумом; после бинаризации Отсу содержит
  объекты разного размера, что наглядно демонстрирует эффект эрозии
  (мелкие точки и шум исчезают, крупные объекты сжимаются);
- ``sample_noise.png`` — равномерный шум; после бинаризации Отсу
  превращается в зерно «salt & pepper».

Запуск (из каталога ``lab_1``)::

    py src/generate_images.py
"""

import os

import cv2
import numpy as np


def make_scene(width, height, seed=0):
    """Синтетическое grayscale-«фото»: градиент + фигуры + шум.

    :param width: ширина изображения
    :param height: высота изображения
    :param seed: зерно генератора случайных чисел
    :return: numpy.ndarray (uint8, высота x ширина)
    """
    rng = np.random.default_rng(seed)

    # Плавный диагональный градиент
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    image = (x[np.newaxis, :] + y[:, np.newaxis]) * 127.5

    min_side = min(width, height)
    cv2.circle(image, (width // 2, height // 2), min_side // 5, 255.0, -1)
    cv2.rectangle(
        image,
        (width // 10, height // 10),
        (width // 3, height // 3),
        30.0,
        -1,
    )
    cv2.rectangle(
        image,
        (int(width * 0.60), int(height * 0.70)),
        (int(width * 0.90), int(height * 0.95)),
        200.0,
        -1,
    )

    # Мелкие точки — имитация шума/деталей, которые эрозия должна убрать
    for _ in range(100):
        px = int(rng.integers(0, width))
        py = int(rng.integers(0, height))
        radius = max(1, min_side // 300)
        cv2.circle(image, (px, py), radius, float(rng.integers(0, 256)), -1)

    noisy = image + rng.normal(0.0, 15.0, image.shape).astype(np.float32)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def make_noise(width, height, seed=1):
    """Равномерный grayscale-шум.

    :return: numpy.ndarray (uint8, высота x ширина)
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (height, width), dtype=np.uint8)


def main():
    os.makedirs("images", exist_ok=True)
    cv2.imwrite(os.path.join("images", "sample_shapes.png"), make_scene(800, 600, seed=0))
    cv2.imwrite(os.path.join("images", "sample_noise.png"), make_noise(512, 512, seed=1))
    print("Создано: images/sample_shapes.png, images/sample_noise.png")


if __name__ == "__main__":
    main()

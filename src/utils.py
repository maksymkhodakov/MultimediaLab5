"""
Допоміжні функції, спільні для кількох скриптів лабораторної роботи.

Основні групи функцій:
    * download_demo_image - завантаження тестового зображення bus.jpg;
    * calculate_iou        - обчислення метрики IoU для двох боксів;
    * non_max_suppression  - власна (наївна) реалізація алгоритму NMS;
    * draw_boxes            - відображення bounding boxes на зображенні.

Функції винесені сюди, щоб не дублювати код у кожному скрипті окремо
та мати єдину, перевірену реалізацію базових алгоритмів.
"""

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import urllib.request

from config import DEMO_IMAGE_PATH, DEMO_IMAGE_URL, VIDEO_SOURCE_IMAGE_PATH, VIDEO_SOURCE_IMAGE_URL


def _download_if_missing(url: str, path: Path) -> Path:
    """Завантажує файл за посиланням url у path, якщо його ще немає локально."""
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    return path


def download_demo_image() -> Path:
    """Завантажує стандартне тестове зображення bus.jpg, якщо його ще немає локально.

    Зображення використовується в офіційній документації Ultralytics як
    приклад для демонстрації роботи YOLO і містить кілька людей та автобус,
    що зручно для перевірки детекції декількох класів об'єктів одночасно.
    """
    return _download_if_missing(DEMO_IMAGE_URL, DEMO_IMAGE_PATH)


def download_video_source_image() -> Path:
    """Завантажує зображення zidane.jpg, яке слугує джерелом для синтетичного
    відео (крок 5). Навмисно використовується інше зображення, ніж bus.jpg,
    щоб відеоряд не повторював кадри з попередніх кроків роботи.
    """
    return _download_if_missing(VIDEO_SOURCE_IMAGE_URL, VIDEO_SOURCE_IMAGE_PATH)


def calculate_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Обчислює IoU (Intersection over Union) для двох прямокутників.

    Бокси задаються у форматі [x1, y1, x2, y2] - координати лівого верхнього
    та правого нижнього кутів у пікселях.

    IoU = площа_перетину / площа_об'єднання

    Значення 1.0 означає ідеальний збіг боксів, 0.0 - бокси не перетинаються.
    Ця метрика лежить в основі як алгоритму NMS (порівняння дублікатів),
    так і обчислення mAP (порівняння передбачення з еталонною розміткою).
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # Координати прямокутника перетину (intersection)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    # Ширина/висота перетину не можуть бути від'ємними -
    # якщо бокси не перетинаються, площа перетину дорівнює 0
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def non_max_suppression(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5
) -> list[int]:
    """Власна (наївна) реалізація алгоритму Non-Maximum Suppression.

    Алгоритм прибирає дублікати bounding boxes, що вказують на один і той
    самий об'єкт, залишаючи лише один - найбільш впевнений - бокс у кожній
    групі перекриттів. Кроки алгоритму:

        1. Відсортувати всі бокси за впевненістю (confidence) за спаданням.
        2. Взяти бокс з найвищою впевненістю, додати його індекс у результат.
        3. Видалити з розгляду всі інші бокси, чий IoU з обраним боксом
           перевищує заданий поріг iou_threshold (вважаємо їх дублікатами).
        4. Повторювати кроки 2-3, поки не залишиться необроблених боксів.

    Повертає список індексів боксів, які потрібно залишити.
    """
    if len(boxes) == 0:
        return []

    # Індекси боксів, відсортовані за спаданням впевненості
    order = list(np.argsort(scores)[::-1])
    keep: list[int] = []

    while order:
        # Індекс боксу з найвищою впевненістю серед тих, що залишились
        # (приводимо np.int64 до звичайного int, щоб результат був сумісний з json.dumps)
        current = int(order.pop(0))
        keep.append(current)

        # Порівнюємо поточний бокс з рештою і залишаємо лише ті,
        # що НЕ є його дублікатами (IoU нижче порогу)
        remaining = []
        for idx in order:
            iou = calculate_iou(boxes[current], boxes[idx])
            if iou <= iou_threshold:
                remaining.append(idx)
        order = remaining

    return keep


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: Sequence[str],
    scores: Sequence[float] | None = None,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Малює прямокутники (bounding boxes) з підписами класів на копії зображення.

    Використовується там, де потрібна ручна (не вбудована в Ultralytics)
    візуалізація результатів - наприклад, для власної реалізації NMS.
    """
    img = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = labels[i]
        if scores is not None:
            label = f"{label} {scores[i]:.2f}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            img, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )
    return img

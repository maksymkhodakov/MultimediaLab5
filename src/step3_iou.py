"""
Крок 3. IoU (Intersection over Union) - перетин над об'єднанням.

IoU - базова метрика, яка показує, наскільки два прямокутники (наприклад,
передбачений моделлю бокс і еталонний (ground truth) бокс) збігаються:

    IoU = площа(A ∩ B) / площа(A ∪ B)

IoU = 1 означає ідеальний збіг, IoU = 0 - прямокутники не перетинаються
взагалі. У задачах детекції об'єктів детекцію зазвичай вважають
"правильною" (True Positive), якщо її IoU з еталонним боксом перевищує
певний поріг (типово 0.5).

Цей крок демонструє власну реалізацію IoU (utils.calculate_iou) на
трьох штучних прикладах боксів і візуалізує їх взаємне розташування.
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from config import DATA_DIR, PLOTS_DIR
from utils import calculate_iou


def draw_box(ax, box, color, label):
    x1, y1, x2, y2 = box
    ax.add_patch(
        plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2, label=label)
    )


def main() -> None:
    # Три штучні бокси для наочної демонстрації різних ступенів перекриття:
    # box1 і box2 - частково перекриваються, box1 і box3 - взагалі не перетинаються
    box1 = [50, 50, 150, 150]
    box2 = [90, 90, 190, 190]
    box3 = [300, 300, 400, 400]

    iou_12 = calculate_iou(box1, box2)
    iou_13 = calculate_iou(box1, box3)
    iou_11 = calculate_iou(box1, box1)  # бокс сам із собою -> IoU має дорівнювати 1.0

    print(f"IoU(box1, box2) = {iou_12:.3f}")
    print(f"IoU(box1, box3) = {iou_13:.3f}")
    print(f"IoU(box1, box1) = {iou_11:.3f}  (перевірка: бокс сам із собою)")

    # Візуалізація box1 і box2, як у демонстраційному прикладі до лабораторної
    fig, ax = plt.subplots(figsize=(5, 5))
    draw_box(ax, box1, "tab:blue", "box1")
    draw_box(ax, box2, "tab:orange", "box2")
    ax.set_xlim(0, 220)
    ax.set_ylim(220, 0)  # інвертуємо вісь Y, щоб відповідати системі координат зображення
    ax.set_title(f"IoU(box1, box2) = {iou_12:.3f}")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "step3_iou_visualization.png", dpi=150)
    plt.close(fig)

    summary = {
        "iou_box1_box2": round(iou_12, 3),
        "iou_box1_box3": round(iou_13, 3),
        "iou_box1_box1": round(iou_11, 3),
        "boxes": {"box1": box1, "box2": box2, "box3": box3},
    }
    (DATA_DIR / "step3_iou.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Графік збережено у {PLOTS_DIR / 'step3_iou_visualization.png'}")


if __name__ == "__main__":
    main()

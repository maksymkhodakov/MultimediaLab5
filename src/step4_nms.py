"""
Крок 4. NMS (Non-Maximum Suppression) - придушення немаксимумів.

Модель детекції зазвичай генерує декілька перекривних bounding boxes
навколо одного й того самого об'єкта (з різною впевненістю). NMS -
алгоритм, який залишає лише найкращий бокс серед групи перекривних,
видаляючи "дублікати" (детальний опис алгоритму - у docstring функції
utils.non_max_suppression).

Крок складається з двох частин:
    1. Демонстрація власної реалізації NMS на штучному прикладі
       (5 боксів, два "кластери" по 2-3 дублікати навколо одного об'єкта).
    2. Демонстрація вбудованого в Ultralytics NMS (параметр iou у
       model.predict()) на реальному зображенні - дослідження того, як
       кількість фінальних боксів залежить від порогу iou_threshold.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from config import DATA_DIR, PLOTS_DIR, YOLO_NANO_WEIGHTS
from utils import download_demo_image, non_max_suppression


def demo_synthetic_boxes():
    """Частина 1: власна реалізація NMS на штучному прикладі."""
    # 5 боксів: перші 3 - дублікати одного об'єкта (лівий кластер),
    # останні 2 - дублікати іншого об'єкта (правий кластер)
    boxes = np.array([
        [50, 50, 200, 200],
        [55, 45, 205, 195],
        [60, 60, 210, 210],
        [300, 50, 400, 220],
        [305, 55, 405, 225],
    ], dtype=float)
    scores = np.array([0.90, 0.75, 0.60, 0.85, 0.50])

    print(f"До NMS: боксів = {len(boxes)}")
    keep = non_max_suppression(boxes, scores, iou_threshold=0.5)
    print(f"Після NMS: боксів = {len(keep)} -> залишені індекси: {keep}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, idxs, title in (
        (axes[0], range(len(boxes)), "До NMS"),
        (axes[1], keep, "Після NMS"),
    ):
        for i in idxs:
            x1, y1, x2, y2 = boxes[i]
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=2))
            ax.text(x1, y1 - 5, f"{scores[i]:.2f}", color="red", fontsize=9)
        ax.set_xlim(0, 450)
        ax.set_ylim(250, 0)
        ax.set_title(title)
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "step4_nms_synthetic.png", dpi=150)
    plt.close(fig)

    return {"before": len(boxes), "after": len(keep), "kept_indices": keep}


def demo_builtin_nms_on_image():
    """Частина 2: вбудований NMS Ultralytics на реальному зображенні -
    дослідження впливу порогу iou_threshold на кількість фінальних боксів."""
    model = YOLO(YOLO_NANO_WEIGHTS)
    image_path = download_demo_image()

    results_by_threshold = {}
    for iou_threshold in (0.3, 0.5, 0.7, 0.9):
        results = model.predict(source=str(image_path), conf=0.1, iou=iou_threshold, verbose=False)
        n_boxes = len(results[0].boxes)
        results_by_threshold[iou_threshold] = n_boxes
        print(f"iou_threshold = {iou_threshold}: залишилось боксів = {n_boxes}")

    return results_by_threshold


def main() -> None:
    synthetic_result = demo_synthetic_boxes()
    builtin_result = demo_builtin_nms_on_image()

    summary = {
        "synthetic_example": synthetic_result,
        "builtin_nms_by_iou_threshold": builtin_result,
    }
    (DATA_DIR / "step4_nms.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Графік збережено у {PLOTS_DIR / 'step4_nms_synthetic.png'}")


if __name__ == "__main__":
    main()

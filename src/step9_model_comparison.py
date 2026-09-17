"""
Крок 9 (розширене дослідження). Порівняння моделей YOLOv8 різного розміру.

Порівнюються дві моделі родини YOLOv8: nano (yolov8n, найменша й
найшвидша) та small (yolov8s, більша й точніша) за принципом
Accuracy ↔ Speed (точність у порівнянні зі швидкодією) - типовий
компроміс для моделей глибокого навчання: більша модель, як правило,
дає вищу точність ціною нижчої швидкодії.

Для кожної моделі вимірюються:
    * кількість параметрів моделі;
    * Precision, Recall, mAP50, mAP50-95 на наборі coco128;
    * середній FPS на тестовому зображенні (imgsz=640).
"""

import json
import time

import cv2
from ultralytics import YOLO

from config import DATA_DIR, VAL_DATASET, YOLO_NANO_WEIGHTS, YOLO_SMALL_WEIGHTS
from utils import download_demo_image

N_WARMUP = 2
N_REPEATS = 10


def measure_fps(model: YOLO, image) -> float:
    for _ in range(N_WARMUP):
        model.predict(source=image, imgsz=640, conf=0.25, verbose=False)
    start = time.perf_counter()
    for _ in range(N_REPEATS):
        model.predict(source=image, imgsz=640, conf=0.25, verbose=False)
    elapsed = time.perf_counter() - start
    return N_REPEATS / elapsed


def evaluate_model(weights: str, image) -> dict:
    model = YOLO(weights)
    n_params = sum(p.numel() for p in model.model.parameters())

    metrics = model.val(data=VAL_DATASET, verbose=False)
    fps = measure_fps(model, image)

    result = {
        "weights": weights,
        "n_parameters": n_params,
        "precision": round(float(metrics.box.mp), 3),
        "recall": round(float(metrics.box.mr), 3),
        "map50": round(float(metrics.box.map50), 3),
        "map50_95": round(float(metrics.box.map), 3),
        "fps": round(fps, 1),
    }
    return result


def main() -> None:
    image_path = download_demo_image()
    image = cv2.imread(str(image_path))

    results = {}
    for weights in (YOLO_NANO_WEIGHTS, YOLO_SMALL_WEIGHTS):
        print(f"\n--- Оцінка моделі {weights} ---")
        results[weights] = evaluate_model(weights, image)
        r = results[weights]
        print(
            f"Параметрів: {r['n_parameters']:,}  "
            f"Precision={r['precision']}  Recall={r['recall']}  "
            f"mAP50={r['map50']}  mAP50-95={r['map50_95']}  FPS={r['fps']}"
        )

    (DATA_DIR / "step9_model_comparison.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2)
    )
    print(f"\nРезультати збережено у {DATA_DIR / 'step9_model_comparison.json'}")


if __name__ == "__main__":
    main()

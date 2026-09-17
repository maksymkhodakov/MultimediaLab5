"""
Крок 2. Inference (детекція) моделі YOLO на статичному зображенні.

Запускаємо попередньо навчену модель YOLOv8n на тестовому зображенні
bus.jpg з параметром conf (confidence threshold) = 0.25 - мінімальний
рівень впевненості, при якому детекція вважається "справжньою" і
потрапляє у результат. Для кожного знайденого об'єкта модель повертає:
    * bounding box  - координати прямокутника [x1, y1, x2, y2];
    * клас об'єкта  - наприклад, "person", "bus";
    * confidence     - рівень впевненості моделі (від 0 до 1).

Результат візуалізується вбудованим методом .plot() (модель сама малює
bounding boxes з підписами класу та впевненості) і зберігається у файл.
"""

import json
import time

from ultralytics import YOLO

from config import DATA_DIR, IMAGES_DIR, YOLO_NANO_WEIGHTS
from utils import download_demo_image


def main() -> None:
    model = YOLO(YOLO_NANO_WEIGHTS)
    image_path = download_demo_image()

    # Заміряємо час одного inference-проходу
    start = time.perf_counter()
    results = model.predict(source=str(image_path), conf=0.25, verbose=False)
    elapsed_ms = (time.perf_counter() - start) * 1000

    result = results[0]
    print(f"Час inference: {elapsed_ms:.1f} мс")
    print(f"Знайдено об'єктів: {len(result.boxes)}")

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        xyxy = [round(v, 1) for v in box.xyxy[0].tolist()]
        detections.append({"class": cls_name, "confidence": round(conf, 3), "box": xyxy})
        print(f"  {cls_name:<15} conf={conf:.2f}  box={xyxy}")

    # Вбудована в Ultralytics візуалізація: малює бокси, підписи класів
    # та рівень впевненості прямо поверх копії зображення
    annotated = result.plot()
    out_path = IMAGES_DIR / "step2_detection_result.jpg"
    import cv2
    cv2.imwrite(str(out_path), annotated)
    print(f"Візуалізацію збережено у {out_path}")

    summary = {
        "inference_time_ms": round(elapsed_ms, 1),
        "num_detections": len(result.boxes),
        "detections": detections,
    }
    (DATA_DIR / "step2_image_inference.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()

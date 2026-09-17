"""
Крок 1. Налаштування середовища.

Мета кроку - переконатися, що всі необхідні бібліотеки встановлені,
завантажити попередньо навчену модель YOLOv8n ("n" = nano, найменша й
найшвидша версія родини YOLOv8) та тестове зображення bus.jpg, а також
вивести базову інформацію про модель (кількість класів COCO, приклади
назв класів). Результат кроку зберігається у форматі JSON, щоб його
можна було використати у звіті.
"""

import json

import ultralytics
from ultralytics import YOLO

from config import DATA_DIR, YOLO_NANO_WEIGHTS
from utils import download_demo_image


def main() -> None:
    print(f"Версія Ultralytics: {ultralytics.__version__}")

    # Завантажуємо ваги моделі YOLOv8n. Якщо файла немає локально,
    # ultralytics автоматично завантажить його з офіційного репозиторію.
    model = YOLO(YOLO_NANO_WEIGHTS)
    print(f"Модель завантажено: {YOLO_NANO_WEIGHTS}")

    # model.names - словник {id_класу: назва_класу}, який модель отримала
    # під час навчання на датасеті COCO (Common Objects in Context, 80 класів)
    class_names = list(model.names.values())
    print(f"Кількість класів (COCO): {len(class_names)}")
    print(f"Приклади класів: {class_names[:10]}")

    image_path = download_demo_image()
    print(f"Тестове зображення завантажено: {image_path}")

    summary = {
        "ultralytics_version": ultralytics.__version__,
        "model_weights": YOLO_NANO_WEIGHTS,
        "num_classes": len(class_names),
        "example_classes": class_names[:10],
        "demo_image": str(image_path),
    }
    out_path = DATA_DIR / "step1_setup.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Результати збережено у {out_path}")


if __name__ == "__main__":
    main()

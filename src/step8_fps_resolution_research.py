"""
Крок 8 (дослідницька складова). Залежність FPS від роздільної здатності зображення.

FPS = f(роздільна здатність зображення)

Для кожного значення розміру зображення (imgsz - довша сторона у
пікселях, менша сторона масштабується пропорційно) виконується серія
повторних inference-проходів на тестовому зображенні, після чого
обчислюється середній час одного проходу та відповідний FPS
(кадрів за секунду = 1 / середній_час_проходу).

Перший ("холодний") прохід виключається з вимірювання, оскільки він
включає одноразові витрати на ініціалізацію обчислювального графа
моделі (warm-up) і суттєво спотворює середній результат.

Результат використовується для відповіді на дослідницьке питання
"як роздільна здатність зображення впливає на mAP та швидкодію?"
(частина щодо швидкодії; частина щодо точності обчислюється окремо
у step9, оскільки для mAP потрібен розмічений набір даних).
"""

import json
import time

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

from config import DATA_DIR, IMAGE_SIZES, PLOTS_DIR, YOLO_NANO_WEIGHTS
from utils import download_demo_image

N_WARMUP = 2
N_REPEATS = 10


def measure_fps(model: YOLO, image, imgsz: int) -> float:
    # Прогріваємо модель (перші виклики повільніші через ініціалізацію)
    for _ in range(N_WARMUP):
        model.predict(source=image, imgsz=imgsz, conf=0.25, verbose=False)

    start = time.perf_counter()
    for _ in range(N_REPEATS):
        model.predict(source=image, imgsz=imgsz, conf=0.25, verbose=False)
    elapsed = time.perf_counter() - start

    avg_time_per_frame = elapsed / N_REPEATS
    return 1.0 / avg_time_per_frame


def main() -> None:
    model = YOLO(YOLO_NANO_WEIGHTS)
    image_path = download_demo_image()
    image = cv2.imread(str(image_path))

    records = []
    for imgsz in IMAGE_SIZES:
        fps = measure_fps(model, image, imgsz)
        records.append({"imgsz": imgsz, "fps": fps})
        print(f"Роздільна здатність (imgsz) = {imgsz:<5} -> FPS = {fps:.1f}")

    sizes = [r["imgsz"] for r in records]
    fps_values = [r["fps"] for r in records]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sizes, fps_values, marker="o", color="tab:red")
    ax.set_xlabel("Роздільна здатність зображення (imgsz, пікселів)")
    ax.set_ylabel("FPS (кадрів за секунду)")
    ax.set_title("Залежність швидкодії (FPS) від роздільної здатності зображення")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "step8_fps_vs_resolution.png", dpi=150)
    plt.close(fig)

    (DATA_DIR / "step8_fps_resolution.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2)
    )
    print(f"Графік збережено у {PLOTS_DIR / 'step8_fps_vs_resolution.png'}")


if __name__ == "__main__":
    main()

"""
Крок 5. Обробка відеопотоку.

Оскільки веб-камера в середовищі виконання недоступна, для демонстрації
обробки відео генерується коротке синтетичне відео на основі окремого
тестового зображення zidane.jpg (офіційний приклад Ultralytics з двома
людьми на футбольному полі - навмисно інше зображення, ніж bus.jpg, яке
використовується у кроках 2-4, щоб відеоряд не дублював вже показані кадри).

Використано ефект плавного наближення камери (zoom-in, "ефект Кена
Бернса"): вікно перегляду поступово звужується навколо центру кадру,
імітуючи наближення камери до об'єктів зйомки. Це дає послідовність
кадрів, на яких об'єкти по-різному видно (спочатку дрібно й повністю,
потім - крупним планом, частково виходячи за межі кадру), що добре
перевіряє стабільність детекції YOLO при зміні масштабу об'єктів.

Далі відео обробляється кадр за кадром за стандартною схемою:
    читаємо кадр -> запускаємо детекцію YOLO -> малюємо бокси ->
    записуємо кадр у вихідний відеофайл.

Наприкінці підраховується середній FPS (кадрів за секунду) обробки та
загальна кількість детекцій за все відео.
"""

import json
import time

import cv2
import numpy as np
from ultralytics import YOLO

from config import DATA_DIR, IMAGES_DIR, VIDEO_DIR, YOLO_NANO_WEIGHTS
from utils import download_video_source_image

N_FRAMES = 30          # кількість кадрів у синтетичному відео
OUTPUT_FPS = 15         # частота кадрів запису (для відтворення відеофайлу)
ZOOM_START = 1.0        # масштаб вікна перегляду на першому кадрі (1.0 = весь кадр)
ZOOM_END = 1.8          # масштаб вікна перегляду на останньому кадрі (наближення)

# Індекси кадрів (0-based), які додатково зберігаються як окремі JPG-скріншоти
# для ілюстрації відео у звіті (початок / середина / кінець панорами наближення)
SCREENSHOT_FRAME_INDICES = (0, N_FRAMES // 2, N_FRAMES - 1)


def generate_synthetic_video(source_image: np.ndarray, n_frames: int) -> list[np.ndarray]:
    """Створює послідовність кадрів з ефектом плавного наближення (zoom-in)
    до центру вихідного зображення - імітує рух камери "вперед" по сцені.
    """
    h, w = source_image.shape[:2]
    cx, cy = w / 2, h / 2

    frames = []
    for i in range(n_frames):
        t = i / max(1, n_frames - 1)
        zoom = ZOOM_START + (ZOOM_END - ZOOM_START) * t

        # Розмір вікна перегляду звужується зі зростанням zoom
        view_w = w / zoom
        view_h = h / zoom

        x1 = int(max(0, cx - view_w / 2))
        y1 = int(max(0, cy - view_h / 2))
        x2 = int(min(w, cx + view_w / 2))
        y2 = int(min(h, cy + view_h / 2))

        crop = source_image[y1:y2, x1:x2]
        frame = cv2.resize(crop, (w, h))
        frames.append(frame)
    return frames


def main() -> None:
    image_path = download_video_source_image()
    source_image = cv2.imread(str(image_path))
    h, w = source_image.shape[:2]

    frames = generate_synthetic_video(source_image, N_FRAMES)
    print(f"Синтетичне відео створено (джерело: {image_path.name}): {len(frames)} кадрів, розмір {w}x{h}")

    model = YOLO(YOLO_NANO_WEIGHTS)

    out_path = VIDEO_DIR / "synthetic_detection.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, OUTPUT_FPS, (w, h))

    screenshot_paths = []
    total_detections = 0
    start = time.perf_counter()
    for i, frame in enumerate(frames):
        # Детекція на поточному кадрі
        results = model.predict(source=frame, conf=0.25, verbose=False)
        total_detections += len(results[0].boxes)

        # Вбудована візуалізація малює бокси прямо на кадрі
        annotated = results[0].plot()
        writer.write(annotated)

        # Зберігаємо кілька анотованих кадрів як окремі скріншоти для звіту
        if i in SCREENSHOT_FRAME_INDICES:
            shot_path = IMAGES_DIR / f"step5_video_frame_{i:02d}.jpg"
            cv2.imwrite(str(shot_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
            screenshot_paths.append(str(shot_path))
    elapsed = time.perf_counter() - start
    writer.release()

    processing_fps = len(frames) / elapsed
    print(f"Опрацьовано {len(frames)} кадрів за {elapsed:.2f} с -> {processing_fps:.1f} FPS")
    print(f"Всього детекцій за все відео: {total_detections}")
    print(f"Відео збережено у {out_path}")

    summary = {
        "source_image": image_path.name,
        "effect": "zoom-in (Ken Burns), центр кадру",
        "n_frames": len(frames),
        "elapsed_seconds": round(elapsed, 2),
        "processing_fps": round(processing_fps, 1),
        "total_detections": total_detections,
        "output_video": str(out_path),
        "screenshot_frames": screenshot_paths,
    }
    (DATA_DIR / "step5_video_processing.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()

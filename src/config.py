"""
Спільні налаштування (константи) для всіх скриптів лабораторної роботи №5.

Тут зібрані шляхи до файлів та каталогів, а також параметри, які
використовуються у декількох скриптах одночасно. Це зроблено для того,
щоб не дублювати "магічні" рядки та числа в кожному файлі окремо.
"""

from pathlib import Path

# Коренева директорія проєкту (на один рівень вище за src/)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Каталог для збереження усіх результатів роботи (зображення, графіки, відео, дані)
RESULTS_DIR = ROOT_DIR / "results"
IMAGES_DIR = RESULTS_DIR / "images"
PLOTS_DIR = RESULTS_DIR / "plots"
VIDEO_DIR = RESULTS_DIR / "video"
DATA_DIR = RESULTS_DIR / "data"

for d in (IMAGES_DIR, PLOTS_DIR, VIDEO_DIR, DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Назви ваг попередньо навчених моделей YOLOv8 (завантажуються автоматично
# бібліотекою ultralytics при першому використанні з офіційного репозиторію).
YOLO_NANO_WEIGHTS = "yolov8n.pt"   # найменша й найшвидша модель ("n" = nano)
YOLO_SMALL_WEIGHTS = "yolov8s.pt"  # модель середнього розміру ("s" = small)

# Тестове зображення, яке постачається разом з бібліотекою ultralytics
# (використовується як стандартний приклад у документації YOLO)
DEMO_IMAGE_URL = "https://ultralytics.com/images/bus.jpg"
DEMO_IMAGE_PATH = IMAGES_DIR / "bus.jpg"

# Окреме зображення-джерело для синтетичного відео (крок 5) - навмисно інше,
# ніж DEMO_IMAGE_PATH, щоб відеоряд відрізнявся від зображення, яке
# використовується для inference/IoU/NMS. Це офіційний тестовий приклад
# ultralytics з двома людьми на футбольному полі.
VIDEO_SOURCE_IMAGE_URL = "https://ultralytics.com/images/zidane.jpg"
VIDEO_SOURCE_IMAGE_PATH = IMAGES_DIR / "zidane.jpg"

# Набір даних для обчислення Precision/Recall/mAP. coco128 - невелика (128
# зображень) підмножина датасету COCO з готовою розміткою (ground truth),
# яку ultralytics вміє завантажувати автоматично.
VAL_DATASET = "coco128.yaml"

# Пороги впевненості (confidence threshold), вплив яких досліджується
# у розділі "Дослідницька складова" завдання лабораторної роботи.
CONFIDENCE_THRESHOLDS = [0.1, 0.25, 0.5, 0.75, 0.9]

# Роздільні здатності зображення (у пікселях, по довшій стороні), для яких
# вимірюється швидкодія (FPS) моделі.
IMAGE_SIZES = [320, 480, 640, 960, 1280]

RANDOM_SEED = 42

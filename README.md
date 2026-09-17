# Лабораторна робота №5
## "Методи обробки мультимедійних даних та інформації"
## Ходаков Максим Олегович ШІ-2

Тема: **Виявлення об'єктів (Object Detection) за допомогою YOLO**

## Структура проєкту

```
src/
  config.py                       - спільні константи (шляхи, пороги, розміри)
  utils.py                        - IoU, власна реалізація NMS, малювання боксів
  step1_setup.py                  - перевірка середовища, завантаження моделі YOLOv8n
  step2_image_inference.py        - inference на статичному зображенні
  step3_iou.py                    - обчислення та візуалізація IoU
  step4_nms.py                    - власна реалізація NMS + вбудований NMS Ultralytics
  step5_video_processing.py       - обробка синтетичного відеопотоку
  step6_map_evaluation.py         - Precision/Recall/mAP на наборі coco128
  step7_confidence_research.py    - дослідження Precision/Recall = f(confidence)
  step8_fps_resolution_research.py- дослідження FPS = f(роздільна здатність)
  step9_model_comparison.py       - порівняння YOLOv8n (nano) та YOLOv8s (small)
  run_all.py                      - послідовний запуск усіх кроків
results/
  images/  - вхідні та анотовані зображення
  plots/   - графіки досліджень (matplotlib)
  video/   - оброблене відео з детекціями
  data/    - результати у форматі JSON
Звіт_Лабораторна_робота_5.docx    - звіт до лабораторної роботи
```

## Запуск

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt
cd src
python run_all.py          # запустити всі кроки послідовно
python step2_image_inference.py   # або окремий крок
```

Модель YOLOv8 (`ultralytics`) та набір даних `coco128` завантажуються
автоматично при першому запуску.

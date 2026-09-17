"""
Крок 6. mAP (mean Average Precision) та Precision/Recall на розміченому наборі даних.

Щоб чесно обчислити Precision, Recall і mAP, потрібні зображення з
еталонною розміткою (ground truth). Використовуємо coco128 - невелику
(128 зображень) підмножину датасету COCO з готовою розміткою, яку
ultralytics вміє завантажувати автоматично при першому виклику model.val().

Метрики, які повертає вбудований валідатор Ultralytics:
    * Precision (точність) - частка правильних детекцій серед усіх зроблених;
    * Recall (повнота)     - частка знайдених об'єктів серед усіх, що
                              насправді присутні на зображенні;
    * mAP50                - mean Average Precision при порозі IoU = 0.5;
    * mAP50-95              - усереднення mAP по порогах IoU від 0.5 до 0.95
                              (стандартна "сувора" метрика COCO).

Додатково виводяться класи, на яких модель детектує найгірше і найкраще
(за значенням AP50 для конкретного класу) - це дає відповідь на
дослідницьке питання "які класи об'єктів детектуються найгірше і чому?".
"""

import json

from ultralytics import YOLO

from config import DATA_DIR, VAL_DATASET, YOLO_NANO_WEIGHTS


def main() -> None:
    model = YOLO(YOLO_NANO_WEIGHTS)

    # model.val() автоматично завантажує coco128 (якщо потрібно), проганяє
    # модель по всіх зображеннях набору та порівнює передбачення з
    # еталонною розміткою, обчислюючи Precision/Recall/mAP
    metrics = model.val(data=VAL_DATASET, verbose=False)

    precision = float(metrics.box.mp)      # mean precision по всіх класах
    recall = float(metrics.box.mr)         # mean recall по всіх класах
    map50 = float(metrics.box.map50)       # mAP при IoU=0.5
    map50_95 = float(metrics.box.map)      # mAP, усереднений по IoU 0.5:0.95

    print(f"Precision (mean, усі класи) = {precision:.3f}")
    print(f"Recall    (mean, усі класи) = {recall:.3f}")
    print(f"mAP50                       = {map50:.3f}")
    print(f"mAP50-95                    = {map50_95:.3f}")

    # AP50 для кожного окремого класу - дозволяє знайти класи,
    # які модель детектує найгірше та найкраще
    class_ap50 = {}
    for i, cls_id in enumerate(metrics.ap_class_index):
        cls_name = model.names[int(cls_id)]
        # metrics.box.ap50 - масив AP при IoU=0.5 для кожного класу з ap_class_index
        class_ap50[cls_name] = float(metrics.box.ap50[i])

    sorted_classes = sorted(class_ap50.items(), key=lambda kv: kv[1])
    worst_5 = sorted_classes[:5]
    best_5 = sorted_classes[-5:][::-1]

    print("\nКласи з найнижчим AP50 (найгірше детектуються):")
    for name, ap in worst_5:
        print(f"  {name:<15} AP50 = {ap:.3f}")

    print("\nКласи з найвищим AP50 (найкраще детектуються):")
    for name, ap in best_5:
        print(f"  {name:<15} AP50 = {ap:.3f}")

    summary = {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "map50": round(map50, 3),
        "map50_95": round(map50_95, 3),
        "worst_classes": [{"class": n, "ap50": round(a, 3)} for n, a in worst_5],
        "best_classes": [{"class": n, "ap50": round(a, 3)} for n, a in best_5],
    }
    (DATA_DIR / "step6_map_evaluation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    print(f"\nРезультати збережено у {DATA_DIR / 'step6_map_evaluation.json'}")


if __name__ == "__main__":
    main()

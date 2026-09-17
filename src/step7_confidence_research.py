"""
Крок 7 (дослідницька складова). Вплив confidence threshold на Precision і Recall.

Confidence threshold (поріг впевненості) c - мінімальний рівень впевненості
моделі, при якому детекція вважається "справжньою" і потрапляє у
підсумковий результат. Дослідження виконується для набору порогів,
заданих у завданні лабораторної роботи:

    c ∈ {0.1, 0.25, 0.5, 0.75, 0.9}

Для кожного значення c запускається валідація моделі (model.val) на
розміченому наборі coco128 з відповідним порогом confidence, після чого
записуються отримані Precision і Recall. За результатами будуються
графіки залежностей Precision = f(c) та Recall = f(c), що дозволяє
відповісти на дослідницьке питання "як confidence threshold впливає на
precision та recall?".

Очікувана закономірність: зі зростанням c модель залишає лише
найбільш "впевнені" детекції -> Precision зростає (менше хибних
спрацювань), але Recall падає (пропускаються реальні об'єкти з
невисокою впевненістю).
"""

import json

import matplotlib.pyplot as plt
from ultralytics import YOLO

from config import CONFIDENCE_THRESHOLDS, DATA_DIR, PLOTS_DIR, VAL_DATASET, YOLO_NANO_WEIGHTS


def main() -> None:
    model = YOLO(YOLO_NANO_WEIGHTS)

    records = []
    for c in CONFIDENCE_THRESHOLDS:
        metrics = model.val(data=VAL_DATASET, conf=c, verbose=False)
        precision = float(metrics.box.mp)
        recall = float(metrics.box.mr)
        map50 = float(metrics.box.map50)
        records.append({"confidence": c, "precision": precision, "recall": recall, "map50": map50})
        print(f"c={c:<5} Precision={precision:.3f}  Recall={recall:.3f}  mAP50={map50:.3f}")

    confidences = [r["confidence"] for r in records]
    precisions = [r["precision"] for r in records]
    recalls = [r["recall"] for r in records]

    # Графік Precision = f(c) та Recall = f(c) на одних осях для наочного порівняння
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(confidences, precisions, marker="o", label="Precision(c)")
    ax.plot(confidences, recalls, marker="s", label="Recall(c)")
    ax.set_xlabel("Confidence threshold (c)")
    ax.set_ylabel("Значення метрики")
    ax.set_title("Вплив confidence threshold на Precision та Recall")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "step7_precision_recall_vs_confidence.png", dpi=150)
    plt.close(fig)

    (DATA_DIR / "step7_confidence_research.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2)
    )
    print(f"Графік збережено у {PLOTS_DIR / 'step7_precision_recall_vs_confidence.png'}")


if __name__ == "__main__":
    main()

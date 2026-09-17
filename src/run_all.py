"""
Головний скрипт: послідовно запускає всі кроки лабораторної роботи №5
та зберігає результати (зображення, графіки, відео, JSON-дані) у каталозі
results/. Зручно використовувати для повного відтворення експерименту
"одним запуском".
"""

import importlib
import time

STEPS = [
    "step1_setup",
    "step2_image_inference",
    "step3_iou",
    "step4_nms",
    "step5_video_processing",
    "step6_map_evaluation",
    "step7_confidence_research",
    "step8_fps_resolution_research",
    "step9_model_comparison",
]


def main() -> None:
    for step in STEPS:
        print("\n" + "=" * 70)
        print(f"ЗАПУСК: {step}")
        print("=" * 70)
        start = time.perf_counter()
        module = importlib.import_module(step)
        module.main()
        print(f"[{step}] завершено за {time.perf_counter() - start:.1f} с")


if __name__ == "__main__":
    main()

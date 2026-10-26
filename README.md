# AIJ2026 — Цифровой детектив

Сегментация локальных ИИ-изменений на фотографиях. Финальное решение —
**NC17 + адаптивный порог**, посылка **№121**, public AIC **0.831234**.
Одна SegFormer MiT-B2, декодер320, вход640×640, epoch=32, FP32, без ансамбля и TTA.
Модель с pooling — **93.116 GFLOPs**; H100 ещё не измерен.

| Каталог | Содержимое |
|---|---|
| [stage2](stage2/README.md) | Обучение учителей, мягкие цели, дистилляция, NB-валидация и инференс |
| [stage1](stage1/README.md) | Первый этап и исходные веса учителей/энкодера |

Python3.10+, CUDA. Все команды — из корня репозитория:

```bash
pip install -r requirements.txt
```

Веса `model.pt`, `policy.pkl`, `encoder_transform.pkl` —
[в отдельном каталоге](https://github.com/zokost/aij_2026/tree/main/weights_final/121_NC17_e32).
Положите их в `weights/121/`. SHA256 — в [weights.json](weights.json).

```bash
python -m stage2.infer --checkpoint weights/121/model.pt --calibration weights/121 --root data/test_stage2 --csv data/test_stage2/test.csv --out work/submission
```

Результат — `work/submission/submission.zip`: CSV и отдельные PNG-маски0/255
в исходном размере. По умолчанию используется исходный FP32 batch4.
`--batch-size 1` включает ускоренный CUDA Graph: на A100 + Xeon8358 среднее
**35.94мс**, p95 **47.07мс**, включая чтение JPEG и калибровку, без записи PNG.
При batch1 возможны небольшие отличия округления от исходной посылки.

[Обучение, калибровка и валидация](stage2/README.md) ·
[EDA и сравнения](stage2/eda_comparisons.ipynb).
Проверки: `python -m unittest discover -s tests -v`. Данные, манифесты и веса
хранятся отдельно.

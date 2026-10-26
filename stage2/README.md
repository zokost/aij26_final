# Финальный этап: NC17

Посылка **№121**, public AIC **0.831234**. MiT-B2 / SegFormer320, squish640,
epoch=32. Сегментатор обучается на stage2 и мягких NB-целях; дополнительно
получает BCE0.6 от учителей на stage2. На инференсе используется один студент
и замороженный калибратор с базовым порогом0.675.

## Запуск всего пайплайна

После подготовки данных и весов первого этапа:

```bash
python -m stage2.pipeline --root data --manifests manifests --weights weights --gpus 0 1 2 --out work/nc17_pipeline
```

На каждой карте последовательно обучаются NB-учитель и stage2-учитель своего
фолда (по 12 эпох). После завершения всех учителей скрипт экспортирует цели,
обучает студента до **epoch=32** с cosine-расписанием на 40 эпох,
обучает калибратор и оценивает NB при фиксированном пороге 0.675.
Одна карта также поддерживается: `--gpus 0`.

Повторный запуск с `--resume` пропускает завершённые шаги и продолжает
незавершённое обучение из `last.pt`. Частично выгруженные цели дополняются.
Хеши входов фиксируются; продолжение с изменёнными списками или весами отклоняется.
Логи шагов — `work/nc17_pipeline/logs/`, отчёт — `nb_validation.json`;
метрики после калибровки на control-частях — `final/calibration.json`.
`--dry-run` печатает команды, `--check-only` проверяет файлы и SHA256 без GPU-работы.

В `work/nc17_pipeline/final/` находятся `model.pt`, `policy.pkl`,
`encoder_transform.pkl`, `calibration.json`. Их можно передать в обычный инференс:

```bash
python -m stage2.infer --checkpoint work/nc17_pipeline/final/model.pt --calibration work/nc17_pipeline/final --root data/test_stage2 --csv data/test_stage2/test.csv --out work/submission
```

Это полное обучение **второго этапа из инициализаций stage1**, не подготовка
разностных NB-масок и не обучение stage1 с нуля. Для первого этапа есть
[отдельные команды](../stage1/README.md). Исходные команды отдельных шагов приведены ниже.

## Данные

Команды выполняются из корня. Изображения и цели — в `data/`, исходные веса —
в `weights/`, подготовленные CSV — в `manifests/`:

- `stage2_train.csv`:19837 строк, включая чистые кадры;
- `nb_pairs.csv`:9693 NB-пары, `pair_id`, `fold`;
- `epoch01.csv` … `epoch40.csv`:1999 NB-пар на эпоху;
- `teacher_extra0.csv` … `teacher_extra2.csv`: дополнения для NB-учителей;
- `teacher_stage2_fold0.csv` … `teacher_stage2_fold2.csv`: полные списки stage2-учителей;
- `stage2_heldout_fold0.csv` … `stage2_heldout_fold2.csv`: исключённые stage2-источники;
- `stage2_pairs.csv`:19837 stage2-кадров с `fold` для экспорта OOF-целей.
- `calibration.csv`:2990 NB/stage2-кадров; `path`, `mask`, `valid`, `positive`,
  `source` и фиксированная `partition` (`policy_fit` / `control`).

Колонки изображений: `orgl_img_path`, `chng_img_path`, `gt_path`, пути относительно
`data/`. Пустой `gt_path` — чистый кадр. Списки и разностные NB-маски являются
входными артефактами; они не строятся заново. Источник и его правки относятся
к одному фолду. Программа проверяет пересечения путей, включая симлинки.
Архив списков — `manifests_nc17_20261026.zip`; размеры и SHA256 — в
[assets/manifests.json](assets/manifests.json).
Для запуска нужны четыре файла инициализации из `weights.json`:
`FULL2_late_avg5.pt` и `DIAG_teacher_fold{0,1,2}of3_cnxl_832_e40.pt`.
Чекпоинт `stage1_final_avg5.pt` используется только для самостоятельного stage1-инференса.

## Учителя и цели

ConvNeXt-L DINOv3 / UPerNet832,12 эпох, batch6, AdamW1e-4, LR энкодера×0.1.
Для каждого `F=0,1,2` запустите на отдельной карте:

```bash
CUDA_VISIBLE_DEVICES=$F python -m stage2.train teacher --fold $F --manifests manifests --initial weights/DIAG_teacher_fold${F}of3_cnxl_832_e40.pt --out work/nb_teacher$F
CUDA_VISIBLE_DEVICES=$F python -m stage2.train stage2-teacher --fold $F --manifests manifests --initial weights/DIAG_teacher_fold${F}of3_cnxl_832_e40.pt --out work/stage2_teacher$F
```

Это два набора учителей: NB-пара исключена из дообучения своего NB-учителя,
а stage2-источник — из дообучения своего stage2-учителя.

```bash
python -m stage2.labels --pairs manifests/nb_pairs.csv --teachers work/nb_teacher0/epoch12.pt work/nb_teacher1/epoch12.pt work/nb_teacher2/epoch12.pt --out data/nanobanana/nb_v2/probs_cfL
python -m stage2.labels --kind stage2 --pairs manifests/stage2_pairs.csv --teachers work/stage2_teacher0/epoch12.pt work/stage2_teacher1/epoch12.pt work/stage2_teacher2/epoch12.pt --out data/stage2_oof
```

NB-цели: `round(255*p)`, `p<0.02` обнуляется. Stage2-цели: native-size float16
логиты в NPZ; sigmoid вычисляется в float32 при загрузке, без клиппинга.

## Студент

```bash
python -m stage2.train student --manifests manifests --stage2-targets data/stage2_oof/index.csv --initial weights/FULL2_late_avg5.pt --out work/nc17
```

Свежий декодер, seed42; из первого этапа переносится только энкодер.
На эпоху:19837 stage2 +1999 мягких NB +1999 парных чистых исходников;
batch18,1324 шага. AdamW1e-4, LR энкодера1e-5, cosine40 эпох, warmup5%; epoch=32.
BF16-обучение. Аугментации: HFlip0.5, VFlip0.2, JPEG90–100 с вероятностью0.2.

Лосс — BCE0.5 с балансировкой позитивных пикселей + per-image Tversky0.5
(β0.7), для чистых целей — штраф площади. Дополнительный pixel-mean BCE×0.6
применяется только к stage2-кадрам с OOF-целями; официальная GT не заменяется.
Все три карты — GT, вероятность учителя и valid — проходят общую геометрию.
NB-пример получает прогноз одного соответствующего учителя, а не среднее трёх.
Вторая тройка даёт отдельные OOF-цели на stage2. На инференсе учителя не запускаются.

`--resume` продолжает совместимый `last.pt`; `--stop-epoch` сохраняет расписание.
`run.json` фиксирует версии и SHA256 манифестов. Повторное обучение не гарантирует
побитового совпадения: для воспроизведения посылки используйте сданные веса.

## Валидация и инференс

После каждой эпохи считаются Dice/FPR/AIC на550 NB-парах из `assets/nb_validation.csv`
при0.75. Маски приблизительные, разностные; панель использовалась неоднократно.
У NC17 epoch32 без калибровки: Dice **0.580080**, FPR **0.027273**, AIC **0.726761**.

```bash
python -m stage2.validate --checkpoint weights/121/model.pt --root data --out work/nc17_nb.json
python -m stage2.infer --checkpoint weights/121/model.pt --calibration weights/121 --root data/test_stage2 --csv data/test_stage2/test.csv --out work/submission
```

## Калибровка и скорость

Калибратор обучен на выделенной части NB/stage2:405 NB и835 stage2-позитивах,
прошедших базовый guard, с равным суммарным весом источников. ExtraTrees32,
depth5, leaf64, seed45.276 признаков: карта вероятностей22, геометрия49,
границы54, RGB119 и PCA32 от3072 ROI-признаков того же forward.

Для повторно обученного студента калибратор строится заново:

```bash
python -m stage2.fit_calibration --checkpoint work/nc17/epoch32.pt --entries manifests/calibration.csv --root data --out work/nc17_calibration
```

Scaler/PCA обучаются только на `policy_fit`; деревья — на положительных
примерах из этой части с базовой площадью ≥1%, с равным суммарным весом NB
и stage2. `control` используется только для отчёта. Сохраняются оба pickle-файла
и `calibration.json` с хешами чекпоинта и артефактов. Инференс проверяет их
совместимость. Замороженные веса №121 продолжают использовать исходный калибратор.

Порог0.675 — максимум AIC на NB550 в проверенной сетке с шагом0.025.
При базовой площади<1% маска заморожена. Иначе выбирается максимальный
предсказанный выигрыш Dice среди0.50/0.60/0.65/0.675; при выигрыше≤1e-8
остаётся0.675. Это сохраняет image-level срабатывания своей базовой карты.

По умолчанию FP32 batch4 воспроизводит все1890 масок №121 пиксель в пиксель.
`--batch-size 1` включает CUDA Graph и ускоренную обработку признаков.
На A100 + Xeon8358: среднее35.94мс, p95=47.07мс, включая чтение JPEG и калибровку,
без записи PNG/ZIP. Это не строгая гарантия50мс для каждого кадра; H100 не измерен.
Batch1 меняет1265 пикселей из1.432млрд относительно batch4, image-level флаги
сохраняются. Необязательный `--compile` даёт30.97мс, но меняет больше предсказаний.

## EDA и подбор

[Ноутбук](eda_comparisons.ipynb) содержит сохранённый EDA, методологию и сравнения.
Таблицы, NB-список, конфигурация поиска и зависимости отчёта — в `assets/`.

| Часть исследования | Где смотреть |
|---|---|
| EDA: типы правок, площади масок, ограничения разметки | [eda_comparisons.ipynb](eda_comparisons.ipynb) |
| История рецептов и public LB | [assets/report_experiments.csv](assets/report_experiments.csv) |
| B1/B2 с общим декодером, учитель и студенты | [assets/model_comparisons.csv](assets/model_comparisons.csv) |
| Лосс для малых правок и чистых кадров | [../common.py](../common.py), [distillation.py](distillation.py) |
| Аугментации и профили исследования | [runtime.py](runtime.py): итоговый `minimal`, дополнительный `forensic` |
| Запускаемый план сравнения архитектур и гиперпараметров | [search.py](search.py), [assets/search_space.json](assets/search_space.json) |

В обеих таблицах NB-метрики относятся к **сырым картам при0.75**, без калибратора.
В `report_experiments.csv` столбец `threshold` означает базовый порог посылки;
public LB относится к полному инференсу этой посылки. Поэтому одинаковые
NB-значения №117/№121 не означают одинаковую калибровку. Сравнение учителя
и студента не является чистой абляцией архитектуры: различаются данные и цели.

Для запуска ноутбука: `pip install -r stage2/assets/requirements-report.txt`.
Проверка плана9 вариантов без обучения:

```bash
python -m stage2.search --manifests manifests --initial weights/FULL2_late_avg5.pt --out work/screen --dry-run
```

Без `--dry-run` запускается NB-only скрининг:4 эпохи на вариант, общий бюджет,
412 tuning и138 control-пар. Control оценивается после фиксации победителя.
Для изолированного сравнения дополнительная stage2 KD в этом плане отключена.
Это короткий скрининг, не замена полного рецепта NC17; его9 запусков для отчёта
не выполнялись. Проверки: `python -m unittest discover -s tests -v`.

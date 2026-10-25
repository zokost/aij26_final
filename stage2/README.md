# KDALL

Воспроизведение конфигурации посылки №113: MiT-B2 / SegFormer320, squish640,
epoch=32, FP32-инференс с batch4, threshold=0.75, без ансамбля и TTA.


Запускать из корня репозитория с зависимостями из `requirements.txt`.
Изображения и подготовленные маски находятся вне Git в `data/`; веса — в `weights/`.
Подготовленные списки передаются отдельно в каталоге `manifests/`:

- `stage2_train.csv`: фиксированные 19837 строк, включая чистые оригиналы;
- `teacher_extra0.csv` … `teacher_extra2.csv`: NB без собственного фолда и дубль stage2;
- `nb_pairs.csv`: 9693 пары NB с колонками `pair_id` и `fold`;
- `epoch01.csv` … `epoch40.csv`: исходные списки по 1999 NB-пар на эпоху.

Общие колонки обучающих CSV: `orgl_img_path`, `chng_img_path`, `gt_path`.
Пути относительны `data/`, пустой `gt_path` означает чистое изображение.
Для учителей нужны исходные разностные NB-маски `nanobanana/nb_v2/masks_affine/`.
Эти подготовленные маски и списки здесь не вычисляются заново: они являются
обязательными входными артефактами воспроизведения. Исходная ротация NB сохранена
в списках эпох без изменений.

Дообучить трёх учителей из весов первого этапа (каждая команда на своей карте):

```bash
CUDA_VISIBLE_DEVICES=0 python -m stage2.train teacher --fold 0 --manifests manifests --initial weights/DIAG_teacher_fold0of3_cnxl_832_e40.pt --out work/stage2_teacher0
CUDA_VISIBLE_DEVICES=1 python -m stage2.train teacher --fold 1 --manifests manifests --initial weights/DIAG_teacher_fold1of3_cnxl_832_e40.pt --out work/stage2_teacher1
CUDA_VISIBLE_DEVICES=2 python -m stage2.train teacher --fold 2 --manifests manifests --initial weights/DIAG_teacher_fold2of3_cnxl_832_e40.pt --out work/stage2_teacher2
```

Учитель: ConvNeXt-L DINOv3 + UPerNet832, 12 эпох, batch6, AdamW1e-4,
LR энкодера ×0.1. NB-пару размечает учитель, не обучавшийся на её фолде.
Мягкая маска: `round(255*p)`, значения `p<0.02` обнуляются.

```bash
python -m stage2.labels --pairs manifests/nb_pairs.csv --teachers work/stage2_teacher0/epoch12.pt work/stage2_teacher1/epoch12.pt work/stage2_teacher2/epoch12.pt --out data/nanobanana/nb_v2/probs_cfL
python -m stage2.train student --manifests manifests --initial weights/FULL2_late_avg5.pt --out work/kdall
python -m stage2.infer --checkpoint work/kdall/epoch32.pt --root data/test_stage2 --csv data/test_stage2/test.csv --out work/submission_stage2
```

Студент: свежий декодер с seed42, энкодер FULL2_late_avg5 обучается с LR1e-5,
остальные веса — LR1e-4. На эпоху 19837 stage2 + 1999 NB + 1999 чистых исходников;
batch18, 1324 шага. Cosine рассчитан на 40 эпох с warmup5%; останавливаемся на32.
Лосс: BCE0.5 с балансировкой позитивных пикселей + per-image Tversky0.5 (β0.7),
для чистых масок — штраф предсказанной площади. Обучение BF16.
Аугментации: HFlip0.5, VFlip0.2, JPEG90–100 с вероятностью0.2.

`--resume` продолжает обучение из `last.pt`. `--stop-epoch` сохраняет исходное
расписание LR. Повторное обучение не гарантирует побитового совпадения весов;
точное воспроизведение посылки требует исходного `epoch32.pt`.

С исходными весами проверено побайтовое совпадение `submission.csv` и всех
1890 PNG-масок с посылкой №113. Проверен один шаг обучения BF16 с обратным
проходом; полное обучение повторно не запускалось.

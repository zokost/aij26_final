# Первый этап AIJ2026: 0.96427 public AIC

SegFormer MiT-B2, декодер шириной 128. Обучение: 704×704; инференс FP32 на ближайшем по
пропорциям холсте 832×608 / 704×704 / 608×832 (высота × ширина), порог 0.70, без TTA.
Максимум 97.217 строгих GFLOPs. Данные и веса хранятся отдельно от кода.

Python 3.10+, две CUDA-карты для исходной конфигурации обучения
(доводка требует около 63 ГБ памяти на каждой), одна карта для инференса:

```bash
pip install -r requirements.txt
# Распаковать официальные данные в data/stage1/ и data/test_stage1/.
# В каждом каталоге соответствующий train.csv/test.csv и относительные пути к изображениям.
python -m torch.distributed.run --nproc_per_node=2 train.py base --out work/base
python -m torch.distributed.run --nproc_per_node=2 train.py finetune --checkpoint work/base/epoch161.pt --out work/finetune
python average.py work/finetune/epoch{1,2,3,4,5}.pt --out work/final.pt
python infer.py --checkpoint work/final.pt --out work/submission
```

База: seed 42, энкодер ImageNet, 134055 train / 4210 val, AdamW с LR 3e-4,
LR энкодера ×0.1, батч 6 на карту, BCE 0.5 + Tversky 0.5 (β=0.7),
warmup 5% и cosine на 200 эпох; обучение останавливается на эпохе 161.
Доводка: все 138265 строк, батч 40 на карту, LR 3e-5, расписание на 10 эпох;
нужны первые 5 эпох. Усредняются все float-тензоры в FP64, целочисленные буферы
берутся из epoch5; BatchNorm заново не пересчитывается.

Сохраняется last.pt для `--resume`. Параметр `--stop-epoch` не меняет расписание LR.
Повторное обучение не гарантирует побитового совпадения весов. Для точного
инференса исходной посылки используйте исходный best.pt (SHA256
`90e3ad1116bfdb39c9cbba32262708b21b236fb7d60b92b5c53bae30c108cf8f`).

Большие модели первого этапа: три независимых фолда ConvNeXt-L DINOv3 + UPerNet,
вход 832×832 squish. Они инициализируют учителей второго этапа. Каждый запускается
на одной CUDA-карте; команды для трёх карт:

```bash
CUDA_VISIBLE_DEVICES=0 python train_teachers.py --fold 0 --out work/teacher0
CUDA_VISIBLE_DEVICES=1 python train_teachers.py --fold 1 --out work/teacher1
CUDA_VISIBLE_DEVICES=2 python train_teachers.py --fold 2 --out work/teacher2
```

Общий отложенный сплит: 12% сцен, для мониторинга фиксированные 8000 кадров.
Оставшаяся обучающая часть делится на три фолда по
`int(md5(scene).hexdigest()[:8], 16) % 3`: учитель k обучается на двух других.
Ключ сцены — путь оригинала, иначе путь маски. Для фолдов 0/1/2 используются
81188/80563/81429 обучающих строк соответственно; чистые исходники добавляются
по тому же правилу доли 25%.

Рецепт учителей: seed 42, batch 6, AdamW 3e-4, LR энкодера ×0.1, weight decay 1e-4,
warmup 5%, cosine на 40 эпох, BF16. Лосс BCE 0.5 + per-image Dice 0.5
с отдельным штрафом площади на чистых кадрах; без Tversky и взвешивания BCE.
Аугментации те же, что у базового MiT: отражения, повороты на 90°, лёгкая
фотометрия, без JPEG и шума. Лучший `best.pt` выбирается по AIC исходного размера
на сетке порогов и отсечения малых масок; ранняя остановка — 12 эпох без улучшения.
Исходные лучшие эпохи трёх фолдов: 37/36/38. Повторное обучение не гарантирует
побитового совпадения. `--resume` продолжает из `last.pt`.

Пять файлов весов передаются отдельно; размеры и SHA256 указаны в
[weights.json](weights.json). После скачивания их можно положить в `weights/`:

| Файл | Назначение |
|---|---|
| `stage1_final_avg5.pt` | Итоговая модель первого этапа, AIC 0.96427 |
| `FULL2_late_avg5.pt` | Только энкодер MiT-B2 для старта второго этапа: среднее эпох 156–160 отдельного FULL2-прогона на всей stage1 |
| `DIAG_teacher_fold0of3_cnxl_832_e40.pt` | Полная большая модель stage1, фолд 0 |
| `DIAG_teacher_fold1of3_cnxl_832_e40.pt` | Полная большая модель stage1, фолд 1 |
| `DIAG_teacher_fold2of3_cnxl_832_e40.pt` | Полная большая модель stage1, фолд 2 |

Для готовых итоговых весов первого этапа:

```bash
python infer.py --checkpoint weights/stage1_final_avg5.pt --out work/submission
```

# Первый этап AIJ2026: 0.96427 public AIC

SegFormer MiT-B2, декодер шириной 128. Обучение: 704×704; инференс FP32 на ближайшем по
пропорциям холсте 832×608 / 704×704 / 608×832 (высота × ширина), порог 0.70, без TTA.
Максимум 97.217 строгих GFLOPs. Веса и данные в репозиторий не включены.

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

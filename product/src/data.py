"""Датасеты и аугментации для задачи локализации манипуляций.

Особенности задачи, которые влияют на препроцессинг:

* Картинки разного размера (от ~300 до 1024 по стороне), метрика считается в
  ИСХОДНОМ разрешении -> на валидации/инференсе поднимаем предсказание обратно.
* В обучающей выборке есть Negative-примеры (пустая GT-маска). Их нельзя
  выкидывать: половина метрики — это отсутствие ложных срабатываний.
* Следы манипуляции — высокочастотные (границы вставки, разрыв в шуме/JPEG).
  Поэтому аугментации подобраны так, чтобы не стирать их: геометрия и мягкая
  фотометрия — да, сильное размытие/шум по всему кадру — нет.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# albumentations при импорте ходит в сеть за версией и на офлайн-машине сыпет
# UserWarning'ами — выключаем до импорта, чтобы логи обучения были чистыми
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A  # noqa: E402
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

cv2.setNumThreads(0)  # иначе конфликтует с num_workers в DataLoader

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
#                                 аугментации                                  #
# --------------------------------------------------------------------------- #
def _degradations(degrade: bool | str) -> list:
    """Деградации порознь: они бьют по разным вещам и мерились только вместе.

    ``P22`` (лучшая посылка, ЛБ 0.946185) отличается от ``P7`` тем, что снял
    СРАЗУ ОБЕ — JPEG и гауссов шум, — и выигрыш +0.005 приписан «деградациям»
    целиком. Это разные механизмы:

    * гауссов шум в тесте не встречается никогда и напрямую забивает
      высокочастотный остаток, по которому ищется подделка;
    * JPEG в тесте встречается всегда, причём 22% кадров сжаты сильно
      (средняя первых 16 коэффициентов таблицы квантования 7.3 против 1.0
      у остальных). Именно эта, сжатая, четверть даёт 55% всей потери Dice
      и разрыв обобщения 0.093 против 0.020 у остальных.

    Поэтому «JPEG без шума» — отдельная гипотеза, а не оттенок.
    """
    mode = {True: "both", False: "none"}.get(degrade, degrade)
    out = []
    if mode in ("jpeg", "both"):
        out.append(A.ImageCompression(quality_range=(40, 95), p=0.35))
    if mode == "mild_jpeg":
        out.append(A.ImageCompression(quality_range=(90, 100), p=0.20))
    if mode in ("noise", "both"):
        out.append(A.GaussNoise(std_range=(0.02, 0.08), p=0.15))
    if mode not in ("none", "jpeg", "mild_jpeg", "noise", "both"):
        raise ValueError(f"неизвестный режим деградаций: {degrade!r}")
    return out


def train_transform(img_size: int, no_upscale: bool = False, prescaled: bool = False,
                    additional_targets: dict[str, str] | None = None,
                    degrade: bool | str = True,
                    canvas: tuple[int, int] | None = None,
                    rot90: bool = True, squish: bool = False,
                    photometry: bool = True) -> A.Compose:
    """Аугментации для обучения.

    JPEG-компрессия здесь не косметика: тестовые кадры — jpg, и модель должна
    ловить манипуляцию поверх артефактов сжатия, а не сами артефакты.

    ``degrade=False`` убирает JPEG-компрессию и гауссов шум. Эти две аугментации
    бьют ровно по высокочастотным следам, по которым модель ищет подделку, и
    ни разу не проверялись абляцией — обоснование «тест приходит в jpg» звучит
    разумно, но измерено не было.

    ``prescaled=True`` означает, что кадр уже приведён к сетке модели снаружи, и
    масштабирующую голову надо убрать. Это нужно дистилляции: мягкая маска
    учителя хранится ровно в сетке модели, и любой ресайз внутри пайплайна
    пересэмплировал бы её ближайшим соседом, размывая ровно ту границу, ради
    которой мягкие цели и берутся. Все оставшиеся геометрические операции
    (добивка, кроп, отражения, повороты на 90) точны и маску не портят.
    """
    # при no_upscale масштабирование уже сделано в датасете (только вниз)
    # с прямоугольным холстом вписывание делает rect_fit в датасете
    if squish:
        # прямой ресайз в квадрат без сохранения пропорций. Мы отвергли это в
        # начале как «рассогласование с инференсом», но рассогласования нет,
        # если squish делается и там, и там. Цена — искажение пропорций
        # (медиана 1.50 -> 1.00), выигрыш — в 1.42 раза больше пикселей
        # содержимого: 692 тыс. против 462 тыс., потому что холст перестаёт
        # заполняться нулями. Для СЕМАНТИКИ пропорции важны, но мы ищем следы
        # обработки — разрыв в шуме, границу вставки, — а они локальные.
        head = [A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA)]
    else:
        head = ([] if (no_upscale or prescaled or canvas)
                else [A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA)])
    H, W = canvas if canvas else (img_size, img_size)
    return A.Compose(head + [
        A.PadIfNeeded(H, W, position="top_left",
                      border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
        A.RandomCrop(H, W),
        # геометрия — безопасна для следов манипуляции
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
    ] + ([] if (canvas or not rot90) else [
        # на прямоугольном холсте поворот на 90 менял бы стороны местами и ломал
        # форму входа; отражения безопасны и остаются
        A.RandomRotate90(p=0.5),
    ]) + [
    ] + ([
        # Мягкая фотометрия. Стоит с начала проекта и НИ РАЗУ не проверялась
        # абляцией: --no-degrade снимает только JPEG и шум, а яркость, контраст
        # и цвет остаются. Для поиска СЛЕДОВ ОБРАБОТКИ это подозрительно —
        # признак локальный и высокочастотный, а мы подкрашиваем кадр.
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=8, p=0.2),
    ] if photometry else []) + _degradations(degrade) + [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ], additional_targets=additional_targets or {})


def valid_transform(img_size: int, keep_aspect: bool = True,
                    no_upscale: bool = False,
                    canvas: tuple[int, int] | None = None,
                    additional_targets: dict[str, str] | None = None) -> A.Compose:
    """Валидация/инференс: приведение к сетке модели, без случайности.

    ``keep_aspect=True`` повторяет геометрию обучения: масштабирование по длинной
    стороне и добивка нулями вправо-вниз. Простой ``Resize`` до квадрата
    сплющивает кадр (91% тестовых изображений искажаются сильнее чем в 1.3 раза)
    и рассогласует инференс с обучением, где пропорции сохранялись. Плюс ресайз
    сам по себе разрушает высокочастотные следы подделки, по которым модель и
    работает.
    """
    if not keep_aspect:
        return A.Compose([
            A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ], additional_targets=additional_targets or {})
    if canvas:
        H, W = canvas
        return A.Compose([
            A.PadIfNeeded(H, W, position="top_left",
                          border_mode=cv2.BORDER_CONSTANT, fill=0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ], additional_targets=additional_targets or {})
    head = [] if no_upscale else [A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA)]
    return A.Compose(head + [
        A.PadIfNeeded(img_size, img_size, position="top_left",
                      border_mode=cv2.BORDER_CONSTANT, fill=0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ], additional_targets=additional_targets or {})


def content_size(h: int, w: int, img_size: int, no_upscale: bool = False) -> tuple[int, int]:
    """Размер полезной области внутри квадрата после LongestMaxSize + Pad.

    Добивка идёт вправо-вниз, поэтому содержимое занимает левый верхний угол
    [0:ch, 0:cw] — по нему и обрезаем предсказание перед возвратом в исходный
    размер кадра.
    """
    scale = img_size / max(h, w)
    if no_upscale:
        scale = min(1.0, scale)
    return max(1, min(img_size, round(h * scale))), max(1, min(img_size, round(w * scale)))


# --------------------------------------------------------------------------- #
#                        прямоугольный холст под форму кадра                   #
# --------------------------------------------------------------------------- #
def rect_content_size(h: int, w: int, canvas: tuple[int, int]) -> tuple[int, int, bool]:
    """Размер содержимого внутри прямоугольного холста и признак поворота.

    Квадратный холст 832x832 у нас заполнен содержимым в среднем на 70%:
    остальное — добивка нулями, на которую честно тратится бюджет FLOPs. Кадры
    в выборке в основном 3:2 (медиана соотношения 1.50), поэтому ландшафтный
    холст 672x1024 при ТОМ ЖЕ бюджете (48.91 GMAC против 49.29) заполняется на
    82.9%, а с поворотом портретных кадров — на 91.9%. Портретных всего 19%,
    поворот обратим, а к поворотам модель и так приучена RandomRotate90.
    """
    H, W = canvas
    rot = h > w
    if rot:
        h, w = w, h
    scale = min(H / h, W / w)
    return max(1, round(h * scale)), max(1, round(w * scale)), rot


def rect_fit(img: np.ndarray, mask: np.ndarray | None,
             canvas: tuple[int, int]) -> tuple[np.ndarray, np.ndarray | None, bool]:
    """Поворот портретного кадра и вписывание в холст с сохранением пропорций."""
    rot = img.shape[0] > img.shape[1]
    if rot:
        img = np.ascontiguousarray(np.rot90(img, 1))
        if mask is not None:
            mask = np.ascontiguousarray(np.rot90(mask, 1))
    h, w = img.shape[:2]
    H, W = canvas
    scale = min(H / h, W / w)
    ch, cw = max(1, round(h * scale)), max(1, round(w * scale))
    if (ch, cw) != (h, w):
        img = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_AREA)
        if mask is not None:
            mask = cv2.resize(mask, (cw, ch), interpolation=cv2.INTER_NEAREST)
    return img, mask, rot


# --------------------------------------------------------------------------- #
#                                  датасеты                                    #
# --------------------------------------------------------------------------- #
def downscale_only(img: np.ndarray, img_size: int,
                   mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Уменьшает кадр до ``img_size`` по длинной стороне, но НИКОГДА не растягивает.

    Зачем это отдельно от ``LongestMaxSize``: тот масштабирует в обе стороны, и
    мелкие кадры растягиваются вверх, теряя высокие частоты — ровно те, по
    которым модель ищет подделку. Беда в том, что растяжение затрагивает классы
    НЕРАВНОМЕРНО: при входе 800 вверх тянутся 80.6% наших Negative-кадров
    (оригиналы из src/ мельче) против 45.9% Positive и 47.2% тестовых. Возникает
    ярлык «нет высоких частот -> кадр чистый», которого на реальном тесте не
    будет, и FPR на валидации оказывается оптимистичным.

    Запрет на растяжение убирает асимметрию: мелкие кадры просто добиваются
    нулями, сохраняя родную частотную структуру.
    """
    h, w = img.shape[:2]
    if max(h, w) <= img_size:
        return img, mask
    scale = img_size / max(h, w)
    new_wh = (max(1, round(w * scale)), max(1, round(h * scale)))
    img = cv2.resize(img, new_wh, interpolation=cv2.INTER_AREA)
    if mask is not None:
        mask = cv2.resize(mask, new_wh, interpolation=cv2.INTER_NEAREST)
    return img, mask


# Матрица ортонормированного DCT-II 8x8: блочный DCT через два матричных
# умножения вместо цикла по блокам (у cv2.dct в src/dct_probe.py на кадр
# уходили десятки миллисекунд, здесь — доли).
_DCT8 = np.zeros((8, 8), np.float32)
for _k in range(8):
    for _n in range(8):
        _DCT8[_k, _n] = np.cos(np.pi * (2 * _n + 1) * _k / 16.0)
_DCT8 *= np.sqrt(2.0 / 8.0)
_DCT8[0] /= np.sqrt(2.0)


def dct_lattice_map(path: Path, shape: tuple[int, int],
                    gray: np.ndarray | None = None) -> np.ndarray:
    """Карта «нецелости» коэффициентов DCT по блокам 8x8, растянутая до кадра.

    Зачем. Единственная внешняя модель, сильная именно в нашем слабом числовом
    семействе, — CAT-Net, и она смотрит не на пиксели, а на след двойного
    сжатия в коэффициентах DCT. `src/dct_probe.py` подтвердил, что след там
    физически есть: числовое семейство реально сжато JPEG (шаг квантования
    Q(AC) 15.92), остальные практически без потерь (Q=1).

    Сырые коэффициенты подать нельзя: squish уничтожает решётку 8x8, а squish
    нам дороже. Поэтому считаем СТАТИСТИКУ на родном разрешении — насколько
    коэффициенты блока далеки от кратных шагу квантования, — и масштабируем уже
    её. Статистику ресайзить осмысленно, сырой сигнал, привязанный к сетке, —
    нет.

    Возвращает float32 в [0, 0.5] размера ``shape``; для не-JPEG отдаёт нули.
    """
    try:
        with Image.open(path) as im:
            qt = im.quantization
            if not qt:
                return np.zeros(shape, np.float32)
            qtab = np.asarray(qt[0], np.float32)
            if qtab.size != 64:
                return np.zeros(shape, np.float32)
            qtab = qtab.reshape(8, 8)
            # серый канал берём из уже прочитанного кадра, если он есть: второе
            # декодирование JPEG стоит дороже самой статистики
            gray = (np.asarray(im.convert("L"), np.float32) if gray is None
                    else gray.astype(np.float32)) - 128.0
    except Exception:
        return np.zeros(shape, np.float32)

    h8, w8 = gray.shape[0] // 8 * 8, gray.shape[1] // 8 * 8
    if h8 < 8 or w8 < 8:
        return np.zeros(shape, np.float32)
    blocks = gray[:h8, :w8].reshape(h8 // 8, 8, w8 // 8, 8).transpose(0, 2, 1, 3)
    coef = np.einsum("ku,bcuv,nv->bckn", _DCT8, blocks, _DCT8, optimize=True)
    r = coef / np.maximum(qtab, 1e-6)
    # DC-коэффициент выбрасываем: он несёт яркость блока, а не след сжатия
    off = np.abs(r - np.rint(r))[:, :, 1:, 1:].mean(axis=(2, 3))
    return cv2.resize(off.astype(np.float32), (shape[1], shape[0]),
                      interpolation=cv2.INTER_LINEAR)


# Карта нецелости DCT лежит примерно в [0, 0.5]; центрируем и масштабируем её
# к тому же порядку, что у RGB после ImageNet-нормализации.
DCT_MEAN, DCT_STD = 0.25, 0.12


def _read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"не читается изображение: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _read_mask(root: Path, gt_path: str, shape: tuple[int, int]) -> np.ndarray:
    """GT-маска; пустой ``gt_path`` означает Negative-кадр (маска из нулей).

    Пустой путь мы ставим сами, когда добавляем чистые оригиналы из ``src/`` в
    качестве Negative-примеров — файла маски для них в датасете нет.
    """
    if not gt_path or gt_path == "nan":
        return np.zeros(shape, dtype=np.uint8)
    m = cv2.imread(str(root / gt_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"не читается маска: {root / gt_path}")
    mask = (m > 127).astype(np.uint8)
    # изредка маска приходит иного размера, чем картинка
    if mask.shape[:2] != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def copy_paste(dst_img: np.ndarray, dst_mask: np.ndarray,
               donor_img: np.ndarray, donor_mask: np.ndarray,
               rng: np.random.Generator,
               scale_range: tuple[float, float] = (0.25, 1.0),
               feather: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Переносит настоящую подделанную область с одного кадра на другой.

    Зачем: метрика — средний Dice ПО КАДРАМ, поэтому мелкая манипуляция весит
    столько же, сколько крупная, а по нашим замерам мелкие и труднее (у кадров
    >=1000px их вдвое больше, и там Dice 0.709 против 0.894). Copy-Paste даёт
    неограниченный запас новых форм, положений и — главное — МАСШТАБОВ манипуляции
    с точной маской, не привлекая ни одной внешней модели: вставляемое содержимое
    произведено теми же SD2/LaMa/PowerPaint, что и весь конкурсный train.

    Граница растушёвывается, а не ставится встык: резкий шов — артефакт
    сплайсинга, которого в тестовых данных нет, и учить на нём значило бы
    повторить ошибку с nanobanana (там расхождение распределений всё испортило).

    ``scale_range`` смещён вниз — нам нужны именно мелкие манипуляции.
    """
    ys, xs = np.where(donor_mask > 0)
    if len(ys) < 64:                                   # донор без внятной области
        return dst_img, dst_mask

    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    patch = donor_img[y0:y1, x0:x1]
    patch_m = donor_mask[y0:y1, x0:x1]

    dh, dw = dst_img.shape[:2]
    scale = float(rng.uniform(*scale_range))
    # не даём вставке вылезти за кадр
    max_scale = min(dh / patch.shape[0], dw / patch.shape[1]) * 0.95
    scale = min(scale, max_scale)
    nh, nw = max(8, int(patch.shape[0] * scale)), max(8, int(patch.shape[1] * scale))
    patch = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_AREA)
    patch_m = cv2.resize(patch_m, (nw, nh), interpolation=cv2.INTER_NEAREST)

    top = int(rng.integers(0, max(1, dh - nh)))
    left = int(rng.integers(0, max(1, dw - nw)))

    # мягкое альфа-смешивание по краю вместо резкого шва
    alpha = cv2.GaussianBlur(patch_m.astype(np.float32), (feather | 1, feather | 1), 0)
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]

    out = dst_img.copy()
    region = out[top:top + nh, left:left + nw].astype(np.float32)
    out[top:top + nh, left:left + nw] = (
        region * (1 - alpha) + patch.astype(np.float32) * alpha).astype(np.uint8)

    new_mask = dst_mask.copy()
    new_mask[top:top + nh, left:left + nw] = np.maximum(
        new_mask[top:top + nh, left:left + nw], patch_m)
    return out, new_mask


def positive_zoom_crop(img: np.ndarray, mask: np.ndarray,
                       rng: np.random.Generator,
                       zoom_range: tuple[float, float] = (1.25, 3.0)
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Кроп вокруг случайного GT-пикселя с увеличением объекта в ``zoom`` раз.

    Окно сохраняет пропорции исходного кадра. После штатного ``squish`` это даёт
    ту же геометрию, что у полного кадра, но мелкая манипуляция занимает больше
    пикселей и не исчезает при ресайзе. Выбранный положительный пиксель всегда
    остаётся внутри окна; сама область может быть обрезана, что полезно для
    разнообразия границ и соответствует случайным patch-crop из IML-решений.
    """
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return img, mask
    h, w = mask.shape[:2]
    lo, hi = zoom_range
    zoom = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    crop_h = max(32, min(h, round(h / zoom)))
    crop_w = max(32, min(w, round(w / zoom)))
    if crop_h == h and crop_w == w:
        return img, mask

    k = int(rng.integers(len(ys)))
    cy, cx = int(ys[k]), int(xs[k])
    # Любая координата начала из этих интервалов удерживает anchor в окне.
    x_lo, x_hi = max(0, cx - crop_w + 1), min(cx, w - crop_w)
    y_lo, y_hi = max(0, cy - crop_h + 1), min(cy, h - crop_h)
    x0 = int(rng.integers(x_lo, x_hi + 1))
    y0 = int(rng.integers(y_lo, y_hi + 1))
    return (np.ascontiguousarray(img[y0:y0 + crop_h, x0:x0 + crop_w]),
            np.ascontiguousarray(mask[y0:y0 + crop_h, x0:x0 + crop_w]))


# Классы вспомогательной задачи «каким генератором сделан кадр».
#
# Метка лежит прямо в имени файла и не требует ни внешних данных, ни разметки.
# Смысл задачи: публичные forensic-веса нам не подошли (шесть проверенных
# моделей дают Dice 0.16-0.38 против наших 0.92), потому что обучены на
# сплайсинге, а у нас на 100% инпейнтинг. Знание «чем один генеративный процесс
# отличается от другого» неоткуда импортировать — но его можно произвести из
# собственных данных.
#
# Порядок важен: побеждает ПЕРВОЕ совпадение, поэтому конкретные генераторы
# идут раньше общего «inpainted».
GENERATOR_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("brushnet", ("brushnet",)),
    ("powerpaint", ("powerpaint",)),
    ("hdpainter", ("hdpainter",)),
    ("inpaintanything", ("inpaintanything",)),
    ("lama", ("lama",)),
    ("inpainted", ("inpainted",)),
]
# 0 — чистый кадр, 1 — числовое семейство (имя вида \d+.jpg, генератор
# неизвестен), дальше по списку, последний класс — подделка без опознанной метки
GEN_CLEAN, GEN_NUMERIC = 0, 1
GEN_CLASSES = ["чистый", "числовое"] + [n for n, _ in GENERATOR_PATTERNS] + ["прочее"]
GEN_OTHER = len(GEN_CLASSES) - 1
_NUMERIC_NAME = re.compile(r"^\d+$")


def generator_label(chng_img_path: str, gt_path: str) -> int:
    """Класс генератора по имени файла; 0 если кадр чистый."""
    if not str(gt_path).strip() or str(gt_path).strip() == "nan":
        return GEN_CLEAN
    stem = Path(chng_img_path).stem.lower()
    if _NUMERIC_NAME.match(stem):
        return GEN_NUMERIC
    for i, (_, keys) in enumerate(GENERATOR_PATTERNS):
        if any(k in stem for k in keys):
            return i + 2
    return GEN_OTHER


class ForgeryTrainDataset(Dataset):
    """Пары (изменённое изображение, GT-маска) для обучения и валидации."""

    def __init__(self, df: pd.DataFrame, root: Path, transform: A.Compose,
                 img_size: int | None = None, no_upscale: bool = False,
                 copy_paste_prob: float = 0.0, seed: int = 42,
                 canvas: tuple[int, int] | None = None,
                 native_crop: bool | float = False,
                 positive_crop_prob: float = 0.0,
                 positive_zoom_range: tuple[float, float] = (1.25, 3.0),
                 gen_labels: bool = False,
                 dct_channel: bool = False,
                 numeric_transform: A.Compose | None = None,
                 numeric_crop_prob: float = 0.0,
                 stage2_edit_prob: float = 0.0,
                 stage2_edit_policy: str = 'original',
                 stage2_focus: bool = False):
        self.df = df.reset_index(drop=True)
        # метки генератора считаем один раз при сборке: разбор имени на каждый
        # __getitem__ — это лишняя работа в воркере на каждый кадр каждой эпохи
        self.gen_labels = None
        if gen_labels:
            self.gen_labels = np.array(
                [generator_label(a, b) for a, b in
                 zip(self.df["chng_img_path"].astype(str),
                     self.df["gt_path"].astype(str))], dtype=np.int64)
        self.root = Path(root)
        self.transform = transform
        self.numeric_transform = numeric_transform
        self.numeric_crop_prob = numeric_crop_prob
        self.numeric = np.asarray([Path(p).stem.isdigit()
                                   for p in self.df["chng_img_path"].astype(str)])
        self.img_size = img_size
        self.no_upscale = no_upscale
        self.copy_paste_prob = copy_paste_prob
        self.stage2_edit_prob = stage2_edit_prob
        self.stage2_focus = stage2_focus
        self.stage2_edit_policy = stage2_edit_policy
        if stage2_edit_policy not in ('original', 'diverse'):
            raise ValueError('stage2_edit_policy must be original or diverse')
        if not 0 <= stage2_edit_prob <= 1:
            raise ValueError('stage2_edit_prob must be in [0,1]')
        self.seed = seed
        self.canvas = canvas
        # native_crop может быть долей: тогда кадр отдаётся в родном масштабе с
        # такой вероятностью, а иначе сжимается как на инференсе. Это масштабное
        # дрожание: модель видит подделку в разных масштабах, но при этом знает
        # и тот масштаб, в котором к ней придут тестовые кадры. Чистый режим
        # native_crop=True учит только родному масштабу, а на инференсе крупный
        # кадр всё равно приходит сжатым — рассогласование.
        # native_crop: кадр крупнее холста НЕ сжимается, а отдаётся как есть —
        # окно вырежет RandomCrop из трансформа. Так крупный кадр даёт пример в
        # РОДНОМ разрешении, без потери высоких частот, по которым модель и
        # работает. Замер на окнах 640x640 из кадров >=1000 px: подделка попадает
        # целиком в 25.9% окон, частично в 68.9%, а 5.3% окон оказываются чистыми
        # — то есть попутно берутся негативы нужного размера, которых у нас мало.
        self.native_crop = native_crop
        self.positive_crop_prob = positive_crop_prob
        self.positive_zoom_range = positive_zoom_range
        # карта нецелости DCT как дополнительный вход: считается на РОДНОМ
        # разрешении и потом проходит те же геометрические преобразования, что
        # и кадр, поэтому остаётся с ним согласованной
        self.dct_channel = dct_channel
        # доноры — только кадры с непустой маской
        gt = self.df["gt_path"].astype(str)
        self.donor_idx = np.flatnonzero(~gt.isin(["", "nan"]).to_numpy())

    def __len__(self) -> int:
        return len(self.df)

    def _load(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        row = self.df.iloc[idx]
        img = _read_rgb(self.root / row["chng_img_path"])
        mask = _read_mask(self.root, str(row["gt_path"]), img.shape[:2])
        if (self.stage2_edit_prob and mask.any()
                and not (self.stage2_focus and mask.mean() < .01)
                and np.random.random() < self.stage2_edit_prob):
            from stage2_augment import local_edit
            rng = np.random.default_rng(np.random.randint(0, 2**31))
            donor_row = self.df.iloc[int(rng.choice(self.donor_idx))]
            donor = _read_rgb(self.root / donor_row.chng_img_path)
            donor_mask = _read_mask(self.root, str(donor_row.gt_path), donor.shape[:2])
            clean = _read_rgb(self.root / row.orgl_img_path)
            img, mask = local_edit(clean, donor, donor_mask, rng, policy=self.stage2_edit_policy)
            if self.stage2_edit_policy == 'original':
                return img, mask
        crop_prob = (self.numeric_crop_prob if self.numeric[idx]
                     and self.numeric_crop_prob > 0.0 else self.positive_crop_prob)
        if (crop_prob > 0.0 and bool(mask.any())
                and np.random.random() < crop_prob):
            rng = np.random.default_rng(np.random.randint(0, 2 ** 31))
            img, mask = positive_zoom_crop(
                img, mask, rng, self.positive_zoom_range)
        if self.stage2_focus:
            from stage2_focus import native_export
            rng = np.random.default_rng(np.random.randint(0, 2**31))
            img, mask = native_export(img, mask, rng)
        elif self.stage2_edit_policy == 'diverse' and np.random.random() < .2:
            # Apply equally to clean and edited frames, before model resizing.
            quality = int(np.random.randint(70, 96))
            ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                       [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                raise RuntimeError('native JPEG augmentation failed')
            img = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if self.canvas:
            img, mask, _ = rect_fit(img, mask, self.canvas)
        elif self.native_crop and (self.native_crop is True
                                   or np.random.random() < float(self.native_crop)):
            pass                                   # окно вырежет RandomCrop
        elif self.no_upscale and self.img_size:
            img, mask = downscale_only(img, self.img_size, mask)
        return img, mask

    def __getitem__(self, idx: int):
        img, mask = self._load(idx)
        transform = (self.numeric_transform if self.numeric[idx]
                     and self.numeric_transform is not None else self.transform)

        if self.copy_paste_prob > 0 and len(self.donor_idx):
            # общий поток случайности процесса-воркера: он засеян из глобального
            # сида, но меняется от эпохи к эпохе — иначе один и тот же кадр
            # получал бы одну и ту же вставку все 10 эпох
            rng = np.random.default_rng(np.random.randint(0, 2 ** 31))
            if rng.random() < self.copy_paste_prob:
                d = int(rng.choice(self.donor_idx))
                if d != idx:
                    donor_img, donor_mask = self._load(d)
                    img, mask = copy_paste(img, mask, donor_img, donor_mask, rng)

        if self.dct_channel:
            # карта DCT идёт ОТДЕЛЬНОЙ целью типа "mask": так она проходит
            # геометрию (ресайз, кроп, отражения, повороты) вместе с кадром, но
            # не попадает под фотометрию. Это и правильно по смыслу — сдвигать
            # яркость статистике незачем, — и необходимо технически:
            # HueSaturationValue не принимает четырёхканальный вход.
            d = dct_lattice_map(self.root / self.df.iloc[idx]["chng_img_path"],
                                img.shape[:2], cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
            out = transform(image=img, mask=mask, dct=d)
            dct_t = ((out["dct"].float() - DCT_MEAN) / DCT_STD).unsqueeze(0)
            image_t = torch.cat([out["image"], dct_t], dim=0)
            if self.gen_labels is not None:
                return image_t, out["mask"].float().unsqueeze(0), int(self.gen_labels[idx])
            return image_t, out["mask"].float().unsqueeze(0)

        out = transform(image=img, mask=mask)
        if self.stage2_focus and mask.any() and not out['mask'].any():
            raise RuntimeError(f'Positive GT disappeared after transform: {idx}')
        if self.gen_labels is not None:
            # copy-paste смешивает два кадра, поэтому метка генератора у склейки
            # была бы неверной; при gen_labels копипаста не используется
            return (out["image"], out["mask"].float().unsqueeze(0),
                    int(self.gen_labels[idx]))
        return out["image"], out["mask"].float().unsqueeze(0)


def soft_mask_path(soft_dir: Path, chng_img_path: str) -> Path:
    """Путь к мягкой маске учителя: зеркалит структуру датасета, расширение .png."""
    return Path(soft_dir) / (Path(chng_img_path).with_suffix("").as_posix() + ".png")


_KD_GEN = ("powerpaint", "inpaintanything", "removeanything", "hdpainter", "brushnet")


def data_family(chng_img_path: str) -> str:
    """Семейство подделки по имени файла — ТОЛЬКО для обучения.

    В тесте имена это 32-символьные хеши, семейства там не видно, поэтому
    признак применим к взвешиванию лосса, но НЕ к маршрутизации на инференсе.
    Классификация дословно повторяет ту, на которой считался разбор 11.09
    (src/teacher_gap_4210.py), иначе веса встанут не на те кадры.
    """
    name = str(chng_img_path).rsplit("/", 1)[-1]
    low = name.lower()
    if any(g in low for g in _KD_GEN):
        return "диффуз. inpaint"
    if re.search(r"_(large|medium|small)_\d+_inpainted", low):
        return "с меткой размера"
    if re.match(r"^\d{6,9}\.", name):
        return "9-значное"
    return "прочее"


class ForgeryDistillDataset(Dataset):
    """То же, что ``ForgeryTrainDataset``, плюс мягкая маска ансамбля учителей.

    Приведение кадра к сетке модели делается ЗДЕСЬ, а не внутри albumentations:
    мягкая маска хранится уже в этой сетке, и повторный ресайз внутри пайплайна
    прошёл бы по ней ближайшим соседом. Размер считается тем же
    ``content_size``, что использует ``LongestMaxSize`` (сверено на 400
    случайных размерах — совпадает пиксель в пиксель), поэтому геометрия
    обучения не меняется и сравнение с контрольным прогоном остаётся честным.

    Трансформ обязан быть построен с ``prescaled=True`` и
    ``additional_targets={"soft": "mask"}``.
    """

    def __init__(self, df: pd.DataFrame, root: Path, transform: A.Compose,
                 img_size: int, soft_dir: Path, no_upscale: bool = False,
                 squish: bool = False, kd_families: set[str] | None = None,
                 soft_resize: bool = False):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform
        self.img_size = img_size
        self.soft_dir = Path(soft_dir)
        self.no_upscale = no_upscale
        # squish добавлен 05.09: при нём кадр растягивается в квадрат целиком,
        # содержимое занимает весь холст, и мягкая маска учителя хранится
        # размером img_size x img_size. Без этого content_size вернул бы
        # прямоугольник письма и проверка размера ниже валила бы обучение.
        self.squish = squish
        # Семейства, на которых дистилляция ВКЛЮЧЕНА. None — на всех, как раньше.
        # Замер 11.09 на общей отложенной 4210: учитель против FIN1 даёт
        # +0.0076 на диффузионном, +0.0095 на прочем, но −0.0015 на 9-значном
        # и −0.0030 на «с меткой размера». Равномерная дистилляция усредняет
        # это в одно число и на 45% кадров учит подражать модели СЛАБЕЕ себя.
        self.kd_families = kd_families
        self.soft_resize = soft_resize

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = _read_rgb(self.root / row["chng_img_path"])
        mask = _read_mask(self.root, str(row["gt_path"]), img.shape[:2])
        h, w = img.shape[:2]
        ch, cw = ((self.img_size, self.img_size) if self.squish
                  else content_size(h, w, self.img_size, self.no_upscale))
        if (ch, cw) != (h, w):
            img = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (cw, ch), interpolation=cv2.INTER_NEAREST)

        path = soft_mask_path(self.soft_dir, str(row["chng_img_path"]))
        soft = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if soft is None:
            raise FileNotFoundError(f"нет мягкой маски: {path}")
        if soft.shape[:2] != (ch, cw):
            if not self.soft_resize:
                raise ValueError(f"мягкая маска {path} размера {soft.shape[:2]}, "
                                 f"а кадр в сетке модели {(ch, cw)} — маски считались "
                                 f"под другой img_size")
            # Осознанная пересборка сетки: обе карты — полный кадр после squish,
            # поэтому это чистая передискретизация одного и того же поля, а не
            # склейка разных геометрий. Билинейно, потому что цель вероятностная.
            # Нужно, чтобы дообучать 672-модель на масках, посчитанных под 704:
            # пересчитывать 138 265 масок тремя учителями по 1234 GFLOPs — сутки.
            soft = cv2.resize(soft, (cw, ch), interpolation=cv2.INTER_LINEAR)

        out = self.transform(image=img, mask=mask, soft=soft)
        kd_w = 1.0
        if self.kd_families is not None:
            kd_w = float(data_family(str(row["chng_img_path"])) in self.kd_families)
        return (out["image"],
                out["mask"].float().unsqueeze(0),
                out["soft"].float().unsqueeze(0) / 255.0,
                torch.tensor(kd_w, dtype=torch.float32))


class ForgeryCropNegDataset(Dataset):
    """Чистые кадры, вырезанные из ПОДДЕЛАННЫХ: область вдали от GT-маски.

    Зачем. Все наши негативы — оригиналы из ``train/src/``, а они принадлежат
    только диффузионному семейству и мелкие: медиана 3.06e5 пикселей против
    6.79e5 у подделанных. Крупные подделки живут в числовом семействе, у
    которого оригиналов нет вовсе. Итог измерен: ложные тревоги на КРУПНЫХ
    чистых кадрах случаются в 11.9 раз чаще (0.0513 против 0.0043), а на
    перевзвешенной под тест валидации FPR удваивается — 0.0211 против 0.0105,
    то есть около −0.005 AIC.

    Взять чистые кадры этого семейства неоткуда, но их можно вырезать: подделка
    занимает там около 10% площади, остальное — чистое содержимое ровно нужного
    распределения. Внешних данных не привлекается.

    Резать надо осторожно, иначе вернём старый ярлык. Вырезка мельче кадра, а
    ``LongestMaxSize`` дотягивает длинную сторону до ``img_size`` — если резать
    как попало, вырезки начнут растягиваться вверх, терять высокие частоты и
    снова помечать себя как «чистые». Поэтому режем ТОЛЬКО поперёк короткой
    стороны, сохраняя длинную, и лишь у кадров, чья длинная сторона и так не
    меньше ``img_size``. Кадры, где так не выходит, пропускаем.
    """

    def __init__(self, df: pd.DataFrame, root: Path, transform: A.Compose,
                 img_size: int, min_frac: float = 0.25, margin: int = 24,
                 tries: int = 8, boxes: pd.DataFrame | None = None):
        self.root = Path(root)
        self.transform = transform
        self.img_size = img_size
        self.min_frac = min_frac
        self.margin = margin
        self.tries = tries
        if boxes is not None:
            # готовые рамки из src/prepare_crop_negs.py: подходит лишь каждый
            # седьмой кадр, и перебирать их в даталоадере — это лишние чтения
            # JPEG на каждый элемент
            self.boxes = boxes.reset_index(drop=True)
            self.df = None
        else:
            self.boxes = None
            gt = df["gt_path"].astype(str)
            self.df = df[~gt.isin(["", "nan"])].reset_index(drop=True)

    def __len__(self) -> int:
        if self.boxes is not None:
            return len(self.boxes)
        return max(1, len(self.df) // self.tries)

    def _crop_box(self, mask: np.ndarray) -> tuple[int, int, int, int] | None:
        """Наибольший прямоугольник без маски с ПРАВДОПОДОБНЫМ соотношением сторон.

        Соотношение обязательно: если резать просто «полосу подальше от маски»,
        получаются ленты 6:1 при настоящих 1.5:1 — и мы заменим один ярлык
        другим, «вытянутый кадр значит чистый».

        Требование длинной стороны >= img_size тоже обязательно: иначе вырезка
        растягивается вверх и приносит ярлык «нет высоких частот -> чистый»,
        который мы и чиним. Ослабление до 0.85 поднимало выход с 7% до 15%, но
        медиана вырезки падала с 513 до 375 тысяч пикселей, то есть к
        распределению НАШИХ чистых кадров (306 тыс.) вместо тестового (531).
        """
        h, w = mask.shape[:2]
        if max(h, w) < self.img_size:
            return None
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            return None
        y0, y1 = max(0, ys.min() - self.margin), min(h, ys.max() + self.margin)
        x0, x1 = max(0, xs.min() - self.margin), min(w, xs.max() + self.margin)

        best = None
        for ar in (1.33, 1.5, 1.78):
            for long_side in (self.img_size, int(self.img_size * 1.15),
                              int(self.img_size * 1.23)):
                cw, ch = (long_side, int(long_side / ar)) if w >= h else \
                         (int(long_side / ar), long_side)
                if cw > w or ch > h:
                    continue
                xs_try = np.unique(np.linspace(0, w - cw, 6).astype(int))
                ys_try = np.unique(np.linspace(0, h - ch, 6).astype(int))
                for cx in xs_try:
                    for cy in ys_try:
                        if cx < x1 and cx + cw > x0 and cy < y1 and cy + ch > y0:
                            continue
                        if best is None or cw * ch > best[2] * best[3]:
                            best = (int(cx), int(cy), cw, ch)
        return best

    def __getitem__(self, idx: int):
        if self.boxes is not None:
            r = self.boxes.iloc[idx]
            img = _read_rgb(self.root / str(r["chng_img_path"]))
            patch = img[int(r["y"]):int(r["y"]) + int(r["h"]),
                        int(r["x"]):int(r["x"]) + int(r["w"])]
            out = self.transform(image=patch,
                                 mask=np.zeros(patch.shape[:2], np.uint8))
            return out["image"], out["mask"].float().unsqueeze(0)
        n = len(self.df)
        for k in range(self.tries):
            row = self.df.iloc[(idx * self.tries + k) % n]
            img = _read_rgb(self.root / row["chng_img_path"])
            mask = _read_mask(self.root, str(row["gt_path"]), img.shape[:2])
            box = self._crop_box(mask)
            if box is not None:
                x, y, bw, bh = box
                patch = img[y:y + bh, x:x + bw]
                out = self.transform(image=patch,
                                     mask=np.zeros(patch.shape[:2], np.uint8))
                return out["image"], out["mask"].float().unsqueeze(0)
        row = self.df.iloc[idx % n]
        img = _read_rgb(self.root / row["chng_img_path"])
        mask = _read_mask(self.root, str(row["gt_path"]), img.shape[:2])
        out = self.transform(image=img, mask=mask)
        return out["image"], out["mask"].float().unsqueeze(0)


class ForgeryEvalDataset(Dataset):
    """Валидация в исходном разрешении: отдаём тензор модели + GT-маску как есть.

    Метрика считается на полном размере, поэтому GT НЕ ресайзим — вместо этого
    на этапе оценки поднимаем вероятностную карту до размера кадра.
    """

    def __init__(self, df: pd.DataFrame, root: Path, img_size: int, keep_aspect: bool = True,
                 no_upscale: bool = False, canvas: tuple[int, int] | None = None,
                 dct_channel: bool = False):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.img_size = img_size
        self.keep_aspect = keep_aspect
        self.no_upscale = no_upscale
        self.canvas = canvas
        self.dct_channel = dct_channel
        self.transform = valid_transform(img_size, keep_aspect, no_upscale, canvas,
                                         additional_targets={"dct": "mask"} if dct_channel else None)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = _read_rgb(self.root / row["chng_img_path"])
        mask = _read_mask(self.root, str(row["gt_path"]), img.shape[:2])
        h, w = img.shape[:2]              # исходный размер: метрика считается в нём

        if self.canvas:
            # rot возвращается вместе с размерами: без него предсказание
            # портретного кадра пришло бы повёрнутым на 90 градусов
            img, _, rot = rect_fit(img, None, self.canvas)
            ch, cw, _ = rect_content_size(h, w, self.canvas)
            tensor = self.transform(image=img)["image"]
            return tensor, torch.from_numpy(mask), idx, ch, cw, rot
        if self.no_upscale:
            img, _ = downscale_only(img, self.img_size)
        if self.dct_channel:
            d = dct_lattice_map(self.root / row["chng_img_path"], img.shape[:2],
                                cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
            o = self.transform(image=img, dct=d)
            tensor = torch.cat([o["image"],
                                ((o["dct"].float() - DCT_MEAN) / DCT_STD).unsqueeze(0)], dim=0)
        else:
            tensor = self.transform(image=img)["image"]
        ch, cw = (content_size(h, w, self.img_size, self.no_upscale) if self.keep_aspect
                  else (self.img_size, self.img_size))
        return tensor, torch.from_numpy(mask), idx, ch, cw, False


class ForgeryTestDataset(Dataset):
    """Тестовые изображения: тензор + исходный размер для обратного ресайза."""

    def __init__(self, df: pd.DataFrame, root: Path, img_size: int, keep_aspect: bool = True,
                 no_upscale: bool = False, canvas: tuple[int, int] | None = None,
                 dct_channel: bool = False):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.img_size = img_size
        self.keep_aspect = keep_aspect
        self.no_upscale = no_upscale
        self.canvas = canvas
        self.dct_channel = dct_channel
        self.transform = valid_transform(img_size, keep_aspect, no_upscale, canvas,
                                         additional_targets={"dct": "mask"} if dct_channel else None)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = _read_rgb(self.root / row["img_path"])
        h, w = img.shape[:2]              # исходный размер для обратного ресайза
        if self.canvas:
            img, _, rot = rect_fit(img, None, self.canvas)
            ch, cw, _ = rect_content_size(h, w, self.canvas)
            return self.transform(image=img)["image"], h, w, idx, ch, cw, rot
        if self.no_upscale:
            img, _ = downscale_only(img, self.img_size)
        if self.dct_channel:
            # 06.09: параметр dct_channel здесь принимался, но карта не
            # считалась — датасет всегда отдавал три канала, и инференс
            # четырёхканальной модели падал прямо в первом conv. Валидация при
            # этом работала (ForgeryEvalDataset карту считает), так что дефект
            # жил ровно до первой попытки собрать посылку.
            d = dct_lattice_map(self.root / row["img_path"], img.shape[:2],
                                cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
            o = self.transform(image=img, dct=d)
            tensor = torch.cat([o["image"],
                                ((o["dct"].float() - DCT_MEAN) / DCT_STD).unsqueeze(0)], dim=0)
        else:
            tensor = self.transform(image=img)["image"]
        ch, cw = (content_size(h, w, self.img_size, self.no_upscale) if self.keep_aspect
                  else (self.img_size, self.img_size))
        return tensor, h, w, idx, ch, cw, False


def eval_collate(batch):
    """GT-маски разного размера — их нельзя стекать, поэтому оставляем списком.

    В content кладём тройку (высота, ширина, повёрнут ли кадр): на прямоугольном
    холсте портретные кадры разворачиваются в ландшафт, и предсказание надо
    повернуть обратно перед сравнением с GT.
    """
    tensors = torch.stack([b[0] for b in batch])
    masks = [b[1] for b in batch]
    idxs = [b[2] for b in batch]
    content = [(b[3], b[4], b[5] if len(b) > 5 else False) for b in batch]
    return tensors, masks, idxs, content


def test_collate(batch):
    tensors = torch.stack([b[0] for b in batch])
    sizes = [(b[1], b[2]) for b in batch]
    # content — тройка (высота, ширина, повёрнут ли): на прямоугольном холсте
    # портретные кадры разворачивались в ландшафт
    idxs = [b[3] for b in batch]
    content = [(b[4], b[5], b[6] if len(b) > 6 else False) for b in batch]
    return tensors, sizes, idxs, content

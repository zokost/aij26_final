"""Инференс финала: холсты задаются параметром, а не зашиты в модуль.

Почему отдельно от ``product/core.py``. Там трио холстов — константа модуля,
посчитанная под вход 704, а на финале сторона обучения меняется (mit_b1 учится
на 832), и трио обязано пересчитываться вместе с ней. Сервис при этом решено не
трогать, поэтому геометрия вынесена сюда с параметром.

Расхождение двух реализаций препроцессинга однажды уже стоило дорого, поэтому
здесь есть ``assert_matches_product``: на трио по умолчанию этот класс обязан
давать побитово те же карты вероятностей, что сервис. Проверка запускается
как ``python3 src/detector2.py``.
"""
from __future__ import annotations

import os

# Ставится ДО импорта torch: переменная читается при создании cuBLAS-handle.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from data import IMAGENET_MEAN, IMAGENET_STD
from model import build_model, count_flops

# Трио отбора, посчитанное под вход 704: (высота, ширина).
DEFAULT_CANVASES = ((832, 608), (704, 704), (608, 832))
# Пропорции трио: портрет, квадрат, ландшафт. |log(W/H)| = 0.313, то есть
# отношение сторон 1.368. Границы выбора по W/H выходят 0.855 и 1.17.
ASPECTS = (608 / 832, 1.0, 832 / 608)
STRIDE = 32          # вход SegFormer обязан делиться на 32


def trio_for_side(side: int, stride: int = STRIDE) -> tuple[tuple[int, int], ...]:
    """Трио холстов для модели, обученной на квадрате ``side``.

    Площадь каждого холста держится около ``side**2``, пропорции — те же, что у
    трио отбора. Стороны округляются к кратному 32: это требование входа, а не
    косметика. Для 704 функция обязана вернуть ровно трио отбора.
    """
    out = []
    for aspect in ASPECTS:
        height = side / np.sqrt(aspect)
        width = height * aspect
        out.append((int(round(height / stride)) * stride,
                    int(round(width / stride)) * stride))
    return tuple(out)


def pick_canvas(h: int, w: int, canvases: tuple[tuple[int, int], ...]) -> int:
    """Холст, ближайший кадру по логарифму отношения сторон."""
    log_ar = np.log([cw / ch for ch, cw in canvases])
    return int(np.argmin(np.abs(np.log(w / h) - log_ar)))


class Detector2:
    """Та же арифметика, что в сервисе, но трио холстов — аргумент."""

    def __init__(self, checkpoint: str | Path, encoder: str = 'mit_b2',
                 img_size: int = 704, decoder_channels: int | None = 128,
                 canvases: tuple[tuple[int, int], ...] | None = None,
                 square: bool = False, device: str = 'cuda') -> None:
        torch.manual_seed(42)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        cfg = state.get('cfg') or {}
        # Геометрия и форма модели берутся из чекпоинта: передавать их руками —
        # способ однажды собрать модель не той ширины и не заметить.
        self.encoder = cfg.get('encoder', encoder)
        self.img_size = int(cfg.get('img_size', img_size))
        self.decoder_channels = cfg.get('decoder_channels', decoder_channels)
        # Обёртки модели тоже лежат в cfg, и их НЕЛЬЗЯ игнорировать: чекпоинт
        # прогона с --refine это RefineNet (ключи backbone.* плюс refine.*), и
        # загрузка в обычный Segformer падает на каждом тензоре. Молча ставить
        # strict=False было бы хуже всего — модель собралась бы со случайными
        # весами и тихо выдавала мусор.
        self.refine = bool(cfg.get('refine', False))
        self.canvases = canvases or trio_for_side(self.img_size)
        self.square = square
        self.device = device
        self.model = build_model(cfg.get('arch', 'segformer'), self.encoder, None,
                                 self.img_size, decoder_channels=self.decoder_channels,
                                 refine=self.refine)
        self.model.load_state_dict(state['model'], strict=True)
        self.model.to(device).eval()
        self.mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
        self.std = np.asarray(IMAGENET_STD, dtype=np.float32)

    def budget(self) -> dict[str, float]:
        """Строгие GFLOPs на каждом холсте, который реально может встретиться."""
        used = [self.canvases[1]] if self.square else list(self.canvases)
        out = {}
        for ch, cw in used:
            flops = count_flops(self.model.cpu(), (ch, cw), device='cpu') / 1e9
            out[f'{ch}x{cw}'] = flops
        self.model.to(self.device)
        return out

    def _tensor(self, img: np.ndarray, canvas: int) -> torch.Tensor:
        ch, cw = self.canvases[canvas]
        small = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_AREA)
        small = (small.astype(np.float32) / 255.0 - self.mean) / self.std
        return torch.from_numpy(small.transpose(2, 0, 1))

    @torch.inference_mode()
    def prob_maps(self, images: list[np.ndarray], batch_size: int = 8) -> list[np.ndarray]:
        out: list[np.ndarray | None] = [None] * len(images)
        groups: dict[int, list[int]] = {c: [] for c in range(len(self.canvases))}
        for i, img in enumerate(images):
            h, w = img.shape[:2]
            groups[1 if self.square else pick_canvas(h, w, self.canvases)].append(i)
        for canvas, idx in groups.items():
            for start in range(0, len(idx), batch_size):
                chunk = idx[start:start + batch_size]
                batch = torch.stack([self._tensor(images[i], canvas) for i in chunk])
                logits = self.model(batch.to(self.device, non_blocking=True))
                for k, i in enumerate(chunk):
                    h, w = images[i].shape[:2]
                    piece = F.interpolate(logits[k:k + 1].float(), size=(h, w),
                                          mode='bilinear', align_corners=False)
                    out[i] = torch.sigmoid(piece)[0, 0].cpu().numpy()
        assert all(p is not None for p in out)
        return out  # type: ignore[return-value]


def assert_matches_product(checkpoint: str, images: list[np.ndarray]) -> None:
    """На трио по умолчанию карты обязаны совпасть с сервисом побитово.

    Сервис по умолчанию добивает неполный батч до степени двойки (это его
    оптимизация под поток запросов), а от размера батча зависит выбор ядра
    cuDNN — поэтому сравнивать надо с ``pad_batch=False``. Разница от добивки
    тоже печатается: она должна быть на уровне ошибки округления, а не нулём.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.core import CANVASES, Detector

    assert trio_for_side(704) == CANVASES, (trio_for_side(704), CANVASES)
    reference = Detector2(checkpoint)
    mine = reference.prob_maps(images, batch_size=8)
    same = Detector(checkpoint, pad_batch=False, square=False).prob_maps(images, batch_size=8)
    for a, b in zip(mine, same):
        assert a.shape == b.shape and np.array_equal(a, b), 'карты разошлись'
    padded = Detector(checkpoint, pad_batch=True, square=False).prob_maps(images, batch_size=8)
    worst = max(float(np.abs(a - b).max()) for a, b in zip(mine, padded))
    flips = sum(int(((a >= .7) != (b >= .7)).sum()) for a, b in zip(mine, padded))
    pixels = sum(a.size for a in mine)
    print(f'холсты {reference.canvases} совпадают с сервисом, карты побитово равны на {len(images)} кадрах\n'
          f'добивка батча меняет вероятность не более чем на {worst:.2e}; '
          f'пикселей за порогом 0.7 разошлось {flips} из {pixels}')


if __name__ == '__main__':
    import sys

    import pandas as pd

    # корень репозитория в пути: пакет product лежит рядом с src
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import TEST_DIR
    from app.core import DEFAULT_CHECKPOINT

    print('трио для 704:', trio_for_side(704))
    print('трио для 832:', trio_for_side(832))
    print('трио для 960:', trio_for_side(960))
    print('трио для 1024:', trio_for_side(1024))
    paths = pd.read_csv(TEST_DIR / 'test.csv').img_path.astype(str).tolist()[:8]
    frames = [cv2.cvtColor(cv2.imread(str(TEST_DIR / p), cv2.IMREAD_COLOR),
                           cv2.COLOR_BGR2RGB) for p in paths]
    assert_matches_product(str(DEFAULT_CHECKPOINT), frames)

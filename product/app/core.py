"""Ядро продуктовой обёртки: изображение на входе, вероятностная карта на выходе.

Загрузка модели и препроцессинг общие с ``src/detector2.py`` и посылкой
``src/infer_stage2.py``. По умолчанию используется победитель второго этапа:
NC17 epoch32, MiT-B2/decoder320, square640, FP32 и калибратор посылки №121.
"""
from __future__ import annotations

import os

# Ставится ДО импорта torch: переменная читается при создании cuBLAS-handle.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import json
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from detector2 import DEFAULT_CANVASES, Detector2, pick_canvas as choose_canvas  # noqa: E402
from nc17_calibration import ARTIFACT_SHA256, FrozenCalibration  # noqa: E402

# (высота, ширина); выбор — argmin |log((W/H)/(Wr/Hr))|, что для этих трёх
# холстов равносильно порогам 0.855 и 1.17 по W/H.
CANVASES = DEFAULT_CANVASES
DEFAULT_CHECKPOINT = ROOT / 'weights/121_NC17_e32/model.pt'
PREVIOUS_CHECKPOINT = ROOT / 'weights/LB08023_b2_640_dec320_nbA2_minaug_e40.pt'
V4_CHECKPOINT = ROOT / 'experiments/results/S2_V4_AUG10_20261015/best.pt'


def pick_canvas(h: int, w: int, canvases=CANVASES) -> int:
    return choose_canvas(h, w, canvases)


@dataclass
class Region:
    """Связная компонента маски: то, что показывается оператору."""
    x: int
    y: int
    width: int
    height: int
    area_px: int
    area_frac: float
    mean_prob: float
    max_prob: float


@dataclass
class Detection:
    score: float                    # шкала описана в metadata.score_kind; не подтверждённая вероятность
    manipulated: bool               # решение по рабочей точке
    area_frac: float                # доля кадра под маской
    regions: list[Region] = field(default_factory=list)
    mask: np.ndarray | None = None  # uint8 0/1 в исходном размере кадра
    elapsed_ms: float = 0.0
    threshold: float | None = None


class Detector(Detector2):
    """Загруженная модель, готовая отвечать на кадры.

    По умолчанию: NC17 square640, базовый порог .675 и lower-only калибратор №121.
    ``min_area_frac`` задаёт совместимый флаг площади, не фильтрует маску:
    он остаётся true только при площади больше 1%.
    """

    def __init__(self, checkpoint: str | Path = DEFAULT_CHECKPOINT, device: str = 'cuda',
                 threshold: float | None = None, min_area_frac: float = .01,
                 square: bool | None = None, deterministic: bool = True,
                 fast: bool = False, preprocess_threads: int = 4,
                 pad_batch: bool = True) -> None:
        # Строгий детерминизм и скорость — обмен, а не настройка по вкусу.
        # Замер на A100, холст 608x832: 26.3 мс на кадр при батче 1 строго
        # против 20.1 мс без детерминизма; при батче 8 — 14.6 против 7.0 мс на
        # кадр с TF32. По умолчанию строгий: посылку надо воспроизводить.
        assert not (fast and deterministic), 'fast отключает детерминизм, включать оба нельзя'
        self.checkpoint = Path(checkpoint)
        recipe_threshold = {
            DEFAULT_CHECKPOINT.resolve(): .675,
            PREVIOUS_CHECKPOINT.resolve(): .93,
            V4_CHECKPOINT.resolve(): .95,
        }.get(self.checkpoint.resolve())
        calibrated = self.checkpoint.resolve() == DEFAULT_CHECKPOINT.resolve()
        if calibrated and (threshold not in (None, .675) or square is False or fast):
            raise ValueError('Submission121 requires square640, FP32 and frozen calibration; remove threshold/fast overrides.')
        super().__init__(self.checkpoint, device=device,
                         square=(recipe_threshold is not None) if square is None else square)
        torch.backends.cudnn.benchmark = fast
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cuda.matmul.allow_tf32 = fast
        torch.use_deterministic_algorithms(deterministic)
        self.threshold = threshold if threshold is not None else (recipe_threshold or .70)
        self.min_area_frac = min_area_frac
        self.fast = fast
        self.pad_batch = pad_batch
        self.frozen_calibration = FrozenCalibration(self.checkpoint.parent, self.model) if calibrated else None
        # Сохраняем прежнюю шкалу только для двух исходных путей этапа 1.
        legacy = self.checkpoint.resolve() in {
            (ROOT / 'submission/best.pt').resolve(), (ROOT / 'submission/stage1_e161.pt').resolve()}
        self.calibration = load_calibration() if legacy else dict(FALLBACK)
        score_kind = 'historical_platt_unvalidated' if self.calibration.get('platt') else 'uncalibrated_top_logit'
        self.metadata = {'architecture': type(self.model).__name__, 'encoder': self.encoder,
                         'decoder_channels': self.decoder_channels, 'img_size': self.img_size,
                         'refine': self.refine, 'square': self.square,
                         'canvases': [list(c) for c in self.canvases],
                         'active_canvases': [list(self.canvases[1])] if self.square else
                             [list(c) for c in self.canvases],
                         'threshold': self.threshold, 'precision': 'fp32', 'device': self.device,
                         'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                         'cudnn_tf32': torch.backends.cudnn.allow_tf32,
                         'score_kind': score_kind, 'score_calibrated': False}
        if calibrated:
            self.metadata.update(threshold_mode='lower_only_calibrated', base_threshold=.675,
                                 candidate_thresholds=[.5, .6, .65, .675], calibration_sha256=ARTIFACT_SHA256,
                                 inference_batch=1)
        # Подготовка кадра — 8.5 мс на CPU против 7 мс счёта на карте при батче
        # 8: в один поток препроцесс становится узким местом. cv2 отпускает GIL.
        self.pool = ThreadPoolExecutor(preprocess_threads) if preprocess_threads > 1 else None

    def warmup(self, rounds: int = 2, batch_sizes: tuple[int, ...] = (1, 2, 4, 8)) -> None:
        """Прогревает КАЖДУЮ форму входа, которая может встретиться в работе.

        Прогреть только батч 1 недостаточно: cuDNN подбирает алгоритм под пару
        (холст, размер батча), и в режиме benchmark этот подбор исчерпывающий.
        Непрогретый сервис в режиме fast отвечал на отдельные запросы по 17
        секунд, пока перебирал алгоритмы под новую форму.
        """
        with torch.inference_mode():
            for ch, cw in ([self.canvases[1]] if self.square else self.canvases):
                for batch in batch_sizes:
                    dummy = torch.zeros(batch, 3, ch, cw, device=self.device)
                    for _ in range(rounds):
                        self.model(dummy)
        if self.device.startswith('cuda'):
            torch.cuda.synchronize()

    def _to_tensor(self, img: np.ndarray, canvas: int) -> torch.Tensor:
        return self._tensor(img, canvas)

    @torch.inference_mode()
    def prob_maps(self, images: list[np.ndarray], batch_size: int = 1,
                  logits_too: bool = False) -> list[np.ndarray] | tuple[list[np.ndarray], list[np.ndarray]]:
        """Вероятностные карты в ИСХОДНОМ размере каждого кадра.

        Кадры группируются по холсту: тензоры одного батча обязаны совпадать по
        форме, а холст у кадров разный.

        ``logits_too`` отдаёт и логиты: сигмоида насыщается в ровно 1.0 уже при
        логите около 17, и по вероятности уверенные кадры неразличимы между
        собой. Для оценки кадра это существенно, для маски — нет.
        """
        out: list[np.ndarray | None] = [None] * len(images)
        raw: list[np.ndarray | None] = [None] * len(images)
        groups: dict[int, list[int]] = {c: [] for c in range(len(self.canvases))}
        for i, img in enumerate(images):
            h, w = img.shape[:2]
            groups[1 if self.square else pick_canvas(h, w, self.canvases)].append(i)
        for canvas, idx in groups.items():
            for start in range(0, len(idx), batch_size):
                chunk = idx[start:start + batch_size]
                prepared = (self.pool.map(lambda i: self._to_tensor(images[i], canvas), chunk)
                            if self.pool and len(chunk) > 1
                            else (self._to_tensor(images[i], canvas) for i in chunk))
                batch = torch.stack(list(prepared))
                # cuDNN подбирает алгоритм и рабочий буфер под КАЖДУЮ новую
                # форму входа. Три холста и произвольный размер батча дают до
                # двух десятков форм, и сервис платит за подбор снова и снова:
                # замер показал 202 мс на кадр против 29 в пакетном режиме.
                # Поэтому неполный батч добивается повтором последнего кадра.
                # Добивать до максимума нельзя: ведро из двух кадров считалось
                # бы как восемь, и цена кадра выросла до 140 мс. Добиваем до
                # ближайшей степени двойки — форм остаётся дюжина, а лишнего
                # счёта не больше чем вдвое.
                filled = len(chunk)
                if self.pad_batch:
                    want = 1 << (filled - 1).bit_length()   # ближайшая степень двойки
                    if want > filled:
                        batch = torch.cat([batch, batch[-1:].expand(want - filled, -1, -1, -1)])
                logits = self.model(batch.to(self.device, non_blocking=True))[:filled]
                for k, i in enumerate(chunk):
                    h, w = images[i].shape[:2]
                    piece = F.interpolate(logits[k:k + 1].float(), size=(h, w),
                                          mode='bilinear', align_corners=False)
                    out[i] = torch.sigmoid(piece)[0, 0].cpu().numpy()
                    if logits_too:
                        raw[i] = piece[0, 0].cpu().numpy()
        assert all(p is not None for p in out)
        return (out, raw) if logits_too else out  # type: ignore[return-value]

    def detect(self, images: list[np.ndarray], batch_size: int = 1,
               with_mask: bool = True, with_regions: bool = True) -> list[Detection]:
        if self.frozen_calibration is not None:
            results = []
            with torch.inference_mode():
                for image in images:
                    began = time.perf_counter()
                    tensor = self._tensor(image, 1)[None].to(self.device)
                    logits, encoder = self.frozen_calibration.net(tensor)
                    native = F.interpolate(logits.float(), size=image.shape[:2], mode='bilinear', align_corners=False)
                    probability = native.sigmoid()[0, 0].cpu().numpy()
                    raw = native[0, 0].cpu().numpy()
                    threshold = self.frozen_calibration.threshold(image, probability, encoder.cpu().numpy())
                    results.append(self.describe(probability, raw, with_mask, with_regions,
                                                 (time.perf_counter() - began) * 1000, threshold))
            return results
        began = time.perf_counter()
        probs, logits = self.prob_maps(images, batch_size, logits_too=True)
        elapsed = (time.perf_counter() - began) * 1000 / max(1, len(images))
        # разбор карты (связные компоненты в полном разрешении) идёт на CPU и
        # для батча стоит столько же, сколько сам счёт на карте
        work = lambda pair: self.describe(*pair, with_mask, with_regions, elapsed)  # noqa: E731
        pairs = list(zip(probs, logits))
        if self.pool and len(pairs) > 1:
            return list(self.pool.map(work, pairs))
        return [work(pair) for pair in pairs]

    def describe(self, prob: np.ndarray, logit: np.ndarray, with_mask: bool = True,
                 with_regions: bool = True, elapsed_ms: float = 0., threshold: float | None = None) -> Detection:
        selected_threshold = self.threshold if threshold is None else threshold
        mask = (prob >= selected_threshold).astype(np.uint8)
        n_pixels = mask.size
        area_frac = float(mask.sum()) / n_pixels
        regions: list[Region] = []
        if with_regions and mask.any():
            n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for c in range(1, n):
                x, y, w, h, area = stats[c]
                inside = prob[labels == c]
                regions.append(Region(int(x), int(y), int(w), int(h), int(area),
                                      area / n_pixels, float(inside.mean()),
                                      float(inside.max())))
            regions.sort(key=lambda r: r.area_px, reverse=True)
        return Detection(score=image_score(logit, self.calibration),
                         manipulated=area_frac > self.min_area_frac,
                         area_frac=area_frac, regions=regions,
                         mask=mask if with_mask else None, elapsed_ms=elapsed_ms, threshold=float(selected_threshold))


CALIBRATION_FILE = Path(__file__).parent / 'calibration.json'
# Запасной вариант на случай, если файла калибровки нет: тот же функционал,
# но без перевода в вероятность. Порядок кадров сохраняется, шкала — нет.
FALLBACK = {'space': 'logit', 'top_fraction': .001, 'platt': None}


def load_calibration(path: Path = CALIBRATION_FILE) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return dict(FALLBACK)


def image_score(logit: np.ndarray, calibration: dict) -> float:
    """Среднее верхних долей логитов; перевод в шкалу 0–1 только с явной калибровкой.

    Функционал, доля и шкала выбраны замером на отложенных 4210 кадрах весами,
    которые её не видели (``product/calibrate.py``), и лежат в
    ``product/calibration.json``. Считается по ЛОГИТАМ: сигмоида упирается в
    ровно 1.0 при логите около 17, и по вероятности уверенные кадры
    неразличимы между собой.
    """
    flat = logit.ravel()
    k = max(1, int(round(flat.size * calibration.get('top_fraction') or .001)))
    raw = (float(flat.mean()) if k >= flat.size
           else float(np.partition(flat, flat.size - k)[flat.size - k:].mean()))
    platt = calibration.get('platt')
    if not platt:
        return raw
    return float(1 / (1 + np.exp(-(platt['a'] * raw + platt['b']))))

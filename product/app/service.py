"""HTTP-сервис детектора: кадр на вход, решение и карта правок на выход.

Отличие от скрипта посылки — в порядке работы. Скрипт читает список файлов и
считает всё разом; сервис держит модель загруженной и отвечает на одиночные
запросы. Одиночный запрос недогружает карту, поэтому между приёмом и счётом
стоит микробатчер: запросы, пришедшие в одном окне ``BATCH_WINDOW_MS``,
считаются одним проходом. Модель одна на процесс, обращения к ней
сериализованы — иначе два потока смешают состояние cuDNN.

    uvicorn app.service:app --host 127.0.0.1 --port 8080
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import mimetypes
import os
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.core import DEFAULT_CHECKPOINT, Detection, Detector, pick_canvas
from app.explanations import availability, describe_region

WEB_DIR = Path(__file__).resolve().parent.parent / 'web'

CHECKPOINT = Path(os.environ.get('DETECTOR_CHECKPOINT', DEFAULT_CHECKPOINT))
THRESHOLD = float(os.environ['DETECTOR_THRESHOLD']) if 'DETECTOR_THRESHOLD' in os.environ else None
MIN_AREA_FRAC = float(os.environ.get('DETECTOR_MIN_AREA_FRAC', '0.01'))
DEVICE = os.environ.get('DETECTOR_DEVICE', 'cuda')
# Локальная проверка на GPU с 6 ГБ: только один кадр за проход.
# MAX_BATCH = int(os.environ.get('DETECTOR_MAX_BATCH', '8'))
MAX_BATCH = 1
BATCH_WINDOW_MS = float(os.environ.get('DETECTOR_BATCH_WINDOW_MS', '8'))
MAX_PIXELS = int(os.environ.get('DETECTOR_MAX_PIXELS', str(15_000_000)))
MAX_FILE_BYTES = 20 * 1024 * 1024
FAST = os.environ.get('DETECTOR_FAST', '0') == '1'
mimetypes.add_type('text/javascript', '.mjs')

state: dict = {'detector': None, 'queue': None, 'gpu': None, 'latency_ms': [], 'requests': 0,
               'manipulated': 0, 'errors': 0, 'batches': 0, 'batched_frames': 0,
               'compute_ms': 0., 'wait_ms': 0.}


async def worker() -> None:
    """Собирает кадры в батчи ПО ХОЛСТАМ и считает каждый батч одним проходом.

    Батчить по порядку прихода нельзя. Холст выбирается по пропорциям кадра, и
    смешанная восьмёрка распадается на три прохода по 6/1/1 кадру, где одиночки
    стоят почти столько же, сколько полный батч. Замер на A100, поток из 256
    кадров теста: по порядку прихода 14.4 кадра/с, с раскладкой по холстам
    34.3 (строгий детерминизм) и 60.9 (режим fast). Поэтому каждое ведро
    копится отдельно и уходит в счёт, когда набралось MAX_BATCH или когда
    самый старый кадр в нём ждёт дольше окна.
    """
    queue: asyncio.Queue = state['queue']
    loop = asyncio.get_running_loop()
    window = BATCH_WINDOW_MS / 1000
    detector = state['detector']
    buckets: dict[int, list] = {c: [] for c in range(len(detector.canvases))}

    def take(item: dict) -> None:
        if item['future'].cancelled():
            return
        h, w = item['image'].shape[:2]
        item['arrived'] = loop.time()
        canvas = 1 if detector.square else pick_canvas(h, w, detector.canvases)
        buckets[canvas].append(item)

    while True:
        if any(len(b) >= MAX_BATCH for b in buckets.values()):
            pass                                   # есть готовый батч, не ждём
        elif any(buckets.values()):
            oldest = min(b[0]['arrived'] for b in buckets.values() if b)
            left = oldest + window - loop.time()
            if left > 0:
                try:
                    take(await asyncio.wait_for(queue.get(), left))
                except asyncio.TimeoutError:
                    pass
        else:
            take(await queue.get())
        while True:                                # добираем всё пришедшее
            try:
                take(queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        for canvas, items in buckets.items():
            if not items:
                continue
            if len(items) < MAX_BATCH and loop.time() - items[0]['arrived'] < window:
                continue
            batch, buckets[canvas] = items[:MAX_BATCH], items[MAX_BATCH:]
            began = loop.time()
            state['wait_ms'] += sum(began - i['arrived'] for i in batch) * 1000
            try:
                # Счёт идёт в ОДНОМ выделенном потоке, а не в общем пуле.
                # cuDNN и cuBLAS держат хэндлы и кэш выбранных алгоритмов
                # по потокам: когда счёт каждый раз попадает на новый поток
                # общего пула (туда же ходит декодирование JPEG), всё это
                # инициализируется заново. Замер: 124 мс на кадр в общем пуле
                # против 31 в выделенном потоке.
                results = await loop.run_in_executor(
                    state['gpu'], state['detector'].detect, [i['image'] for i in batch])
            except Exception as error:             # noqa: BLE001
                logging.exception('Image inference failed')
                for item in batch:
                    if not item['future'].done():
                        item['future'].set_exception(error)
                continue
            state['batches'] += 1
            state['batched_frames'] += len(batch)
            state['compute_ms'] += (loop.time() - began) * 1000
            for item, result in zip(batch, results):
                if not item['future'].done():
                    item['future'].set_result(result)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state['detector'] = Detector(CHECKPOINT, device=DEVICE, threshold=THRESHOLD,
                                 min_area_frac=MIN_AREA_FRAC,
                                 deterministic=not FAST, fast=FAST, preprocess_threads=1)
    state['checkpoint_sha256'] = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    # Прогрев временно отключён для локального запуска без лишних проходов.
    # state['detector'].warmup()
    state['queue'] = asyncio.Queue(maxsize=4)
    state['gpu'] = ThreadPoolExecutor(1, thread_name_prefix='gpu')
    state['predictions'] = OrderedDict()
    state['explanation_lock'] = asyncio.Lock()
    task = asyncio.create_task(worker())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        state['gpu'].shutdown(wait=True)


app = FastAPI(title='Цифровой детектив — детектор ИИ-манипуляций',
              version='1.0', lifespan=lifespan)


def decode(raw: bytes, name: str) -> np.ndarray:
    array = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(415, f'{name}: не распознан как изображение')
    if img.shape[0] * img.shape[1] > MAX_PIXELS:
        raise HTTPException(413, f'{name}: {img.shape[1]}x{img.shape[0]} больше лимита')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


async def run(image: np.ndarray) -> Detection:
    future: asyncio.Future = asyncio.get_running_loop().create_future()
    try:
        state['queue'].put_nowait({'image': image, 'future': future})
    except asyncio.QueueFull:
        raise HTTPException(503, 'Очередь заполнена. Повторите проверку через несколько секунд.')
    return await future


def payload(result: Detection, with_mask: bool) -> dict:
    body = {'manipulated': result.manipulated, 'score': round(result.score, 6),
            'area_frac': result.area_frac,
            'regions': [{k: (round(v, 6) if isinstance(v, float) and k != 'area_frac' else v)
                         for k, v in asdict(r).items()} for r in result.regions[:32]],
            'n_regions': len(result.regions),
            'inference_ms': round(result.elapsed_ms, 2),
            'score_kind': state['detector'].metadata['score_kind'],
            'score_calibrated': state['detector'].metadata['score_calibrated'],
            'model': {'checkpoint': CHECKPOINT.name,
                      'experiment': CHECKPOINT.parent.name,
                      'sha256': state.get('checkpoint_sha256'),
                      **state['detector'].metadata},
            'prediction_id': str(uuid4()),
            'analysed_at': datetime.now(timezone.utc).isoformat()}
    if with_mask and result.mask is not None:
        ok, buffer = cv2.imencode('.png', result.mask * 255)
        assert ok
        body['mask_png_base64'] = base64.b64encode(buffer.tobytes()).decode()
    if getattr(result, 'threshold', None) is not None:
        body['model']['threshold'] = result.threshold
    return body


@app.get('/health')
def health() -> dict:
    detector: Detector | None = state['detector']
    return {'status': 'ok' if detector else 'loading',
            'mode': 'inference', 'pid': os.getpid(), 'checkpoint_sha256': state.get('checkpoint_sha256'),
            'checkpoint': CHECKPOINT.name, 'threshold': detector.threshold if detector else THRESHOLD,
            'model': {'checkpoint': CHECKPOINT.name, 'experiment': CHECKPOINT.parent.name,
                      'sha256': state.get('checkpoint_sha256'),
                      **detector.metadata} if detector else None,
            'min_area_frac': MIN_AREA_FRAC, 'device': DEVICE, 'fast': FAST,
            'max_batch': MAX_BATCH, 'batch_window_ms': BATCH_WINDOW_MS,
            'explanations': availability()}


async def read_image(file: UploadFile) -> tuple[bytes, np.ndarray]:
    raw = await file.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES:
        raise HTTPException(413, 'Изображение превышает 20 МБ.')
    image = await asyncio.to_thread(decode, raw, file.filename or 'file')
    return raw, image


@app.get('/metrics')
def metrics() -> dict:
    latency = sorted(state['latency_ms'])
    def percentile(q: float) -> float:
        return round(latency[min(len(latency) - 1, int(q * len(latency)))], 2) if latency else 0.
    frames = max(1, state['batched_frames'])
    return {'requests': state['requests'], 'manipulated': state['manipulated'],
            'errors': state['errors'], 'batches': state['batches'],
            'mean_batch_size': round(state['batched_frames'] / max(1, state['batches']), 2),
            # сколько кадр ждал очереди и сколько считался: две разные беды
            'mean_queue_wait_ms': round(state['wait_ms'] / frames, 1),
            'mean_batch_compute_ms': round(state['compute_ms'] / max(1, state['batches']), 1),
            'mean_compute_ms_per_frame': round(state['compute_ms'] / frames, 1),
            'latency_ms': {'p50': percentile(.5), 'p95': percentile(.95),
                           'p99': percentile(.99)}}


@app.post('/detect')
async def detect(file: UploadFile = File(...),
                 mask: bool = Query(True, description='вернуть PNG-маску в base64')) -> JSONResponse:
    began = time.perf_counter()
    try:
        # декодирование JPEG это 5-10 мс на кадр; в event loop оно встало бы
        # поперёк приёма остальных запросов
        raw, image = await read_image(file)
        result = await run(image)
    except HTTPException:
        state['errors'] += 1
        raise
    except Exception:
        state['errors'] += 1
        raise HTTPException(503, 'Не удалось выполнить проверку модели. Повторите запрос или проверьте память GPU.')
    state['requests'] += 1
    state['manipulated'] += int(result.manipulated)
    state['latency_ms'].append((time.perf_counter() - began) * 1000)
    del state['latency_ms'][:-10000]
    body = {'filename': file.filename, 'width': image.shape[1], 'height': image.shape[0],
            'input_sha256': hashlib.sha256(raw).hexdigest(), **payload(result, mask)}
    if mask:
        # Store only provenance and text, not uploaded images or decoded masks.
        state['predictions'][body['prediction_id']] = {
            'input_sha256': body['input_sha256'], 'model': body['model'],
            'mask_sha256': hashlib.sha256(base64.b64decode(body['mask_png_base64'])).hexdigest(),
            'width': body['width'], 'height': body['height'], 'regions': body['regions'], 'explanations': {}}
        while len(state['predictions']) > 20:
            state['predictions'].popitem(last=False)
    return JSONResponse(body)


@app.post('/explain')
async def explain(request: Request, file: UploadFile = File(...), mask: UploadFile = File(...),
                  prediction_id: str = Form(...), region_id: int = Form(...)) -> dict:
    origin = request.headers.get('origin')
    if origin and origin != str(request.base_url).rstrip('/'):
        raise HTTPException(403, 'Запрос пояснения должен исходить из этого интерфейса.')
    status = availability()
    if not status['available']:
        raise HTTPException(503, status['reason'])
    prediction = state['predictions'].get(prediction_id)
    if prediction is None:
        raise HTTPException(410, 'Результат проверки устарел. Загрузите изображение заново.')
    if not 1 <= region_id <= len(prediction['regions']):
        raise HTTPException(422, 'Выберите область модели, а не ручную отметку.')
    raw = await file.read(MAX_FILE_BYTES + 1)
    mask_raw = await mask.read(MAX_FILE_BYTES + 1)
    if max(len(raw), len(mask_raw)) > MAX_FILE_BYTES:
        raise HTTPException(413, 'Изображение или маска превышает 20 МБ.')
    if (hashlib.sha256(raw).hexdigest() != prediction['input_sha256'] or
            hashlib.sha256(mask_raw).hexdigest() != prediction['mask_sha256']):
        raise HTTPException(409, 'Файл или маска не совпадают с результатом проверки.')
    if region_id in prediction['explanations']:
        return {**prediction['explanations'][region_id], 'cached': True}
    lock = state['explanation_lock']
    if lock.locked():
        raise HTTPException(409, 'Уже запрашивается другое пояснение. Дождитесь ответа.')
    async with lock:
        image = await asyncio.to_thread(decode, raw, file.filename or 'file')
        binary_mask = cv2.imdecode(np.frombuffer(mask_raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        if binary_mask is None or binary_mask.shape != image.shape[:2]:
            raise HTTPException(422, 'Маска не соответствует размеру изображения.')
        result = await asyncio.to_thread(describe_region, image, binary_mask, prediction['regions'][region_id-1])
        result.update({'prediction_id': prediction_id, 'input_sha256': prediction['input_sha256'],
                       'mask_sha256': prediction['mask_sha256'], 'detector_model': prediction['model'],
                       'region_id': region_id, 'cached': False})
        prediction['explanations'][region_id] = result
        return result


@app.post('/detect/batch')
async def detect_batch(files: list[UploadFile] = File(...),
                       mask: bool = Query(False)) -> JSONResponse:
    if not 1 <= len(files) <= 20:
        raise HTTPException(413, 'Выберите от 1 до 20 изображений.')
    items = []
    for file in files:
        response = await detect(file, mask)
        items.append(json.loads(response.body))
    return JSONResponse({'items': items})


@app.get('/', response_class=HTMLResponse)
def index() -> str:
    return (WEB_DIR / 'index.html').read_text(encoding='utf-8')


@app.get('/legacy', response_class=HTMLResponse)
def legacy_index() -> str:
    return (WEB_DIR / 'legacy.html').read_text(encoding='utf-8')


app.mount('/', StaticFiles(directory=WEB_DIR), name='interface')

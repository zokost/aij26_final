"""Check a running inference service without Docker SDK or loading local weights.

python tests/test_smoke.py --url http://127.0.0.1:8080 \
    --image photo.jpg --expected-mask mask.png --output work/docker_smoke
python tests/test_smoke.py --self-test
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import mimetypes
from pathlib import Path
import re
import sys
import time
import unittest
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4

from PIL import Image, ImageOps


CHECKPOINT_SHA256 = '40841b9c867d5c6f1ed8fcfacefb14787a377d58f770da86e67067f1d57a4cf1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_health(health):
    require(health['status'] == 'ok' and health['mode'] == 'inference', 'Service is not ready for real inference')
    require(health['checkpoint_sha256'] == CHECKPOINT_SHA256, 'Unexpected checkpoint SHA256')
    require(health['threshold'] == .675 and health['max_batch'] == 1 and health['fast'] is False,
            'Expected base threshold=.675, batch=1, fast=false')
    model = health['model']
    expected = {'sha256': CHECKPOINT_SHA256, 'threshold': .675, 'square': True,
                'encoder': 'mit_b2', 'decoder_channels': 320,
                'img_size': 640, 'active_canvases': [[640, 640]], 'precision': 'fp32',
                'score_kind': 'uncalibrated_top_logit', 'score_calibrated': False,
                'matmul_tf32': False, 'cudnn_tf32': True,
                'threshold_mode': 'lower_only_calibrated', 'base_threshold': .675,
                'candidate_thresholds': [.5, .6, .65, .675]}
    for key, value in expected.items():
        require(model.get(key) == value, f'Unexpected model.{key}: {model.get(key)!r}')
    return model


def validate_result(result, raw_image, size, model, expected_mask=None):
    require(result['input_sha256'] == hashlib.sha256(raw_image).hexdigest(), 'Input image SHA256 differs')
    require((result['width'], result['height']) == size, 'Response geometry differs from displayed image')
    if model.get('threshold_mode') == 'lower_only_calibrated':
        require(result['model'].get('threshold') in model['candidate_thresholds'], 'Unexpected calibrated threshold')
        require({**result['model'], 'threshold': model['threshold']} == model, 'Detection model differs from /health')
    else:
        require(result['model'] == model, 'Detection model differs from /health')
    require(result['score_kind'] == 'uncalibrated_top_logit' and result['score_calibrated'] is False,
            'Expected raw logit score, not a calibrated probability')
    require(isinstance(result['score'], (int, float)) and math.isfinite(result['score']), 'Invalid logit score')
    mask_raw = base64.b64decode(result['mask_png_base64'], validate=True)
    with Image.open(io.BytesIO(mask_raw)) as mask:
        require(mask.format == 'PNG' and mask.mode == 'L', 'Mask must be a grayscale PNG')
        require(mask.size == size, 'Mask geometry differs from image')
        histogram = mask.histogram()
        require(sum(histogram[1:255]) == 0, 'Mask contains values other than 0 and 255')
        require(abs(result['area_frac'] - histogram[255] / (size[0] * size[1])) < 1e-12,
                'Mask pixel area differs from response area_frac')
        if expected_mask:
            with Image.open(expected_mask) as expected:
                require(expected.size == size, 'Expected mask geometry differs from image')
                expected = expected.convert('L')
                require(sum(expected.histogram()[1:255]) == 0, 'Expected mask is not binary 0/255')
                require(mask.tobytes() == expected.tobytes(), 'Mask pixels differ from expected mask')
    return mask_raw, histogram[255]


def run(args):
    base = args.url.rstrip('/') + '/'
    require(urlsplit(base).scheme in ('http', 'https') and urlsplit(base).netloc, 'Use an HTTP(S) service URL')
    raw_image = args.image.read_bytes()
    with Image.open(io.BytesIO(raw_image)) as image:
        size = ImageOps.exif_transpose(image).size

    def fetch(path, data=None, content_type=None):
        url = urljoin(base, path)
        require(urlsplit(url).netloc == urlsplit(base).netloc, 'Unexpected external asset URL')
        headers = {'Content-Type': content_type} if content_type else {}
        with urlopen(Request(url, data=data, headers=headers), timeout=args.timeout) as response:
            require(response.status == 200, f'{path}: HTTP {response.status}')
            return response.read(), response.headers.get_content_type()

    health = json.loads(fetch('health')[0])
    (args.output / 'health.json').write_text(json.dumps(health, ensure_ascii=False, indent=2), encoding='utf-8')
    model = validate_health(health)
    html, content_type = fetch('')
    require(content_type == 'text/html', 'Home page is not HTML')
    assets = re.findall(r'<(?:script|link)\b[^>]*(?:src|href)="([^"]+)"', html.decode('utf-8'))
    pending = [path for path in assets if not path.startswith('data:')]
    require(any(urlsplit(path).path.endswith('app.mjs') for path in pending), 'Home page has no app.mjs')
    checked_assets = []
    while pending:
        path = pending.pop(0)
        if path in checked_assets:
            continue
        body, content_type = fetch(path)
        require(bool(body), f'{path}: empty static asset')
        checked_assets.append(path)
        if urlsplit(path).path.endswith('.mjs'):
            require('javascript' in content_type, f'{path}: wrong JavaScript MIME type')
            source = body.decode('utf-8')
            if urlsplit(path).path.endswith('app.mjs'):
                require('/detect?mask=true' in source and '/health' in source, 'UI lacks inference API calls')
            pending.extend(urljoin(path, imported) for imported in re.findall(r'\bfrom\s+[\'\"]([^\'\"]+)[\'\"]', source))

    boundary = 'smoke-' + uuid4().hex
    mime = mimetypes.guess_type(args.image.name)[0] or 'application/octet-stream'
    # A fixed multipart filename avoids quoting arbitrary local filenames in HTTP headers.
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="smoke-image"\r\n'
            f'Content-Type: {mime}\r\n\r\n').encode() + raw_image + f'\r\n--{boundary}--\r\n'.encode()
    began = time.perf_counter()
    result = json.loads(fetch('detect?mask=true', body, f'multipart/form-data; boundary={boundary}')[0])
    elapsed = time.perf_counter() - began
    saved_result = {key: value for key, value in result.items() if key != 'mask_png_base64'}
    saved_result['mask_file'] = 'mask.png'
    (args.output / 'result.json').write_text(json.dumps(saved_result, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output / 'mask.png').write_bytes(base64.b64decode(result['mask_png_base64'], validate=True))
    mask_raw, positive_pixels = validate_result(result, raw_image, size, model, args.expected_mask)
    return {'passed': True, 'url': base, 'image': str(args.image.resolve()),
            'input_sha256': result['input_sha256'], 'checkpoint_sha256': model['sha256'],
            'device': model['device'], 'size': list(size), 'positive_pixels': positive_pixels,
            'score': result['score'], 'score_kind': result['score_kind'],
            'inference_ms': result['inference_ms'], 'request_seconds': round(elapsed, 3),
            'mask_sha256': hashlib.sha256(mask_raw).hexdigest(),
            'expected_mask': str(args.expected_mask.resolve()) if args.expected_mask else None,
            'expected_mask_pixels_equal': True if args.expected_mask else None,
            'static_assets': checked_assets}


class SmokeCheckerTest(unittest.TestCase):
    def test_health_guards(self):
        model = {'sha256': CHECKPOINT_SHA256, 'threshold': .675, 'square': True,
                 'encoder': 'mit_b2', 'decoder_channels': 320,
                 'img_size': 640, 'active_canvases': [[640, 640]], 'precision': 'fp32',
                 'score_kind': 'uncalibrated_top_logit', 'score_calibrated': False,
                 'matmul_tf32': False, 'cudnn_tf32': True,
                 'threshold_mode': 'lower_only_calibrated', 'base_threshold': .675,
                 'candidate_thresholds': [.5, .6, .65, .675]}
        health = {'status': 'ok', 'mode': 'inference', 'checkpoint_sha256': CHECKPOINT_SHA256,
                  'threshold': .675, 'max_batch': 1, 'fast': False, 'model': model}
        self.assertEqual(validate_health(health), model)
        with self.assertRaises(ValueError):
            validate_health({**health, 'mode': 'demo'})
        with self.assertRaises(ValueError):
            validate_health({**health, 'model': {**model, 'square': False}})

    def test_mask_and_provenance_guards(self):
        buffer = io.BytesIO()
        Image.new('L', (2, 3), 255).save(buffer, format='PNG')
        result = {'input_sha256': hashlib.sha256(b'image').hexdigest(), 'width': 2, 'height': 3,
                  'model': {}, 'score': 12.5, 'score_kind': 'uncalibrated_top_logit',
                  'score_calibrated': False, 'area_frac': 1.,
                  'mask_png_base64': base64.b64encode(buffer.getvalue()).decode()}
        self.assertEqual(validate_result(result, b'image', (2, 3), {})[1], 6)
        self.assertEqual(validate_result(result, b'image', (2, 3), {}, io.BytesIO(buffer.getvalue()))[1], 6)
        expected = io.BytesIO()
        Image.new('L', (2, 3), 0).save(expected, format='PNG')
        expected.seek(0)
        with self.assertRaises(ValueError):
            validate_result(result, b'image', (2, 3), {}, expected)
        with self.assertRaises(ValueError):
            validate_result(result, b'wrong image', (2, 3), {})
        with self.assertRaises(ValueError):
            validate_result({**result, 'score_calibrated': True}, b'image', (2, 3), {})
        buffer = io.BytesIO()
        Image.new('L', (2, 3), 127).save(buffer, format='PNG')
        with self.assertRaises(ValueError):
            validate_result({**result, 'mask_png_base64': base64.b64encode(buffer.getvalue()).decode()}, b'image', (2, 3), {})


def main():
    if sys.argv[1:] == ['--self-test']:
        return 0 if unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SmokeCheckerTest)).wasSuccessful() else 1
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8080')
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--expected-mask', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=240., help='HTTP timeout in seconds (default: 240)')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        require(args.timeout > 0, 'Timeout must be positive')
        summary = run(args)
    except Exception as error:
        summary = {'passed': False, 'error': f'{type(error).__name__}: {error}', 'url': args.url}
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

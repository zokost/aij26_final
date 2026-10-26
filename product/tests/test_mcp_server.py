"""MCP contract and adapter checks; no GPU, OpenRouter or file access outside fixtures."""
import base64
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

try:
    from mcp import Client
except ImportError:
    Client = None
from PIL import Image, ImageOps

if Client is not None:
    from app import mcp_server as module


@unittest.skipIf(Client is None, 'Optional MCP SDK2 is not installed; install requirements-mcp.txt in a separate environment.')
class AdapterTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'images'
        self.root.mkdir()
        self.photo = self.root / 'photo.png'
        Image.new('RGB', (40, 20), (20, 80, 120)).save(self.photo)
        self.adapter = module.DetectiveAdapter(self.root)
        self.calls = []
        self.api = patch.object(self.adapter, 'request', side_effect=self.fake_api)
        self.api.start()
        self.addCleanup(self.api.stop)

    def fake_api(self, path, fields=None, files=None):
        self.calls.append((path, fields, files))
        if path == '/health':
            return {'status': 'ok', 'device': 'cuda'}
        if path == '/explain':
            return {'provider': 'Together', 'cached': False, 'usage': {'cost': .001}}
        self.assertEqual(path, '/detect')
        raw = files['file'][1]
        with Image.open(io.BytesIO(raw)) as image:
            size = ImageOps.exif_transpose(image).size
        mask = Image.new('L', size)
        mask.putpixel((2, 3), 255)
        png = io.BytesIO()
        mask.save(png, format='PNG')
        return {'prediction_id': str(uuid4()), 'width': size[0], 'height': size[1],
            'input_sha256': hashlib.sha256(raw).hexdigest(), 'regions': [{'x': 2, 'y': 3, 'width': 1, 'height': 1, 'area_px': 1}],
            'area_frac': 1/(size[0]*size[1]), 'mask_png_base64': base64.b64encode(png.getvalue()).decode()}

    def test_views_have_real_image_blocks_aligned_mask_and_artificial_overlay(self):
        analysis = self.adapter.analyze_image('photo.png')
        self.assertNotIn('mask_png_base64', analysis)
        result = self.adapter.get_analysis_views(analysis['analysis_id'])
        blocks = [c for c in result.content if c.type == 'image']
        self.assertEqual(len(blocks), 3)
        pictures = [Image.open(io.BytesIO(base64.b64decode(c.data))) for c in blocks]
        self.assertEqual([p.size for p in pictures], [(40, 20)]*3)
        self.assertEqual(pictures[1].getpixel((2, 3)), 255)
        self.assertEqual(pictures[0].getpixel((0, 0)), pictures[2].getpixel((0, 0)))
        self.assertNotEqual(pictures[0].getpixel((2, 3)), pictures[2].getpixel((2, 3)))
        self.assertFalse(pictures[0].getexif())
        self.assertEqual(result.structured_content['views'], ['image', 'mask', 'overlay'])
        self.assertEqual([c[0] for c in self.calls], ['/detect'])

    def test_orientation_uses_the_same_display_coordinates_as_api(self):
        rotated = self.root / 'rotated.jpg'
        image = Image.new('RGB', (40, 20))
        exif = image.getexif()
        exif[274] = 6
        exif[315] = 'private-artist'
        image.save(rotated, exif=exif)
        analysis = self.adapter.analyze_image(str(rotated))
        result = self.adapter.get_analysis_views(analysis['analysis_id'])
        self.assertEqual((result.structured_content['width'], result.structured_content['height']), (20, 40))
        for block in (c for c in result.content if c.type == 'image'):
            raw = base64.b64decode(block.data)
            self.assertNotIn(b'private-artist', raw)
            self.assertFalse(Image.open(io.BytesIO(raw)).getexif())
        self.assertEqual(self.adapter.stored(analysis['analysis_id'])['photo'], rotated.read_bytes())

    def test_paths_outside_the_allowed_folder_never_reach_api(self):
        outside = self.root.parent / 'outside.png'
        Image.new('RGB', (4, 4)).save(outside)
        for path in ('../outside.png', str(outside)):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'image_root'):
                self.adapter.analyze_image(path)
        self.assertEqual(self.calls, [])

    def test_symlink_cannot_escape_image_root(self):
        outside = self.root.parent / 'outside.png'
        Image.new('RGB', (4, 4)).save(outside)
        link = self.root / 'link.png'
        try:
            link.symlink_to(outside)
        except OSError:
            self.skipTest('OS does not grant symlink creation')
        with self.assertRaisesRegex(ValueError, 'image_root'):
            self.adapter.analyze_image('link.png')
        self.assertEqual(self.calls, [])

    def test_nonimages_and_input_limits_are_rejected_without_api(self):
        (self.root / 'bad.png').write_bytes(b'not an image')
        with self.assertRaises(ValueError):
            self.adapter.analyze_image('bad.png')
        with patch.object(module, 'MAX_FILE_BYTES', 10), self.assertRaises(ValueError):
            self.adapter.analyze_image('photo.png')
        with patch.object(module, 'MAX_PIXELS', 10), self.assertRaises(ValueError):
            self.adapter.analyze_image('photo.png')
        self.assertEqual(self.calls, [])

    def test_analysis_count_and_byte_budget_evict_old_inputs(self):
        with patch.object(module, 'MAX_ANALYSES', 1):
            first = self.adapter.analyze_image('photo.png')
            second = self.adapter.analyze_image('photo.png')
        with self.assertRaisesRegex(ValueError, 'вытеснена'):
            self.adapter.get_analysis_views(first['analysis_id'])
        entry = self.adapter.stored(second['analysis_id'])
        budget = len(entry['photo']) + len(entry['mask']) + 1
        with patch.object(module, 'MAX_CACHE_BYTES', budget):
            third = self.adapter.analyze_image('photo.png')
        self.assertEqual(list(self.adapter.analyses), [third['analysis_id']])

    def test_paid_explanation_is_separate_and_reuses_original_bound_bytes(self):
        analysis = self.adapter.analyze_image('photo.png')
        self.adapter.get_analysis_views(analysis['analysis_id'])
        self.assertEqual(len(self.calls), 1)
        for invalid in (0, 2, True):
            with self.assertRaises(ValueError):
                self.adapter.explain_region(analysis['analysis_id'], invalid)
        result = self.adapter.explain_region(analysis['analysis_id'], 1)
        self.assertEqual(result['provider'], 'Together')
        path, fields, files = self.calls[-1]
        self.assertEqual(path, '/explain')
        self.assertEqual(fields, {'prediction_id': analysis['prediction_id'], 'region_id': '1'})
        self.assertEqual(files['file'][1], self.photo.read_bytes())
        self.assertEqual(hashlib.sha256(files['mask'][1]).hexdigest(), analysis['mask_sha256'])

    def test_empty_mask_has_no_local_explanation(self):
        analysis = self.adapter.analyze_image('photo.png')
        self.adapter.stored(analysis['analysis_id'])['result']['regions'] = []
        with self.assertRaisesRegex(ValueError, 'Пустой маске'):
            self.adapter.explain_region(analysis['analysis_id'], 1)
        self.assertEqual(len(self.calls), 1)

    def test_remote_api_or_credentials_in_api_url_are_not_allowed(self):
        for url in ('https://example.org', 'http://user:pass@127.0.0.1:8765', 'http://127.0.0.1:8765/other'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                module.DetectiveAdapter(self.root, url)

    def test_tampered_source_hash_from_api_is_rejected(self):
        good_api = self.fake_api

        def wrong_hash(*args, **kwargs):
            result = good_api(*args, **kwargs)
            result['input_sha256'] = '0'*64
            return result

        self.adapter.request.side_effect = wrong_hash
        with self.assertRaisesRegex(ValueError, 'Хеш'):
            self.adapter.analyze_image('photo.png')
        self.assertEqual(len(self.adapter.analyses), 0)


@unittest.skipIf(Client is None, 'Optional MCP SDK2 is not installed; install requirements-mcp.txt in a separate environment.')
class ProtocolTest(unittest.IsolatedAsyncioTestCase):
    async def test_tool_discovery_and_calls_work_in_both_protocol_eras(self):
        fixture = AdapterTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        server = module.create_server(fixture.adapter)
        for mode in ('auto', 'legacy'):
            with self.subTest(mode=mode):
                async with Client(server, mode=mode) as client:
                    tools = (await client.list_tools()).tools
                    self.assertEqual({t.name for t in tools}, {'service_status', 'analyze_image', 'get_analysis_views', 'explain_region'})
                    status = await client.call_tool('service_status')
                    self.assertFalse(status.is_error)
                    analysis = await client.call_tool('analyze_image', {'image_path': 'photo.png'})
                    self.assertFalse(analysis.is_error)
                    analysis_id = analysis.structured_content['analysis_id']
                    result = await client.call_tool('get_analysis_views', {'analysis_id': analysis_id})
                    self.assertEqual(sum(c.type == 'image' for c in result.content), 3)
                    invalid = await client.call_tool('get_analysis_views', {'analysis_id': 'missing'})
                    self.assertTrue(invalid.is_error)
                    self.assertIn('analyze_image', invalid.content[0].text)
                    missing = await client.call_tool('analyze_image', {})
                    self.assertTrue(missing.is_error)


if __name__ == '__main__':
    unittest.main()

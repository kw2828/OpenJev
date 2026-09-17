"""Create a smaller README preview while preserving the original gameplay recording."""

import hashlib
import json
from io import BytesIO
from pathlib import Path

from PIL import Image
from PIL import __version__ as pillow_version

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs/assets/jepa-policy-181000.gif'
SOURCE_RECEIPT = SOURCE.with_suffix('.json')
OUTPUT = SOURCE.with_name('jepa-policy-181000-preview.gif')
WIDTH = 320
COLORS = 64
MAX_BYTES = 3_000_000


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def describe(data):
    with Image.open(BytesIO(data)) as image:
        durations = []
        for index in range(image.n_frames):
            image.seek(index)
            durations.append(image.info['duration'])
        return {
            'sha256': sha256(data),
            'bytes': len(data),
            'width': image.width,
            'height': image.height,
            'frames': image.n_frames,
            'frame_durations_ms': durations,
            'total_duration_ms': sum(durations),
            'loop': image.info['loop'],
        }


def prepare():
    source_bytes = SOURCE.read_bytes()
    receipt_bytes = SOURCE_RECEIPT.read_bytes()
    source = describe(source_bytes)
    if json.loads(receipt_bytes)['gif_sha256'] != source['sha256']:
        raise ValueError('Original recording no longer matches its preserved receipt')
    if source['loop'] != 0:
        raise ValueError('Expected an infinitely looping original recording')

    dimensions = (WIDTH, round(source['height'] * WIDTH / source['width']))
    resized = []
    with Image.open(BytesIO(source_bytes)) as image:
        for index in range(image.n_frames):
            image.seek(index)
            resized.append(image.convert('RGB').resize(dimensions, Image.Resampling.LANCZOS))

    sample_size = (WIDTH // 2, round(dimensions[1] / 2))
    samples = Image.new('RGB', (sample_size[0], sample_size[1] * len(resized)))
    for index, frame in enumerate(resized):
        samples.paste(frame.resize(sample_size, Image.Resampling.LANCZOS), (0, index * sample_size[1]))
    palette = samples.quantize(colors=COLORS, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in resized]

    buffer = BytesIO()
    frames[0].save(
        buffer, format='GIF', save_all=True, append_images=frames[1:],
        duration=source['frame_durations_ms'], loop=0, disposal=1, optimize=True)
    output_bytes = buffer.getvalue()
    output = describe(output_bytes)
    if output['frame_durations_ms'] != source['frame_durations_ms']:
        raise ValueError('Preview changed the recording frame count or playback timing')
    if output['loop'] != 0 or output['bytes'] >= MAX_BYTES:
        raise ValueError('Preview must loop forever and remain below 3 MB')
    with Image.open(BytesIO(output_bytes)) as image:
        for index, frame in enumerate(frames):
            image.seek(index)
            if image.convert('RGB').tobytes() != frame.convert('RGB').tobytes():
                raise ValueError(f'Preview frame {index} changed during GIF encoding')
    if SOURCE.read_bytes() != source_bytes or SOURCE_RECEIPT.read_bytes() != receipt_bytes:
        raise ValueError('Original recording or receipt changed during preview preparation')

    provenance = {
        'purpose': 'Mechanically resized README preview of the preserved recorded gameplay',
        'source': {'path': str(SOURCE.relative_to(ROOT)), **source},
        'source_receipt': {
            'path': str(SOURCE_RECEIPT.relative_to(ROOT)), 'sha256': sha256(receipt_bytes)},
        'output': {'path': str(OUTPUT.relative_to(ROOT)), **output},
        'transform': {
            'script': str(Path(__file__).relative_to(ROOT)),
            'pillow_version': pillow_version,
            'resize': 'Lanczos, width 320 pixels, height rounded to preserve aspect ratio',
            'palette': '64 shared colors, median cut from half-size samples of every frame, no dithering',
            'gif_encoding': 'Optimized GIF with retain-frame disposal; decoded frames verified',
            'frame_selection': 'Every original frame in its original order',
            'timing': 'Original frame durations and infinite loop preserved',
            'generated_content': False,
        },
    }
    OUTPUT.write_bytes(output_bytes)
    OUTPUT.with_suffix('.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({
        'output': str(OUTPUT.relative_to(ROOT)),
        'bytes': output['bytes'], 'original_bytes': source['bytes'],
        'dimensions': dimensions, 'frames': output['frames'],
        'total_duration_ms': output['total_duration_ms'], 'sha256': output['sha256'],
    }, indent=2))


if __name__ == '__main__':
    prepare()

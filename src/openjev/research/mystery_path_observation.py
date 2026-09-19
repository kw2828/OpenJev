"""RGB-only parser for the pinned GridMysteryPath public observation.

Contract: pygame 2.4.0, scale .25, hidden origin/goal/path, visible fall feedback,
84x84 uint8 RGB in pygame's (x, y, channel) order, not render()'s image order.
Templates describe only the public sprite, not a path or environment state.
Unknown/corrupt frames are rejected instead of guessing. In particular this
cannot infer a fall when visual feedback is disabled.

The raster masks below are checked against the actual pinned upstream drawing
code in tests. At this scale the hand outline width rounds to zero, which makes
pygame fill each hand with its outline color (50, 50, 50).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Observation:
    """Public tile coordinates; heading is upstream rotation // 90 (N,W,S,E)."""

    x: int
    y: int
    heading: int
    off_path: bool


def _templates() -> dict[bytes, tuple[int, bool]]:
    # Body circle radius 6: row widths from pygame.draw.circle, center (6,6).
    body = np.zeros((12, 12, 3), dtype=np.uint8)
    for y, width in enumerate((4, 6, 8, 10, 12, 12, 12, 12, 10, 8, 6, 4)):
        left = (12 - width) // 2
        body[left:left + width, y] = (250, 204, 153)
    hands = (
        ((2, 2), (10, 2)),  # north
        ((2, 2), (2, 10)),  # west
        ((2, 10), (10, 10)),  # south
        ((10, 2), (10, 10)),  # east
    )
    # Upstream's 10x10 red X, centered in the tile, two width-3 diagonal lines.
    marker = np.array([[c == 'X' for c in row] for row in (
        'XX......XX', 'XXX....XXX', '.XXX..XXX.', '..XXXXXX..', '...XXXX...',
        '...XXXX...', '..XXXXXX..', '.XXX..XXX.', 'XXX....XXX', 'XX......XX',
    )], dtype=bool).T
    templates = {}
    for heading, centers in enumerate(hands):
        tile = body.copy()
        for x, y in centers:
            for dy, width in enumerate((2, 4, 4, 2)):
                tile[x - width // 2:x + width // 2, y - 2 + dy] = (50, 50, 50)
        templates[tile.tobytes()] = (heading, False)
        tile[1:11, 1:11][marker] = (255, 0, 0)
        templates[tile.tobytes()] = (heading, True)
    return templates


_TEMPLATES = _templates()


def parse_observation(rgb: np.ndarray) -> Observation:
    """Decode one public frame without retaining history or accessing an env.

    All nonblack pixels must belong to exactly one valid 12x12 sprite tile.
    This also rejects extra markers, multiple sprites and partial red crosses.
    Input storage is never mutated; noncontiguous/read-only arrays are accepted.
    """
    if not isinstance(rgb, np.ndarray):
        raise TypeError('rgb must be a NumPy array')
    if rgb.dtype != np.uint8 or rgb.shape != (84, 84, 3):
        raise ValueError('rgb must have dtype uint8 and shape (84,84,3) in x,y,channel order')
    xs, ys = np.nonzero(np.any(rgb != 0, axis=2))
    if not len(xs):
        raise ValueError('frame contains no public sprite')
    x, y = int(xs[0] // 12), int(ys[0] // 12)
    if np.any(xs // 12 != x) or np.any(ys // 12 != y):
        raise ValueError('frame has foreground outside one sprite tile')
    tile = rgb[x * 12:(x + 1) * 12, y * 12:(y + 1) * 12]
    identity = _TEMPLATES.get(tile.tobytes())
    if identity is None:
        raise ValueError('frame does not match a pinned public sprite/feedback template')
    return Observation(x=x, y=y, heading=identity[0], off_path=identity[1])

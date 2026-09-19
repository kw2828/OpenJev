"""Render-only checks against pinned upstream code; no env/reset/path/RNG calls."""

import ast
import hashlib
import importlib.util
import itertools
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pygame
import pytest

from openjev.research.mystery_path_observation import Observation, parse_observation

UPSTREAM = Path(__file__).resolve().parents[1] / 'evidence/mystery-path-qualification-v1/upstream/memory_gym'
PINS = {
    'character_controller.py': '534709f229ba1c0ca4c7420083e757a3e3f25947c1e99260040a2ed57f498bcc',
    'mystery_path_grid.py': 'c5975919407bec099b870843872f993990158e528c56eca77cdcb9789ad304a1',
}


@pytest.fixture(scope='module')
def render():
    assert pygame.version.ver == '2.4.0'
    for name, expected in PINS.items():
        assert hashlib.sha256((UPSTREAM / name).read_bytes()).hexdigest() == expected
    spec = importlib.util.spec_from_file_location('pinned_public_sprite', UPSTREAM / 'character_controller.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Execute only the original red-X construction statements, never the env's
    # constructor/reset. This avoids a second independently reimplemented raster.
    tree = ast.parse((UPSTREAM / 'mystery_path_grid.py').read_text())
    scale = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'SCALE' for t in n.targets))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GridMysteryPathEnv')
    reset = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'reset')
    start = next(i for i, n in enumerate(reset.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'dim' for t in n.targets))
    stop = next(i + 1 for i, n in enumerate(reset.body[start:], start) if isinstance(n, ast.Expr)
                and ast.unparse(n) == 'self.fall_off_surface.set_alpha(0)')
    marker = SimpleNamespace()
    namespace = {'self': marker, 'pygame': pygame, 'SCALE': scale}
    exec(compile(ast.Module(body=reset.body[start:stop], type_ignores=[]),  # noqa: S102 - only pinned draw code.
                 str(UPSTREAM / 'mystery_path_grid.py'), 'exec'), namespace)
    grid = [[SimpleNamespace(x=x * 12 + 6, y=y * 12 + 6) for y in range(7)] for x in range(7)]

    def frame(x, y, heading, off_path):
        # The actual GridCharacterController places the sprite on this public grid.
        controller = module.GridCharacterController(scale, (x, y), grid, heading * 90)
        sprite, rect = controller.get_rotated_sprite(heading * 90)
        screen = pygame.Surface((84, 84))
        screen.fill(0)
        screen.blit(sprite, rect)
        marker.fall_off_rect.center = rect.center
        marker.fall_off_surface.set_alpha(255 if off_path else 0)
        screen.blit(marker.fall_off_surface, marker.fall_off_rect)
        return pygame.surfarray.array3d(screen), rect

    return frame


@pytest.mark.parametrize(('x', 'y', 'heading', 'off_path'),
                         itertools.product(range(7), range(7), range(4), (False, True)))
def test_every_upstream_tile_heading_and_failure_frame(render, x, y, heading, off_path):
    frame, rect = render(x, y, heading, off_path)
    original = frame.copy()
    assert parse_observation(frame) == Observation(x, y, heading, off_path)
    assert np.array_equal(frame, original)
    if x in (0, 6) or y in (0, 6):
        # Upstream clips the sprite's transparent padding at screen boundaries.
        assert not pygame.Rect(0, 0, 84, 84).contains(rect)
    if off_path:
        plain, _ = render(x, y, heading, False)
        red = np.all(frame == (255, 0, 0), axis=2)
        assert np.any(red & np.any(plain != 0, axis=2))


def test_off_path_is_visual_feedback_and_heading_survives_overlay(render):
    frames = [render(5, 1, h, off)[0] for h, off in itertools.product(range(4), (False, True))]
    assert len({f.tobytes() for f in frames}) == 8
    for _ in range(2):
        for h in reversed(range(4)):
            assert parse_observation(render(5, 1, h, True)[0]) == Observation(5, 1, h, True)
            assert parse_observation(render(0, 6, h, False)[0]) == Observation(0, 6, h, False)


def test_read_only_noncontiguous_input_and_public_dataclass(render):
    backing = np.zeros((168, 168, 3), np.uint8)
    frame = backing[::2, ::2]
    frame[:] = render(6, 2, 1, True)[0]
    frame.flags.writeable = False
    observation = parse_observation(frame)
    assert observation == Observation(6, 2, 1, True)
    assert type(observation.x) is int and type(observation.off_path) is bool
    with pytest.raises(FrozenInstanceError):
        observation.heading = 0


@pytest.mark.parametrize('bad', [None, [], np.zeros((84, 84, 3), np.float32),
                               np.full((84, 84, 3), np.nan), np.zeros((84, 84), np.uint8),
                               np.zeros((83, 84, 3), np.uint8), np.zeros((84, 84, 4), np.uint8)])
def test_reject_nonpublic_input_types_shapes_and_dtypes(bad):
    with pytest.raises((ValueError, TypeError)):
        parse_observation(bad)


@pytest.mark.parametrize('kind', ('blank', 'two_sprites', 'stray_red', 'partial_x',
                                 'changed_body_color', 'offset_sprite', 'goal_overlay'))
def test_corrupt_or_unsupported_rasters_fail_closed(render, kind):
    frame = render(2, 4, 3, kind == 'partial_x')[0]
    if kind == 'blank':
        frame[:] = 0
    elif kind == 'two_sprites':
        other = render(1, 1, 0, False)[0]
        frame = np.maximum(frame, other)
    elif kind == 'stray_red':
        frame[0, 0] = (255, 0, 0)
    elif kind == 'partial_x':
        frame[25, 49] = 0
    elif kind == 'changed_body_color':
        frame[30, 54] = (249, 204, 153)
    elif kind == 'offset_sprite':
        frame = np.roll(frame, 1, axis=0)
    elif kind == 'goal_overlay':
        frame[0:12, 0:12] = (0, 255, 0)
    with pytest.raises(ValueError):
        parse_observation(frame)

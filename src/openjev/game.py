import math
import tempfile
from pathlib import Path

import vizdoom as vzd

from .domain import SCENARIOS, Observation

# Labels are engine-provided visible actors, not a detector learned from pixels.
ENEMIES = frozenset(
    {
        "Demon",
        "MarineChainsawVzd",
        "Cacodemon",
        "DoomImp",
        "ZombieMan",
        "ShotgunGuy",
        "ChaingunGuy",
        "HellKnight",
        "BaronOfHell",
        "Spectre",
    }
)


class Doom:
    def __init__(self, scenario="defend_the_center", seed=42):
        if scenario not in SCENARIOS:
            raise ValueError("Unsupported scenario")
        self.scenario = scenario
        self.seed = seed
        self.tmp = tempfile.TemporaryDirectory(prefix="openjev-")
        self.game = vzd.DoomGame()
        g = self.game
        g.load_config(str(Path(vzd.scenarios_path) / f"{scenario}.cfg"))
        g.set_doom_config_path(str(Path(self.tmp.name) / "vizdoom.ini"))
        g.set_window_visible(False)
        g.set_sound_enabled(False)
        g.set_labels_buffer_enabled(True)
        g.set_screen_format(vzd.ScreenFormat.RGB24)
        g.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        g.set_render_hud(True)
        g.set_render_crosshair(True)
        g.set_seed(seed)
        # Retain the scenario's canonical action space and rewards.
        self.buttons = [b.name for b in g.get_available_buttons()]
        expected = (
            ["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"]
            if scenario == "basic"
            else ["TURN_LEFT", "TURN_RIGHT", "ATTACK"]
        )
        if self.buttons != expected:
            raise ValueError(f"Unexpected scenario controls: {self.buttons}")
        g.init()

    def observe(self, directive="hunt"):
        g = self.game
        state = g.get_state()
        if state is None:
            return None
        width = state.screen_buffer.shape[1]
        px, py = (g.get_game_variable(v) for v in (vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y))
        targets = [label for label in state.labels if label.object_name in ENEMIES]
        # Prefer the visible actor nearest the crosshair; no hidden map actors.
        target = min(targets, key=lambda a: abs(a.x + a.width / 2 - width / 2), default=None)
        return Observation(
            visible=target is not None,
            aim_error=(2 * (target.x + target.width / 2) / width - 1) if target else 0,
            half_width=target.width / width if target else 0,
            distance=math.hypot(target.object_position_x - px, target.object_position_y - py)
            if target
            else 0,
            health=g.get_game_variable(vzd.GameVariable.HEALTH),
            ammo=g.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO),
            enemies=len(targets),
            target=target.object_name if target else "none",
            scenario=self.scenario,
            directive=directive,
        )

    def frame(self):
        state = self.game.get_state()
        return state.screen_buffer.copy() if state else None

    def step(self, decision, observation, tics=2):
        return self.game.make_action(decision.buttons(observation), tics)

    def stats(self):
        g = self.game
        return {
            "kills": int(g.get_game_variable(vzd.GameVariable.KILLCOUNT)),
            "health": max(0, g.get_game_variable(vzd.GameVariable.HEALTH)),
            "ammo": g.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO),
            "reward": g.get_total_reward(),
            "game_seconds": g.get_episode_time() / 35,
            "dead": g.is_player_dead(),
            "finished": g.is_episode_finished(),
        }

    def close(self):
        self.game.close()
        self.tmp.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

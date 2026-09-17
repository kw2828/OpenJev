"""Record an actual learned MiniGrid episode; rendering never enters the policy.

Playback is 150 ms per environment action for readability, not a speed benchmark.
The full map is displayed to viewers while the model receives only the native
partial image, direction, previous action and recurrent state.
"""

import argparse
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from openjev.research.memory_env import OBS_SIZE, encode_obs, make_env
from openjev.research.predictive_memory import PredictiveMemory

MAX_STEPS = 128
FRAME_MS = 150
MAX_GIF_BYTES = 3_000_000
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_policy(checkpoint):
    weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
    hidden, observation_size = weights['encoder.0.weight'].shape
    if observation_size != OBS_SIZE:
        raise ValueError(f'Expected the native {OBS_SIZE}-feature policy input')
    model = PredictiveMemory(observation_size, hidden)
    model.load_state_dict(weights, strict=True)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError('Checkpoint contains nonfinite weights')
    return model.eval()


def expected_episode(path, checkpoint_hash, size, seed):
    """Read exactly one intact world-prediction episode from its evaluation receipt."""
    if path is None:
        return None
    record = json.loads(path.read_text())
    if record.get('checkpoint_sha256') != checkpoint_hash:
        raise ValueError('Evaluation checkpoint hash does not match the recording checkpoint')
    if record.get('arm') != 'world_prediction' or record.get('mode') != 'intact':
        raise ValueError('Expected an intact world_prediction evaluation receipt')
    matches = [row for result in record['results'] if result['size'] == size
               for row in result['episodes'] if row['seed'] == seed]
    if len(matches) != 1:
        raise ValueError('Evaluation receipt must contain exactly one matching size and episode seed')
    return matches[0]


def render_frame(env, size, seed, steps, status):
    # Full-map rendering is exclusively for the viewer, never encode_obs or observe.
    native = Image.fromarray(env.get_frame(highlight=True, tile_size=24)).convert('RGB')
    width = max(512, native.width)
    frame = Image.new('RGB', (width, native.height + 78), '#101922')
    frame.paste(native, ((width-native.width)//2, 78))
    draw = ImageDraw.Draw(frame)
    font = ImageFont.load_default(size=15)
    draw.text((12, 8), 'Predictive recurrent PPO | partial observations', font=font, fill='#ffffff')
    color = '#8df0af' if status == 'success' else '#e6eaf0'
    draw.text((12, 31), f'MemoryS{size} | seed {seed} | step {steps:03d} | {status}',
              font=font, fill=color)
    draw.text((12, 54), 'Full map for viewers only. Policy receives the native 7x7 view.',
              font=ImageFont.load_default(size=13), fill='#aab8c7')
    return frame.quantize(colors=128, method=Image.Quantize.FASTOCTREE)


@torch.inference_mode()
def rollout(model, size, seed):
    env = make_env(size, seed, MAX_STEPS)
    try:
        obs, _ = env.reset(seed=seed)
        previous = torch.zeros(1, dtype=torch.long)
        state = torch.zeros(1, model.hidden)
        reset = torch.ones(1, dtype=torch.bool)
        actions, rewards = [], []
        frames = [render_frame(env, size, seed, 0, 'start')]
        terminated = truncated = False
        for step in range(1, MAX_STEPS+1):
            # The engine's full rendered map is never referenced in this input path.
            encoded = torch.from_numpy(encode_obs(obs)).unsqueeze(0)
            logits, _, state = model.observe(encoded, previous, state, reset)
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite policy output')
            action = int(logits.argmax(-1).item())
            obs, reward, terminated, truncated, _ = env.step(action)
            actions.append(action)
            rewards.append(float(reward))
            if terminated or truncated:
                status = 'success' if sum(rewards) > 0 else ('wrong goal' if terminated else 'timeout')
            else:
                status = 'running'
            frames.append(render_frame(env, size, seed, step, status))
            if terminated or truncated:
                break
            previous = torch.tensor([action], dtype=torch.long)
            reset = torch.zeros(1, dtype=torch.bool)
        if not (terminated or truncated):
            raise ValueError('Native environment did not finish within its configured limit')
        native_return = sum(rewards)
        episode = {'seed': seed, 'length': len(actions), 'return': native_return,
                   'success': native_return > 0, 'wrong_goal': bool(terminated and native_return <= 0),
                   'timeout': bool(truncated and not terminated)}
        return frames, episode, actions, rewards, bool(terminated), bool(truncated)
    finally:
        env.close()


def verify_episode(actual, expected):
    if expected is None:
        return
    for field in ['seed', 'length', 'success', 'wrong_goal', 'timeout']:
        if actual[field] != expected[field]:
            raise ValueError(f'Replay disagrees with scored episode on {field}: '
                             f'{actual[field]!r} versus {expected[field]!r}')
    if not math.isclose(actual['return'], expected['return'], rel_tol=0, abs_tol=1e-9):
        raise ValueError('Replay native reward disagrees with scored episode')


def encode_gif(frames):
    buffer = io.BytesIO()
    frames[0].save(buffer, format='GIF', save_all=True, append_images=frames[1:],
                   duration=FRAME_MS, loop=0, optimize=False, disposal=2)
    content = buffer.getvalue()
    if len(content) > MAX_GIF_BYTES:
        raise ValueError(f'GIF exceeds the 3 MB budget: {len(content)} bytes')
    # Every frame has a chronological step label, including initial and terminal frames.
    with Image.open(io.BytesIO(content)) as saved:
        if saved.n_frames != len(frames):
            raise ValueError('GIF encoder changed the chronological frame count')
        for i in range(saved.n_frames):
            saved.seek(i)
            if saved.info['duration'] != FRAME_MS:
                raise ValueError('GIF playback timing changed during encoding')
    return content


def record(checkpoint, seed, size, output, evaluation=None):
    receipt_path = output.with_suffix('.json')
    if output.suffix.lower() != '.gif':
        raise ValueError('Output must have a .gif extension')
    if output.exists() or receipt_path.exists():
        raise FileExistsError('Recording and receipt paths must both be fresh')
    checkpoint_hash = sha(checkpoint)
    expected = expected_episode(evaluation, checkpoint_hash, size, seed)
    model = load_policy(checkpoint)
    frames, episode, actions, rewards, terminated, truncated = rollout(model, size, seed)
    verify_episode(episode, expected)
    content = encode_gif(frames)
    if sha(checkpoint) != checkpoint_hash:
        raise ValueError('Checkpoint changed while recording')
    receipt = {
        'kind': 'actual_simulator_policy_recording', 'policy': 'predictive_recurrent_ppo',
        'checkpoint_sha256': checkpoint_hash,
        'dependencies': {name: importlib.metadata.version(name)
                         for name in ['minigrid', 'gymnasium', 'torch', 'numpy', 'pillow']},
        'source_sha256': {name: sha(ROOT / name) for name in
                          ['scripts/record_memory_policy.py', 'src/openjev/research/predictive_memory.py',
                           'src/openjev/research/memory_env.py']},
        'size': size, 'seed': seed, 'max_steps': MAX_STEPS, 'view_size': 7,
        'action_selection': 'greedy_argmax', 'actions': actions, 'rewards': rewards,
        'steps': len(actions), 'native_reward': episode['return'], 'episode': episode,
        'terminated': terminated, 'truncated': truncated,
        'model_input': 'Native partial image, direction, previous action and recurrent state',
        'rendering': 'Native full map for human viewers only; excluded from model input',
        'evaluation_sha256': sha(evaluation) if evaluation is not None else None,
        'matches_scored_episode': expected is not None,
        'gif_sha256': hashlib.sha256(content).hexdigest(), 'gif_bytes': len(content),
        'frames': len(frames), 'display_ms_per_frame': FRAME_MS,
        'display_seconds': len(frames)*FRAME_MS/1000,
        'display_timing': 'Illustrative playback, not measured policy or simulator speed',
    }
    # Validate everything before creating either published artifact. Exclusive file
    # creation prevents an interrupted/repeated command overwriting prior evidence.
    encoded_receipt = json.dumps(receipt, indent=2, allow_nan=False) + '\n'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        stream.write(content)
    with receipt_path.open('x') as stream:
        stream.write(encoded_receipt)
    print(json.dumps({'gif': str(output), 'receipt': str(receipt_path), 'episode': episode,
                      'frames': len(frames), 'bytes': len(content),
                      'matches_scored_episode': expected is not None}, indent=2))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--size', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path,
                        help='Optional intact world_prediction evaluation JSON to verify before writing')
    args = parser.parse_args()
    torch.set_num_threads(1)
    record(args.checkpoint, args.seed, args.size, args.output, args.evaluation)


if __name__ == '__main__':
    main()

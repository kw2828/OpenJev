"""Frozen official V-JEPA 2 and architecture-matched random video encoders.

RGB clips are retrospective training reward inputs, never action-policy inputs.
The HUD is excluded. All arms share the same clip sampling and image transform.
"""
import hashlib
import time

import numpy as np
import torch
from PIL import Image

MODEL_ID = 'facebook/vjepa2-vitl-fpc64-256'
REVISION = 'b3c1679b7c34d3255ef3547f27c7b226aefab26f'


def prepare_frame(frame):
    if frame.shape != (480, 640, 3) or frame.dtype != np.uint8:
        raise ValueError('Expected ViZDoom 640x480 RGB uint8 frame')
    # Remove the bottom HUD before the checkpoint's resize/center-crop convention.
    image = Image.fromarray(frame[:400]).resize((467, 292), Image.Resampling.BILINEAR)
    return np.asarray(image.crop((105, 18, 361, 274))).copy()


def sample_clip(frames, count=16):
    if not frames:
        raise ValueError('Empty video')
    indices = np.linspace(0, len(frames)-1, count).astype(int)
    return np.stack([frames[i] for i in indices]), indices


def clip_hash(clip):
    return hashlib.sha256(clip.tobytes()).hexdigest()


def unit_vector(x):
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all() or np.linalg.norm(x) < 1e-8:
        raise ValueError('Invalid video embedding')
    return x/np.linalg.norm(x)


def pixel_embedding(clip):
    # Same temporal average as the JEPA pooling; 8x8 RGB cells, 192 dimensions.
    x = torch.as_tensor(clip.copy()).permute(0, 3, 1, 2).float()/255.
    z = torch.nn.functional.adaptive_avg_pool2d(x, (8, 8)).mean(0).flatten().numpy()
    return unit_vector(z)


class FrozenVideoEncoder:
    def __init__(self, random_seed=None, device='mps'):
        from transformers import VJEPA2Config, VJEPA2Model
        started = time.monotonic()
        self.device = device
        with torch.random.fork_rng(devices=[]):
            if random_seed is None:
                model = VJEPA2Model.from_pretrained(MODEL_ID, revision=REVISION,
                    local_files_only=True, attn_implementation='sdpa', dtype=torch.float16)
            else:
                torch.manual_seed(random_seed)
                config = VJEPA2Config.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True)
                config._attn_implementation = 'sdpa'
                model = VJEPA2Model(config).half()
        # Only the encoder is used. The predictor is neither run nor fine-tuned.
        self.model = model.encoder.eval().to(device)
        for param in self.model.parameters():
            param.requires_grad_(False)
        state_hash = hashlib.sha256()
        for name, value in self.model.state_dict().items():
            state_hash.update(name.encode())
            state_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
        self.receipt = {'model_id': MODEL_ID, 'revision': REVISION, 'random_seed': random_seed,
                        'encoder_state_sha256': state_hash.hexdigest(),
                        'parameters': sum(v.numel() for v in self.model.parameters()),
                        'device': device, 'dtype': 'float16', 'frames': 16,
                        'load_wall_seconds': time.monotonic()-started}

    @torch.no_grad()
    def __call__(self, clip):
        x = torch.as_tensor(clip.copy()).permute(0, 3, 1, 2).float()/255.
        mean = torch.tensor([.485, .456, .406])[:, None, None]
        std = torch.tensor([.229, .224, .225])[:, None, None]
        x = ((x-mean)/std)[None].to(self.device, dtype=torch.float16)
        # Global spatiotemporal average of the last encoder layer.
        features = self.model(pixel_values_videos=x).last_hidden_state.float().mean(dim=1)
        embedding = features[0].cpu().numpy()  # Synchronizes the encoder's GPU work.
        return unit_vector(embedding)

"""Visual shifts applied to observation and goal pixels inside the policy's image transform.

Shifts: gamma, hue, contrast, shift (here) and blur, snp, dark (image_corruption.py, from AdaJEPA).
snp noise modes: stream = fresh noise per frame, reseeded per episode; episode = one pattern per episode (redrawn at the per-episode reseed), used for the goal image
(the goal is corrupted once and observations get fresh noise).
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np
import torch

from . import image_corruption as ic

log = logging.getLogger("lewm_optta.corruption")


def apply_gamma(frame: np.ndarray, gamma) -> np.ndarray:
    """Gamma curve on intensities: <1 brightens midtones, >1 darkens them."""
    out = 255.0 * (frame.astype(np.float32) / 255.0) ** float(gamma)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_hue(frame: np.ndarray, degrees) -> np.ndarray:
    """Rotate hue by `degrees` (frames are RGB)."""
    import cv2
    hsv = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2HSV)
    hsv[..., 0] = ((hsv[..., 0].astype(np.int32) + int(round(float(degrees) / 2.0))) % 180).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def apply_contrast(frame: np.ndarray, factor) -> np.ndarray:
    """Contrast about mid-gray: <1 compresses, >1 stretches."""
    out = 128.0 + float(factor) * (frame.astype(np.float32) - 128.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_shift(frame: np.ndarray, pixels) -> np.ndarray:
    """Translate by `pixels` right and down, edge-replicated."""
    import cv2
    h, w = frame.shape[:2]
    m = np.float32([[1, 0, float(pixels)], [0, 1, float(pixels)]])
    return cv2.warpAffine(np.ascontiguousarray(frame), m, (w, h), borderMode=cv2.BORDER_REPLICATE)


CORRUPTIONS = {
    **ic.CORRUPTIONS,
    "gamma": (apply_gamma, 0.5), "hue": (apply_hue, 30.0), "contrast": (apply_contrast, 0.6),
    "shift": (apply_shift, 10),
}


def nphash(arr) -> str:
    """Short content hash of an array."""
    a = np.ascontiguousarray(np.asarray(arr))
    return hashlib.sha1(a.tobytes()).hexdigest()[:16]


class Corruptor:
    def __init__(self, transform, key: str, name: str, level, seed: int, mode: str, reported: set):
        if name not in CORRUPTIONS:
            raise ValueError(f"unknown corruption {name!r}; choose from {list(CORRUPTIONS)}")
        if mode not in ("stream", "episode"):
            raise ValueError(f"corruption mode must be stream|episode, got {mode!r}")
        self.transform, self.key, self.name, self.mode, self.seed = transform, key, name, mode, int(seed)
        self.func, default_level = CORRUPTIONS[name]
        self.level = default_level if level is None else float(level)
        self.stream_rng = np.random.RandomState(self.seed) if mode == "stream" else None
        self.episode_seed = self.seed % (2**32)
        self.reported = reported

    def reseed(self, seed: int) -> None:
        """Restart the stream RNG (stream mode) or redraw the episode pattern (episode mode)."""
        if self.stream_rng is not None:
            self.stream_rng = np.random.RandomState(int(seed) % (2**32))
        if self.mode == "episode":
            self.episode_seed = int(seed) % (2**32)

    def corrupt_hwc(self, hwc: np.ndarray) -> np.ndarray:
        if self.name.startswith("snp"):
            if self.stream_rng is not None:
                rng = self.stream_rng
            else:
                rng = np.random.RandomState(self.episode_seed)      # same pattern on every call within the episode
            return self.func(hwc, self.level, rng)
        return self.func(hwc, self.level)

    def __call__(self, img):
        arr = np.asarray(img)                                   # (C, H, W) uint8
        hwc = np.ascontiguousarray(np.transpose(arr, (1, 2, 0)))
        out = self.corrupt_hwc(hwc)
        if self.key not in self.reported:
            self.reported.add(self.key)
            delta = float(np.abs(out.astype(np.int16) - hwc.astype(np.int16)).mean())
            log.info("[corruption] %s: %s level=%s mode=%s mean|delta|=%.3f",
                     self.key, self.name, self.level, self.mode, delta)
        chw = np.ascontiguousarray(np.transpose(out, (2, 0, 1)))
        return self.transform(torch.from_numpy(chw))


def wrap_transform(transform, key: str, name: str, level, seed: int, mode: str, reported: set) -> Corruptor:
    return Corruptor(transform, key, name, level, seed, mode, reported)

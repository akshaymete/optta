"""Online image corruption for OOD evaluation.

Supports: blur, snp (salt-and-pepper), dark (brightness reduction), and gamma, hue,
contrast, shift (translation), the last four with the same definitions as
lewm_optta/corruption.py so both world models share one shift set.
Applied to uint8 HWC numpy arrays, compatible with observation dicts
from env.rollout().
"""

import numpy as np
import cv2


def apply_salt_pepper(frame, density, rng):
    """Salt-and-pepper noise on a single HxWxC uint8 frame."""
    h, w = frame.shape[:2]
    out = frame.copy()
    n = int(density * h * w)
    ys = rng.randint(0, h, n); xs = rng.randint(0, w, n)
    out[ys, xs] = 255
    ys = rng.randint(0, h, n); xs = rng.randint(0, w, n)
    out[ys, xs] = 0
    return out


def apply_blur(frame, sigma):
    """Gaussian blur with given sigma."""
    k = max(3, int(2 * round(3 * sigma) + 1))
    return cv2.GaussianBlur(frame, (k, k), sigmaX=sigma, sigmaY=sigma)


def apply_dark(frame, factor):
    """Darken by multiplying pixel values by factor < 1."""
    out = frame.astype(np.float32) * factor
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_gamma(frame, gamma):
    """Gamma curve on intensities: <1 brightens midtones, >1 darkens them."""
    out = 255.0 * (frame.astype(np.float32) / 255.0) ** float(gamma)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_hue(frame, degrees):
    """Rotate hue by `degrees` (frames are RGB)."""
    hsv = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2HSV)
    hsv[..., 0] = ((hsv[..., 0].astype(np.int32) + int(round(float(degrees) / 2.0))) % 180).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def apply_contrast(frame, factor):
    """Contrast about mid-gray: <1 compresses, >1 stretches."""
    out = 128.0 + float(factor) * (frame.astype(np.float32) - 128.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_shift(frame, pixels):
    """Translate by `pixels` right and down, edge-replicated."""
    h, w = frame.shape[:2]
    m = np.float32([[1, 0, float(pixels)], [0, 1, float(pixels)]])
    return cv2.warpAffine(np.ascontiguousarray(frame), m, (w, h), borderMode=cv2.BORDER_REPLICATE)


# Registry: name -> (function, default_level)
CORRUPTIONS = {
    "blur":  (apply_blur, 2.0),
    "snp1":  (apply_salt_pepper, 0.01),
    "snp5":  (apply_salt_pepper, 0.05),
    "dark":  (apply_dark, 0.5),
    "gamma": (apply_gamma, 0.5),
    "hue":   (apply_hue, 30.0),
    "contrast": (apply_contrast, 0.6),
    "shift": (apply_shift, 10),
}


def corrupt_frames(frames, corruption_name, level=None, seed=0, rng=None):
    """Apply corruption to a batch of frames.

    Args:
        frames: np.ndarray, shape (..., H, W, C) uint8.
            Can be (H,W,C), (T,H,W,C), (B,T,H,W,C), etc.
        corruption_name: one of "blur", "snp1", "snp5", "dark", or None/"none".
        level: override default level. If None, use default.
        seed: RNG seed for stochastic corruptions (snp).
        rng: optional np.random.RandomState to draw from instead (seed is then ignored).

    Returns:
        Corrupted frames, same shape and dtype.
    """
    if corruption_name is None or corruption_name == "none":
        return frames

    if corruption_name not in CORRUPTIONS:
        raise ValueError(
            f"Unknown corruption '{corruption_name}'. "
            f"Choose from: {list(CORRUPTIONS.keys())}"
        )

    func, default_level = CORRUPTIONS[corruption_name]
    if level is None:
        level = default_level

    # seed=None -> truly random (system entropy), fresh noise each call.
    # seed=int -> reproducible noise for that seed.
    rng = np.random.RandomState(seed) if rng is None else rng
    orig_shape = frames.shape
    flat = frames.reshape(-1, *frames.shape[-3:])  # (N, H, W, C)
    out = np.empty_like(flat)
    for i in range(flat.shape[0]):
        if corruption_name.startswith("snp"):
            out[i] = func(flat[i], level, rng)
        else:
            out[i] = func(flat[i], level)
    return out.reshape(orig_shape)


def corrupt_obs_dict(obs, corruption_name, level=None, seed=0, rng=None):
    """Apply corruption to the 'visual' key of an observation dict.

    Only modifies 'visual'; other keys (proprio, etc.) are unchanged.

    Args:
        obs: dict with "visual" key containing uint8 numpy array.
        corruption_name: corruption type or None/"none".
        level: optional level override.
        seed: RNG seed.

    Returns:
        New dict with corrupted visual, other keys shared.
    """
    if corruption_name is None or corruption_name == "none":
        return obs
    out = dict(obs)
    out["visual"] = corrupt_frames(obs["visual"], corruption_name, level, seed, rng)
    return out

"""Online image corruption for OOD evaluation.

Supports: blur, snp (salt-and-pepper), dark (brightness reduction).
Applied to uint8 HWC numpy arrays.
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


# Registry: name -> (function, default_level)
CORRUPTIONS = {
    "blur":  (apply_blur, 2.0),
    "snp":   (apply_salt_pepper, 0.05),
    "dark":  (apply_dark, 0.5),
}

"""Determinism pin and plan hashing.

CUBLAS_WORKSPACE_CONFIG must be in the environment before the CUDA context exists;
eval.py sets a default before importing torch.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np
import torch


def pin(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:  # pragma: no cover
        print(f"[determinism] strict mode unavailable ({e}); falling back to warn_only", flush=True)
        torch.use_deterministic_algorithms(True, warn_only=True)

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") is None:
        print("[determinism] CUBLAS_WORKSPACE_CONFIG is unset; cuBLAS may be nondeterministic", flush=True)


def plan_hash(actions: torch.Tensor) -> str:
    """Short content hash of a plan tensor, for bit-identity checks across runs."""
    data = actions.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha1(data).hexdigest()[:16]

# -*- coding: utf-8 -*-
"""Common utility functions."""

from __future__ import annotations

import datetime as dt
import os
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalize a 2D numpy array."""
    norms = np.sqrt((x * x).sum(axis=1, keepdims=True)) + eps
    return x / norms


def now_kst() -> dt.datetime:
    """Return current time in Asia/Seoul (KST)."""
    # UTC+9, standard library only
    return dt.datetime.utcnow() + dt.timedelta(hours=9)


def timestamp_folder_kst() -> str:
    """Return a timestamp string for folder names, e.g., '240911_153000'."""
    return now_kst().strftime("%y%m%d_%H%M%S")


def safe_model_name_for_path(model_name: str) -> str:
    """Sanitize a model name to be used in a file path."""
    return model_name.replace("/", "-").replace(" ", "_")


def ensure_dir(path: str) -> None:
    """Ensure that a directory exists."""
    os.makedirs(path, exist_ok=True)

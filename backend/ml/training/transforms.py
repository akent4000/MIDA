"""Custom MONAI-compatible transforms for CXR preprocessing.

CLAHE (Contrast Limited Adaptive Histogram Equalization) is applied
deterministically as preprocessing — not as augmentation. It improves
local contrast in chest radiographs, making subtle opacities more visible.

Applied *before* resizing so the full-resolution detail is preserved.
Expected input: float32 ndarray in [0, 1], shape (1, H, W) after EnsureChannelFirst.
"""

from __future__ import annotations

import numpy as np
from monai.transforms import Transform
from skimage.exposure import equalize_adapthist


class CLAHETransform(Transform):
    """CLAHE applied per channel on a float32 image in [0, 1].

    Args:
        clip_limit: Fraction of max histogram bin that triggers clipping.
            0.01 is a gentle contrast boost; 0.03 is more aggressive.
        kernel_size: Size of the contextual region for equalization.
            None → skimage default (1/8 of image in each dimension).
    """

    def __init__(self, clip_limit: float = 0.01, kernel_size: int | None = None) -> None:
        self.clip_limit = clip_limit
        self.kernel_size = kernel_size

    def __call__(self, img: np.ndarray | object) -> np.ndarray:
        # After MONAI EnsureChannelFirst, img may be a MetaTensor (Tensor subclass).
        # Convert to plain numpy for skimage, then back to the same array type.
        import torch

        is_tensor = isinstance(img, torch.Tensor)
        arr: np.ndarray = img.numpy() if is_tensor else np.asarray(img)  # type: ignore[union-attr]
        # arr shape: (C, H, W), float32 in [0, 1]
        out = np.empty_like(arr)
        for c in range(arr.shape[0]):
            out[c] = equalize_adapthist(
                arr[c],
                kernel_size=self.kernel_size,
                clip_limit=self.clip_limit,
            ).astype(np.float32)
        if is_tensor:
            return torch.as_tensor(out)  # type: ignore[return-value]
        return out

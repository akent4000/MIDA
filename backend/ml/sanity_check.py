"""Reproducible stack sanity check.

Confirms that the ML environment has working torch+CUDA (on the dev box) and
that MONAI / pydicom / scikit-image imports resolve. Exits non-zero on any
import or runtime failure so it can be wired into CI later.

Usage:
    python backend/ml/sanity_check.py
"""

from __future__ import annotations

import sys


def main() -> int:
    import torch

    print(f"python       : {sys.version.split()[0]}")
    print(f"torch        : {torch.__version__}")
    print(f"cuda runtime : {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device       : {torch.cuda.get_device_name(0)}")
        x = torch.randn(1024, 1024, device="cuda")
        y = (x @ x.T).sum().item()
        print(f"cuda matmul  : ok (checksum={y:.2f})")

    import monai
    import pydicom
    import skimage

    print(f"monai        : {monai.__version__}")
    print(f"pydicom      : {pydicom.__version__}")
    print(f"scikit-image : {skimage.__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

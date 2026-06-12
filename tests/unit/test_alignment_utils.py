from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest

from guanwu.video.features.spatial.alignment_utils import build_depth_point_cloud


def _install_fake_pycocotools(monkeypatch: pytest.MonkeyPatch, mask: np.ndarray) -> None:
    fake_mask_module = types.SimpleNamespace(decode=lambda rle: mask)
    fake_package = types.SimpleNamespace(mask=fake_mask_module)
    monkeypatch.setitem(sys.modules, "pycocotools", fake_package)
    monkeypatch.setitem(sys.modules, "pycocotools.mask", fake_mask_module)


def test_build_depth_point_cloud_rejects_unaligned_depth_mask_sizes(monkeypatch, tmp_path) -> None:
    np.save(tmp_path / "00000.npy", np.ones((3, 5), dtype=np.float32))
    mask = np.ones((18, 26), dtype=np.uint8)
    _install_fake_pycocotools(monkeypatch, mask)

    with pytest.raises(ValueError, match="Please use aligned depth"):
        build_depth_point_cloud(
            tmp_path,
            1,
            json.dumps({"size": [18, 26], "counts": ""}),
            wildgs_K={"fx": 100.0, "fy": 100.0, "cx": 13.0, "cy": 9.0},
        )

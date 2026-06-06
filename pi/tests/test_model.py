"""Tests for fog.model.FoGNet — forward shape + the checkpoint arch round-trip.

These need torch, so they are skipped in a torch-free environment and run in
Colab / on the laptop. The round-trip tests guard a real bug: a tuned-width model
(non-default c1/c2/c3/fc) used to be unreloadable because the checkpoint saved no
architecture, so it rebuilt at the default shape and ``load_state_dict`` raised.
The fix records :attr:`FoGNet.arch` and every loader rebuilds with
``FoGNet(num_classes=..., **ckpt.get('arch', {}))``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from fog.config import LABELS, NUM_AXES, WINDOW_SIZE

pytestmark = pytest.mark.needs_torch


def test_forward_output_shape() -> None:
    import torch

    from fog.model import FoGNet

    model = FoGNet().eval()
    out = model(torch.zeros(4, NUM_AXES, WINDOW_SIZE))
    assert out.shape == (4, len(LABELS))


def test_arch_records_shape_without_num_classes() -> None:
    from fog.model import FoGNet

    model = FoGNet(num_classes=len(LABELS), c1=24, c2=48, c3=96, fc=40, dropout=0.4)
    assert "num_classes" not in model.arch
    assert model.arch == dict(
        num_axes=NUM_AXES, c1=24, c2=48, c3=96, fc=40, dropout=0.4
    )


def test_tuned_checkpoint_reloads_at_same_shape(tmp_path: Path) -> None:
    import torch

    from fog.model import FoGNet

    model = FoGNet(num_classes=len(LABELS), c1=24, c2=48, c3=96, fc=40, dropout=0.4)
    ckpt = {"labels": list(LABELS), "arch": model.arch,
            "model_state": model.state_dict()}
    path = tmp_path / "fog_model.pth"
    torch.save(ckpt, path)

    loaded = torch.load(path, map_location="cpu", weights_only=False)
    rebuilt = FoGNet(num_classes=len(loaded["labels"]), **loaded.get("arch", {}))
    rebuilt.load_state_dict(loaded["model_state"])  # raises on any shape mismatch


def test_checkpoint_without_arch_key_falls_back_to_default(tmp_path: Path) -> None:
    import torch

    from fog.model import FoGNet

    # An *old* checkpoint with no 'arch' key (default-width model) must still load
    # via the `ckpt.get('arch', {})` fallback the deployment scripts use.
    base = FoGNet(num_classes=len(LABELS))
    path = tmp_path / "old.pth"
    torch.save({"labels": list(LABELS), "model_state": base.state_dict()}, path)

    loaded = torch.load(path, map_location="cpu", weights_only=False)
    rebuilt = FoGNet(num_classes=len(loaded["labels"]), **loaded.get("arch", {}))
    rebuilt.load_state_dict(loaded["model_state"])


def test_tuned_state_into_default_shape_raises() -> None:
    from fog.model import FoGNet

    # The pre-fix failure mode: rebuilding a tuned model at the default shape.
    tuned = FoGNet(num_classes=len(LABELS), c1=24, c2=48, c3=96)
    with pytest.raises(RuntimeError):
        FoGNet(num_classes=len(LABELS)).load_state_dict(tuned.state_dict())

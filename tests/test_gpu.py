"""実 GPU を使うテスト。`uv run pytest -m gpu` で実行する。"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.gpu


def test_cuda_available() -> None:
    import torch
    assert torch.cuda.is_available(), "CUDA が使えません（uv sync --extra gpu とドライバを確認）"
    name = torch.cuda.get_device_name(0)
    print(f"GPU: {name} / CUDA {torch.version.cuda}")
    assert name


def test_simple_gpu_matmul() -> None:
    import torch
    a = torch.randn(256, 256, device="cuda")
    b = torch.randn(256, 256, device="cuda")
    c = a @ b
    expected = (a.cpu() @ b.cpu())
    assert c.device.type == "cuda"
    assert torch.allclose(c.cpu(), expected, atol=1e-2)


def test_fp16_on_gpu() -> None:
    import torch
    x = torch.ones(1024, device="cuda", dtype=torch.float16)
    assert float(x.sum().item()) == 1024.0

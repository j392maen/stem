"""onnxruntime の版の違いの注意を1行にまとめる（GPU なしで確かめる）。"""

from __future__ import annotations

import logging

import pytest

from stemapp.separation import audio_separator_backend as backend

NOISE = [
    "\x1b[33mWARNING: The installed PyTorch 2.11.0+cu128 uses CUDA 12.x, but onnxruntime-gpu is "
    "built with CUDA 13.x. Please install PyTorch for CUDA 13.x to be compatible.\x1b[0m",
    "Failed to load cublasLt64_13.dll: Could not find module 'cublasLt64_13.dll'.",
    "Failed to load cudart64_13.dll: Could not find module 'cudart64_13.dll'.",
    "Please follow https://onnxruntime.ai/docs/install/#cuda-and-cudnn to install CUDA.",
]


def test_noise_is_folded_into_one_note(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(backend, "_ort_noted", False)
    caplog.set_level(logging.INFO, logger=backend.log.name)
    for _ in range(2):  # 2回目（次のモデル）は注意を出さない
        with backend.quiet_onnxruntime_preload():
            for line in NOISE:
                print(line)
            print("ほかの出力")
    assert capsys.readouterr().out == ""  # 標準出力には何も出ない
    messages = [r.getMessage() for r in caplog.records]
    assert messages.count(backend.ORT_NOTE) == 1
    assert messages.count("ほかの出力") == 2
    assert not any("Failed to load" in m or "cublas" in m for m in messages)


def test_no_note_without_noise(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(backend, "_ort_noted", False)
    caplog.set_level(logging.INFO, logger=backend.log.name)
    with backend.quiet_onnxruntime_preload():
        pass
    assert caplog.records == []


def test_output_is_logged_even_when_failing(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger=backend.log.name)
    with pytest.raises(RuntimeError), backend.quiet_onnxruntime_preload():
        print("途中の出力")
        raise RuntimeError("失敗")
    assert "途中の出力" in [r.getMessage() for r in caplog.records]

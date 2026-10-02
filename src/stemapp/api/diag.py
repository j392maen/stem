"""端末の診断の API。

- `GET /api/diag/samples/{形式}`: 診断用のテスト音声（webm・m4a・mp3・flac・wav）
- `POST /api/diag`: 診断結果（JSON のオブジェクト）を保存する
- `GET /api/diag`: 保存した診断結果の一覧（新しい順、中身つき）
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

from stemapp import diag
from stemapp.api.common import is_https, is_local_request, is_proxied
from stemapp.audio import AudioError
from stemapp.config import Settings

router = APIRouter(prefix="/api/diag", tags=["diag"])

MSG_TOO_LARGE = f"診断結果が大きすぎます（上限 {diag.MAX_DIAG_BYTES // 1024} KB）。"
MSG_NOT_JSON = "診断結果は JSON で送ってください。"
MSG_NOT_OBJECT = "診断結果は JSON のオブジェクト（{...}）で送ってください。"


@router.get("/samples/{code}")
def sample(code: str, request: Request) -> FileResponse:
    fmt = diag.SAMPLE_FORMATS.get(code)
    if fmt is None:
        raise HTTPException(status_code=404, detail="その形式のテスト音声はありません。")
    settings: Settings = request.app.state.settings
    runner = getattr(request.app.state, "diag_ffmpeg_runner", None)
    try:
        path = diag.ensure_sample(settings, code, runner)
    except AudioError as e:
        raise HTTPException(status_code=503, detail=f"テスト音声を作れませんでした。{e}") from e
    return FileResponse(path, media_type=fmt.media_type, headers={"Cache-Control": "no-store"})


@router.get("/samples")
def samples() -> dict[str, Any]:
    return {
        "samples": [
            {"code": f.code, "label": f.label, "media_type": f.media_type,
             "url": f"/api/diag/samples/{f.code}"}
            for f in diag.SAMPLE_FORMATS.values()
        ]
    }


async def _read_limited(request: Request, limit: int) -> bytes:
    length = request.headers.get("content-length")
    if length is not None and length.isascii() and length.isdigit() and int(length) > limit:
        raise HTTPException(status_code=413, detail=MSG_TOO_LARGE)
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise HTTPException(status_code=413, detail=MSG_TOO_LARGE)
        chunks.append(chunk)
    return b"".join(chunks)


@router.post("")
async def post_result(request: Request) -> dict[str, str]:
    # JSON 以外（フォームなど）は受け付けない。他のサイトのページからの単純な POST を防ぐ
    ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if ctype != "application/json":
        raise HTTPException(status_code=415, detail=MSG_NOT_JSON)
    raw = await _read_limited(request, diag.MAX_DIAG_BYTES)
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=MSG_NOT_JSON) from e
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail=MSG_NOT_OBJECT)
    server_info = {
        "proxied": is_proxied(request),
        "local_client": is_local_request(request),
        "https": is_https(request),
        "user_agent": request.headers.get("user-agent", ""),
    }
    name = diag.save_result(request.app.state.settings, data, server_info)
    return {"name": name}


@router.get("")
def list_results(
    request: Request, limit: int = Query(20, ge=1, le=diag.MAX_LIST)
) -> dict[str, Any]:
    return {"results": diag.list_results(request.app.state.settings, limit)}

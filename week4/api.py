import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from search import FashionSearchEngine


class TextSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        config_path = os.path.join(
            CURRENT_DIR,
            "config",
            "search_config.yaml",
        )

        app.state.engine = FashionSearchEngine(
            config_path=config_path
        )
        app.state.startup_error = None

    except Exception as exc:
        app.state.engine = None
        app.state.startup_error = str(exc)
        print(f"[!] 검색 엔진 로드 실패: {exc}")

    yield

    app.state.engine = None


app = FastAPI(
    title="Fashion Search Model API",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request):
    ready = request.app.state.engine is not None

    return {
        "status": "ok" if ready else "not_ready",
        "service": "fashion-search-model-api",
        "model_ready": ready,
        "error": request.app.state.startup_error,
    }


@app.post("/search/text")
def search_text(
    payload: TextSearchRequest,
    request: Request,
):
    engine = request.app.state.engine

    if engine is None:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "검색 엔진이 준비되지 않았습니다.",
                "error": request.app.state.startup_error,
            },
        )

    results = engine.search_by_text(
        payload.query,
        top_k=payload.top_k,
    )

    return {
        "query": payload.query,
        "top_k": payload.top_k,
        "results": results,
    }

from fastapi.responses import FileResponse

def resolve_image_path(metadata_path: str) -> str:
    """Windows에서 생성된 메타데이터 경로를 현재 서버 경로로 변환합니다."""
    normalized = metadata_path.replace("\\", "/")

    windows_root = "c:/Users/User/Projects/cv-bootcamp"
    server_root = os.environ.get(
        "FASHION_PROJECT_ROOT",
        "/mnt/c/Users/User/Projects/cv-bootcamp",
    )

    if normalized.lower().startswith(windows_root.lower()):
        relative_path = normalized[len(windows_root):].lstrip("/")
        return os.path.join(server_root, *relative_path.split("/"))

    return normalized

@app.get("/items/{item_id}/image")
def get_item_image(item_id: int, request: Request):
    engine = request.app.state.engine

    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="검색 엔진이 준비되지 않았습니다.",
        )

    if item_id < 0 or item_id >= len(engine.metadata):
        raise HTTPException(
            status_code=404,
            detail="상품을 찾을 수 없습니다.",
        )

    item = engine.metadata[item_id]
    image_path = resolve_image_path(item["image_path"])

    if not os.path.isfile(image_path):
        raise HTTPException(
            status_code=404,
            detail={
                "message": "이미지 파일을 찾을 수 없습니다.",
                "resolved_path": image_path,
            },
        )

    return FileResponse(image_path)


import cv2
import numpy as np

from fastapi import File, UploadFile
@app.post("/search/image")
def search_image(
    request: Request,
    file: UploadFile = File(...),
    top_k: int = 5,
):
    engine = request.app.state.engine

    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="검색 엔진이 준비되지 않았습니다.",
        )

    if top_k < 1 or top_k > 10:
        raise HTTPException(
            status_code=422,
            detail="top_k는 1 이상 10 이하여야 합니다.",
        )

    if file.content_type not in {
        "image/jpeg",
        "image/png",
    }:
        raise HTTPException(
            status_code=415,
            detail="JPEG 또는 PNG 이미지만 지원합니다.",
        )

    file_bytes = file.file.read()

    image_array = np.frombuffer(
        file_bytes,
        dtype=np.uint8,
    )

    image_bgr = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if image_bgr is None:
        raise HTTPException(
            status_code=400,
            detail="이미지 파일을 해석할 수 없습니다.",
        )

    try:
        results, detected_class, _ = engine.search_by_image(
            image_bgr,
            top_k=top_k,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "이미지 검색 중 오류가 발생했습니다.",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        ) from exc

    return {
        "filename": file.filename,
        "detected_class": detected_class,
        "top_k": top_k,
        "results": results,
    }

import asyncio
import io
import os
import json
import base64
import sys
import shutil
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse


current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env")
load_dotenv(env_path)

# OCR 서버 포트
ocr_port = os.getenv("OCR_PORT", "8090")
os.environ.setdefault("PORT", ocr_port)
os.environ.setdefault("UVICORN_PORT", ocr_port)

app = FastAPI()


BASE_DIR = Path(current_dir)
OCR_DIR = BASE_DIR / "ocr"
OCR_UPLOAD_DIR = OCR_DIR / "uploads"
OCR_IMG_DIR = OCR_DIR / "img"

OCR_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OCR_IMG_DIR.mkdir(parents=True, exist_ok=True)

if str(OCR_DIR) not in sys.path:
    sys.path.insert(0, str(OCR_DIR))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Docker health check endpoint"""
    return {"status": "healthy"}


def _save_upload_file(upload: UploadFile, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)


def _encode_image_base64_resized(path: Path, max_bytes: int = 1_400_000, max_long_side: int = 1200, quality: int = 85) -> str | None:
    """이미지를 읽어, 크기가 max_bytes 초과면 해상도/품질을 낮춰 JPEG로 압축한 뒤 data URL 반환."""
    if not path.exists() or not path.is_file():
        return None
    raw = path.read_bytes()
    if len(raw) <= max_bytes:
        ext = path.suffix.lower().lstrip(".") or "jpg"
        mime = "jpeg" if ext in {"jpg", "jpeg"} else ext
        return f"data:image/{mime};base64,{base64.b64encode(raw).decode('utf-8')}"
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        w, h = img.size
        if max(w, h) > max_long_side:
            img.thumbnail((max_long_side, max_long_side), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        data = buf.getvalue()
        while len(data) > max_bytes and (max_long_side > 320 or quality > 40):
            if quality > 40:
                quality -= 10
            else:
                max_long_side = int(max_long_side * 0.75)
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                if max(img.size) > max_long_side:
                    img.thumbnail((max_long_side, max_long_side), Image.Resampling.LANCZOS)
                quality = 75
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            data = buf.getvalue()
        encoded = base64.b64encode(data).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception as e:
        print(f"[diary-ocr] 이미지 리사이즈 실패 (원본 사용 시도): {e}")
        if len(raw) <= 2 * max_bytes:
            return f"data:image/jpeg;base64,{base64.b64encode(raw).decode('utf-8')}"
        return None


@app.post("/diary-ocr")
async def diary_ocr(file: UploadFile = File(...)):
    """그림일기 OCR 처리 엔드포인트"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="파일을 업로드해 주세요.")

    ext = Path(file.filename).suffix or ".jpg"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"diary_{timestamp}"
    input_path = OCR_UPLOAD_DIR / f"{base_name}{ext}"
    _save_upload_file(file, input_path)

    # diary_ocr_only.run(): 이미지 저장 + 텍스트 추출 + Gemini 후처리 (bbox 제거된 원본)
    try:
        import diary_ocr_only
        response_item = diary_ocr_only.run(str(input_path.resolve()))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"일기 OCR 실패: {e}")

    # null → 빈 문자열
    def _str(v):
        return v if isinstance(v, str) else (v or "")

    response_item = {k: _str(v) for k, v in response_item.items()}

    # 크롭된 그림(그림_저장경로)을 data URL로 넣어 프론트 카드 사진란에서 사용. 크면 해상도 낮춰서라도 포함.
    MAX_IMAGE_BYTES_FOR_RESPONSE = 1_400_000  # 약 1.4MB 초과 시 리사이즈 후 포함
    try:
        cropped_path = (response_item.get("그림_저장경로") or "").strip()
        if cropped_path:
            p = Path(cropped_path)
            if p.exists() and p.is_file():
                url = _encode_image_base64_resized(p, max_bytes=MAX_IMAGE_BYTES_FOR_RESPONSE)
                response_item["image_data_url"] = url
                if url and len(url) > 500:
                    print(f"[diary-ocr] 크롭 이미지 포함 (data URL 길이={len(url)})")
            else:
                response_item["image_data_url"] = None
        else:
            response_item["image_data_url"] = None
    except Exception as e:
        print(f"[diary-ocr] 크롭 이미지 data URL 변환 실패 (무시): {e}")
        response_item["image_data_url"] = None

    payload_for_json = {k: v for k, v in response_item.items() if k != "image_data_url"}
    json_path = OCR_DIR / f"{base_name}_diary_result.json"
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload_for_json, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[diary-ocr] JSON 저장 실패 (무시): {e}")

    print("\n" + "=" * 60)
    print("[그림일기 OCR] 분석 결과 (diary_ocr_only)")
    print("=" * 60)
    print(json.dumps(payload_for_json, ensure_ascii=False, indent=2))
    print("=" * 60 + "\n")

    try:
        return [{**response_item, "교정된_내용": response_item.get("내용", "") or ""}]
    except Exception as e:
        print(f"[diary-ocr] 응답 반환 직전 오류 (이미지 제외 후 재시도): {e}")
        response_item["image_data_url"] = None
        return [{**response_item, "교정된_내용": response_item.get("내용", "") or ""}]


MAX_IMAGE_BYTES_FOR_RESPONSE = 1_400_000


def _build_diary_ocr_response_item(raw_result: dict) -> dict:
    """run() 결과에 image_data_url 등을 붙여 클라이언트 응답용 dict 생성."""
    def _str(v):
        return v if isinstance(v, str) else (v or "")
    response_item = {k: _str(v) for k, v in raw_result.items()}
    try:
        cropped_path = (response_item.get("그림_저장경로") or "").strip()
        if cropped_path:
            p = Path(cropped_path)
            if p.exists() and p.is_file():
                url = _encode_image_base64_resized(p, max_bytes=MAX_IMAGE_BYTES_FOR_RESPONSE)
                response_item["image_data_url"] = url
            else:
                response_item["image_data_url"] = None
        else:
            response_item["image_data_url"] = None
    except Exception:
        response_item["image_data_url"] = None
    response_item["교정된_내용"] = response_item.get("내용", "") or ""
    return response_item


@app.post("/diary-ocr-stream")
async def diary_ocr_stream(file: UploadFile = File(...)):
    """그림일기 OCR 처리 (진행률 스트리밍). SSE로 progress/done 이벤트 전송."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="파일을 업로드해 주세요.")

    ext = Path(file.filename).suffix or ".jpg"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"diary_{timestamp}"
    input_path = OCR_UPLOAD_DIR / f"{base_name}{ext}"
    _save_upload_file(file, input_path)

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def progress_cb(percent: int, stage: str) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, {"type": "progress", "progress": percent, "stage": stage})

    def run_pipeline() -> None:
        try:
            import diary_ocr_only
            raw_result = diary_ocr_only.run(str(input_path.resolve()), progress_callback=progress_cb)
            response_item = _build_diary_ocr_response_item(raw_result)
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "done", "result": [response_item]})
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "detail": str(e)})

    async def event_stream():
        task = asyncio.get_event_loop().run_in_executor(None, run_pipeline)
        try:
            while True:
                item = await asyncio.wait_for(queue.get(), timeout=300.0)
                if item["type"] == "progress":
                    yield f"data: {json.dumps({'progress': item['progress'], 'stage': item['stage']}, ensure_ascii=False)}\n\n"
                elif item["type"] == "done":
                    yield f"data: {json.dumps({'done': True, 'result': item['result']}, ensure_ascii=False)}\n\n"
                    break
                elif item["type"] == "error":
                    yield f"data: {json.dumps({'error': True, 'detail': item['detail']}, ensure_ascii=False)}\n\n"
                    break
        finally:
            await task

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("OCR_PORT", "8090"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

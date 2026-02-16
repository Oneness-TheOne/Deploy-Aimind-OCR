# AiMind OCR (Docker)

**그림일기 OCR** 전용 FastAPI 서비스입니다. 업로드된 그림일기 이미지에서 글씨 영역을 추출하고, Gemini를 이용해 텍스트로 인식·정리합니다.

## 기능

- **그림일기 OCR** (`POST /diary-ocr`): 이미지 업로드 → 텍스트 추출 + 후처리 → 제목/내용/그림 영역 등 구조화된 결과 반환
- 응답에 크롭된 그림을 data URL로 포함 가능 (프론트 표시용)

## 기술 스택

- **Python 3.11**, FastAPI, Uvicorn
- **Google Gemini API**: 이미지 내 텍스트 인식 및 후처리
- **Pillow**: 이미지 리사이즈·압축 (응답 크기 제한)

## 디렉터리 구조

- `main.py` — FastAPI 앱, `/diary-ocr` 등 라우트
- `ocr/` — OCR 파이프라인 모듈 (`diary_ocr_only`, `diary_ocr_pipeline` 등)
- `ocr/uploads/`, `ocr/img/` — 업로드·처리 이미지 저장 (컨테이너 내)

## 빌드 및 실행

```bash
docker build -t aimind-ocr .
docker run -p 8090:8090 --env-file .env aimind-ocr
```

## 환경 변수

- `OCR_PORT`: 서버 포트 (기본 `8090`)
- `GEMINI_API_KEY` (또는 해당 OCR 모듈에서 사용하는 키): Gemini API 키

## API

- `GET /health` — 헬스 체크
- `POST /diary-ocr` — 그림일기 이미지 업로드 → OCR 결과 JSON

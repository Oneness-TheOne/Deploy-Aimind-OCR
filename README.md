# Deploy-Aimind-OCR (아이마음 그림일기 OCR)

**아이마음**은 보호자·상담사가 아동의 그림일기를 업로드하면, AI가 글씨를 인식·정리해 주는 **아동 심리 지원 플랫폼**입니다.  
이 저장소는 **그림일기 OCR** 전용 FastAPI 서비스를 담당하며, 업로드된 그림일기 이미지에서 VLM·Gemini로 글씨 영역을 추출·후처리해 제목·내용·그림 영역 등 구조화된 결과를 반환합니다.

---

## 아이마음 프로젝트 구성

아이마음은 4개 저장소로 구성됩니다. 이 저장소(OCR)는 그림일기 이미지 → 텍스트 추출을 담당합니다.

```
[사용자] → Frontend(3000) → Backend(8000) → Aimodels(8080) / OCR(8090)
```

| 저장소 | 역할 | 기본 포트 |
|--------|------|-----------|
| **Deploy-Aimind-Frontend** | 웹 UI (로그인, 그림 분석, 그림일기 OCR, 커뮤니티, 마이페이지) | 3000 |
| **Deploy-Aimind-Backend** | REST API (인증, 사용자/아동, 분석·OCR 저장, 커뮤니티), AiModels·OCR 프록시 | 8000 |
| **Deploy-Aimind-Aimodels** | HTP 그림 분석·해석·T-Score·챗봇 (YOLO + RAG + Gemini) | 8080 |
| **Deploy-Aimind-OCR** (본 저장소) | 그림일기 이미지 → 텍스트 추출 (VLM + Gemini) | 8090 |

**전체 서비스 실행 순서 (로컬):** 1) Backend → 2) Aimodels → 3) OCR → 4) Frontend.  
Backend `.env`에 `OCR_BASE_URL=http://localhost:8090` 설정. Frontend `NEXT_PUBLIC_OCR_BASE_URL`를 같은 주소로 맞추면 됩니다.

**누구를 위한 서비스인가요?**  
- **보호자**: 자녀의 그림일기 업로드 → 글씨 인식·정리  
- **상담사·교육기관**: 아동 기록·심리 지원 참고 자료  
- **개발자**: OCR·VLM 파이프라인 참고 및 확장

---

## 기능

- **그림일기 OCR** (`POST /diary-ocr`): 이미지 업로드 → 전처리(문서 보정·그림 영역 추출) → 손글씨 OCR → Gemini 후처리 → 제목/내용/그림_저장경로 등 JSON 반환
- 응답에 크롭된 그림을 **data URL**로 포함 가능 (프론트 카드 표시용, 크기 제한 시 리사이즈)
- **스트리밍** (`POST /diary-ocr-stream`): SSE로 진행률(progress/done) 이벤트 전송 후 최종 결과 반환

---

## 기술 스택

- **Python 3.11**, FastAPI, Uvicorn
- **VLM/OCR**: 발표에서 언급한 LLAVA-ONEVISION 기반 등 — `ocr/diary_ocr_pipeline.py`에서 모델 로드·텍스트 영역 인식
- **Google Gemini API**: 텍스트 추출·후처리 (제목, 본문 정리 등)
- **Pillow**: 이미지 리사이즈·압축 (응답 data URL 크기 제한)

---

## 디렉터리 구조

| 경로 | 설명 |
|------|------|
| `main.py` | FastAPI 앱: `/health`, `/diary-ocr`, `/diary-ocr-stream` |
| `ocr/` | OCR 파이프라인 모듈 |
| `ocr/diary_ocr_only.py` | 단일 이미지 진입점 `run(이미지경로)` — 전처리 → OCR → Gemini 후처리 |
| `ocr/diary_ocr_pipeline.py` | OCR 처리 단계 정의, VLM OCR 실행 |
| `ocr/diary_pipeline.py` | 일기 처리 흐름 (이미지 저장, 텍스트 추출, Gemini 후처리 등) |
| `ocr/check.py` | 검증/테스트용 스크립트 |
| `ocr/uploads/` | 업로드 이미지 저장 (실행 시 생성) |
| `ocr/img/` | 전처리·크롭 이미지 저장 (실행 시 생성) |
| `ocr/*_diary_result.json` | 실행 시 생성되는 결과 JSON 예시 |

---

## 사전 요구사항

- **Python 3.11+**
- **Gemini API 키** (후처리용)
- (선택) Docker

---

## 빌드 및 실행

### 로컬

```bash
pip install -r requirements.txt
# .env에 GEMINI_API_KEY (또는 OCR 모듈에서 사용하는 키) 설정
python main.py
# 또는
uvicorn main:app --host 0.0.0.0 --port 8090
```

기본 포트: **8090** (`OCR_PORT` 환경 변수로 변경 가능).

### Docker (Dockerfile 제공 시)

```bash
docker build -t aimind-ocr .
docker run -p 8090:8090 --env-file .env aimind-ocr
```

---

## 환경 변수

| 변수 | 설명 |
|------|------|
| `OCR_PORT` | 서버 포트 (기본 `8090`) |
| `GEMINI_API_KEY` (또는 OCR 모듈에서 사용하는 키) | Gemini API 키 (후처리 등) |

OCR 파이프라인 내부에서 사용하는 모델 경로·키 등은 `ocr/diary_ocr_pipeline.py`, `ocr/diary_ocr_only.py`를 참고하세요.

---

## API 요약

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/health` | 헬스 체크 |
| POST | `/diary-ocr` | 그림일기 이미지 업로드 → OCR 결과 JSON (단일 항목 배열로 반환, `image_data_url` 포함 가능) |
| POST | `/diary-ocr-stream` | 동일 OCR 처리, SSE로 progress/done 스트리밍 |

- `/diary-ocr`: multipart form, 필드 `file` (이미지 파일).
- 반환 필드 예: `원본`, `날짜`, `제목`, `내용`, `그림_저장경로`, `교정된_내용`, `image_data_url`(선택).

---

## 참고

- 백엔드(Deploy-Aimind-Backend)는 `OCR_BASE_URL`로 이 서비스를 호출합니다 (`/diary-ocr`, `/diary-ocr-stream`).
- CORS는 `main.py`에서 localhost/127.0.0.1 기준으로 설정되어 있으므로, 배포 시 허용 오리진을 필요에 따라 수정하세요.


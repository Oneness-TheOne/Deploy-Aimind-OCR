# ocr

그림일기 **OCR 파이프라인** 모듈이 위치한 디렉터리입니다.

## 주요 파일

| 파일 | 설명 |
|------|------|
| `diary_ocr_only.py` | 단일 이미지 그림일기 OCR 실행 진입점 (`run(이미지경로)` 호출) |
| `diary_ocr_pipeline.py` | OCR 처리 파이프라인 단계 정의 |
| `diary_pipeline.py` | 일기 처리 흐름 (이미지 저장, 텍스트 추출, Gemini 후처리 등) |
| `check.py` | 검증/테스트용 스크립트 |

## 동작

- 업로드된 그림일기 이미지에서 글씨 영역 인식
- **Gemini**로 텍스트 추출 및 후처리 (제목, 본문, bbox 제거 등)
- 결과는 `main.py`의 `/diary-ocr` 응답으로 반환 (그림 크롭 data URL 포함 가능)

## 기타

- `uploads/`, `img/`는 Dockerfile에서 상위 또는 이 경로 아래 생성됨
- `*_diary_result.json` 등은 실행 시 생성되는 결과 파일 예시

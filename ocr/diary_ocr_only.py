"""
그림일기 OCR 전용 스크립트: 이미지 저장 + 텍스트 추출 + Gemini 후처리만 수행.
실행: python diary_ocr_only.py [이미지경로]
결과: 이미지는 ocr/img에 저장, JSON은 {이미지이름}_diary_result.json (2page3_diary_result.json 형식).
"""
import os
import re
import sys
import json
import warnings

# ocr 폴더를 path에 넣어서 diary_ocr_pipeline 임포트 (프로젝트 루트 .env는 파이프라인에서 로드)
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

import diary_ocr_pipeline as pipeline


def run(file_path: str, max_new_tokens: int = 2800, diary_mode: bool = False):
    """
    이미지 경로 하나 받아서: 전처리(이미지 저장) → OCR → Gemini 후처리 → JSON 저장.
    반환: {"원본", "날짜", "제목", "내용", "그림_저장경로"}
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {file_path}")

    warnings.filterwarnings("ignore", message=".*[Gg]lyph.*missing from font", category=UserWarning)

    if pipeline.model is None or pipeline.processor is None:
        pipeline.load_model()

    # 1) 전처리 (문서 보정 + 그림 영역 추출 → ocr/img 에 저장)
    result = pipeline.preprocess_diary_image(
        file_path,
        show_result=False,
        return_detection_vis=False,
    )

    img_path = result[5] if len(result) == 6 else (result[3] if len(result) == 4 else None)
    max_long_side = 512
    # diary_mode=False: 프롬프트 "<ocr>"만 사용 → 조기 종료 적음. True면 긴 안내문으로 일부 환경에서 짧게 끊김.

    # 2) OCR
    if len(result) == 6:
        left_pil, right_pil, _, _, _, _ = result
        raw_left, text_boxes_left, full_text_left, pil_left_sent = pipeline.run_varco_ocr(
            left_pil, max_long_side=max_long_side, diary_mode=diary_mode, max_new_tokens=max_new_tokens
        )
        raw_right, text_boxes_right, full_text_right, pil_right_sent = pipeline.run_varco_ocr(
            right_pil, max_long_side=max_long_side, diary_mode=diary_mode, max_new_tokens=max_new_tokens
        )
        full_text = full_text_left + full_text_right
        raw_output = (raw_left or "") + "\n" + (raw_right or "")
        text_boxes = list(text_boxes_left)
        w_left = pil_left_sent.size[0]
        w_right = pil_right_sent.size[0]
        w_comb = w_left + w_right
        for item in text_boxes_right:
            b = item.get("bbox")
            if b and len(b) >= 4 and max(b) <= 1.0:
                x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
                x1_new = (x1 * w_right + w_left) / w_comb
                x2_new = (x2 * w_right + w_left) / w_comb
                text_boxes.append({"text": item["text"], "bbox": [x1_new, y1, x2_new, y2]})
            else:
                text_boxes.append(item)
    elif len(result) == 4:
        pil_image, _, _, _ = result
        raw_output, text_boxes, full_text, _ = pipeline.run_varco_ocr(
            pil_image, max_long_side=max_long_side, diary_mode=diary_mode, max_new_tokens=max_new_tokens
        )
    else:
        pil_image = result[0]
        raw_output, text_boxes, full_text, _ = pipeline.run_varco_ocr(
            pil_image, max_long_side=max_long_side, diary_mode=diary_mode, max_new_tokens=max_new_tokens
        )

    min_bbox_area = 0.0003
    text_boxes = [
        item for item in text_boxes
        if item.get("bbox") and len(item["bbox"]) >= 4
        and (item["bbox"][2] - item["bbox"][0]) * (item["bbox"][3] - item["bbox"][1]) >= min_bbox_area
    ]
    full_text = "".join(item["text"] for item in text_boxes)
    display_text = pipeline.get_diary_text(raw_output, full_text)

    # 3) Gemini 후처리 (날짜·제목·내용 추출)
    diary_result = pipeline.process_diary_text(display_text)
    if img_path:
        diary_result["그림_저장경로"] = img_path

    pipeline.clean_memory()

    # 원본에서 <bbox>...</bbox>, <char>...</char> 제거 (글만 남김)
    def _strip_bbox_and_char(s):
        if not s or not isinstance(s, str):
            return ""
        t = re.sub(r"<bbox>.*?</bbox>", "", s, flags=re.DOTALL)
        t = re.sub(r"<char>.*?</char>", "", t, flags=re.DOTALL)
        t = re.sub(r"\n+", "", t).strip()
        return t

    raw_origin = (diary_result.get("원본", "") or "")
    origin_clean = _strip_bbox_and_char(raw_origin) or raw_origin

    # 4) 2page3_diary_result.json 형식으로 정리 (원본, 날짜, 제목, 내용, 그림_저장경로만)
    out = {
        "원본": origin_clean,
        "날짜": diary_result.get("날짜", "") or "",
        "제목": diary_result.get("제목", "") or "",
        "내용": diary_result.get("내용", "") or "",
        "그림_저장경로": diary_result.get("그림_저장경로", "") or "",
    }
    if isinstance(out.get("제목"), type(None)):
        out["제목"] = ""
    if isinstance(out.get("내용"), type(None)):
        out["내용"] = ""

    return out


def main():
    file_path = "2page3.jpg"
    if len(sys.argv) >= 2:
        file_path = sys.argv[1]

    file_path = os.path.abspath(file_path)
    print(f"입력: {file_path}")

    result = run(file_path)

    # JSON 저장: 입력 이미지와 같은 디렉터리에 {stem}_diary_result.json
    stem = os.path.splitext(os.path.basename(file_path))[0]
    parent = os.path.dirname(file_path)
    json_path = os.path.join(parent, f"{stem}_diary_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"이미지 저장: {result.get('그림_저장경로', '')}")
    print(f"JSON 저장: {json_path}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

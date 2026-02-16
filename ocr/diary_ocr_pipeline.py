"""
그림일기 텍스트 탐지·인식 파이프라인 (VARCO-VISION-2.0-1.7B-OCR)

아동 심리 분석용 그림일기 이미지에서 텍스트 영역을 탐지하고 OCR로 인식합니다.
- 전처리: 문서 보정 (4점 변환, adaptive cleanup)
- 모델: NCSOFT VARCO-VISION-2.0-1.7B-OCR (4-bit 양자화)
"""
import os
import sys
import gc
import re
import json
import warnings
from datetime import datetime

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, BitsAndBytesConfig
try:
    from google import genai as genai_client
except ImportError:
    genai_client = None
from dotenv import load_dotenv

# .env 파일에서 GEMINI_API_KEY 등 로드 (main.py에서 호출 시 cwd와 무관하게 프로젝트 루트 .env 사용)
_this_dir = os.path.dirname(os.path.abspath(__file__))
_env_path = os.path.join(os.path.dirname(_this_dir), ".env")
load_dotenv(_env_path)
load_dotenv()  # cwd 기준 .env도 시도 (스크립트 직접 실행 시)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# API 키는 .env 에서 읽어 사용 (GOOGLE_API_KEY, GEMINI_API_KEY 등)

# 전역 (모델 로드 후 설정)
model = None
processor = None


def check_cuda():
    """GPU/CUDA 환경 확인."""
    print(f"Python: {sys.executable}")
    print(f"torch 위치: {torch.__file__}")
    print(f"torch CUDA 버전: {torch.version.cuda}")
    print()
    cuda_available = torch.cuda.is_available()
    print(f"GPU 사용 가능: {cuda_available}")
    if cuda_available:
        dev = torch.cuda.current_device()
        print(f"장치: {torch.cuda.get_device_name(dev)}")
        print(f"VRAM: {torch.cuda.get_device_properties(dev).total_memory / 1e9:.2f} GB")
    return cuda_available


def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model():
    """VARCO OCR 모델 로드."""
    global model, processor
    clean_memory()
    globals()["model"] = None
    clean_memory()
    model_id = "NCSOFT/VARCO-VISION-2.0-1.7B-OCR"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    print("모델 로딩 중 (4-bit)...")
    loaded = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    proc = AutoProcessor.from_pretrained(model_id)
    torch.backends.cudnn.benchmark = True
    globals()["model"] = loaded
    globals()["processor"] = proc
    print("모델 로드 완료.")


def _strip_model_special_tokens(text: str) -> str:
    """모델이 조기 종료 시 넣는 특수 토큰·태그 제거 (서버 등에서 <|im_end|>, </char> 잔여 등)."""
    if not text:
        return text
    s = text
    s = re.sub(r"<\|im_end\|>", "", s)
    s = re.sub(r"<\|im_start\|>", "", s)
    s = re.sub(r"<\|[^|]+\|>", "", s)  # 기타 <|...|> 형태
    # 조기 종료로 남은 불완전 태그 제거 (원본 텍스트에 </char>, <char> 안 남기기)
    s = s.replace("</char>", "").replace("<char>", "")
    return s.strip()


def parse_ocr_output(output_text):
    output_text = _strip_model_special_tokens(output_text or "")
    text_boxes = []
    full_text_parts = []
    pattern = re.compile(r"<char>\s*([^<]*?)\s*</char>\s*<bbox>\s*([^<]*?)\s*</bbox>", re.DOTALL)
    for m in pattern.finditer(output_text):
        text = m.group(1).strip()
        bbox_str = m.group(2).strip()
        full_text_parts.append(text)
        try:
            coords = [float(x.strip()) for x in bbox_str.split(",")]
            if len(coords) >= 4:
                text_boxes.append({"text": text, "bbox": coords[:4]})
        except (ValueError, AttributeError):
            text_boxes.append({"text": text, "bbox": None})
    full_text = "".join(full_text_parts) if full_text_parts else output_text
    return text_boxes, full_text


def get_diary_text(raw_output, full_text):
    s = (full_text or "").strip()
    if len(s) > 4:
        from collections import Counter
        c = Counter(s.replace(" ", "").replace("\n", ""))
        most = c.most_common(1)[0] if c else ("", 0)
        if most[1] >= len(s) * 0.5:
            raw_clean = re.sub(r"<char>.*?</char>", "", raw_output or "", flags=re.DOTALL)
            raw_clean = re.sub(r"<bbox>.*?</bbox>", "", raw_clean, flags=re.DOTALL)
            raw_clean = raw_clean.replace("<ocr>", "").strip()
            raw_clean = _strip_model_special_tokens(raw_clean)
            return raw_clean if raw_clean else (raw_output or full_text)
    out = full_text or raw_output or ""
    return _strip_model_special_tokens(out) if out else ""


def run_varco_ocr(pil_image, max_long_side=800, max_new_tokens=1400, diary_mode=False):
    w, h = pil_image.size
    if max(w, h) > max_long_side:
        scale = max_long_side / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        pil_image = pil_image.resize((new_w, new_h), resample=Image.LANCZOS)
    pil_image_sent = pil_image
    ocr_prompt = "<ocr>\n이미지에 손으로 쓴 그림일기 글을 위에서 아래로 한 줄씩 그대로 읽어서 출력해줘." if diary_mode else "<ocr>"
    conversation = [{"role": "user", "content": [{"type": "image", "image": pil_image}, {"type": "text", "text": ocr_prompt}]}]
    inputs = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min(400, max_new_tokens // 2),  # EOS 조기 종료 완화: 최소 400토큰 생성
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )
    generated = generate_ids[0][len(inputs.input_ids[0]):]
    raw_output = processor.decode(generated, skip_special_tokens=False)
    raw_output = _strip_model_special_tokens(raw_output)
    text_boxes, full_text = parse_ocr_output(raw_output)
    return raw_output, text_boxes, full_text, pil_image_sent


def imshow(title, image):
    # 시각화 전용 함수 (현재는 비활성화됨)
    # plt.figure(figsize=(12, 12))
    # plt.title(title)
    # if len(image.shape) == 3:
    #     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # else:
    #     plt.imshow(image, cmap="gray")
    # plt.axis("off")
    # plt.show()
    pass


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))


def expand_contour(cnt, scale=1.03):
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return cnt
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_new = cnt_scaled + [cx, cy]
    return cnt_new.astype(np.int32)


def adaptive_cleanup(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
    bg_estimation = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    normalized = cv2.divide(gray, bg_estimation, scale=255)
    smoothed = cv2.bilateralFilter(normalized, d=3, sigmaColor=30, sigmaSpace=30)
    otsu_thresh, _ = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cutoff = min(otsu_thresh + 35, 255)
    _, clean_result = cv2.threshold(smoothed, cutoff, 255, cv2.THRESH_TRUNC)
    final_output = cv2.normalize(clean_result, None, 0, 255, cv2.NORM_MINMAX)
    _, binary_inv = cv2.threshold(final_output, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_inv, connectivity=8)
    h_img, w_img = binary_inv.shape[0], binary_inv.shape[1]
    min_len_h = max(50, int(0.15 * w_img))
    min_len_v = max(50, int(0.15 * h_img))
    aspect = 4
    no_lines = binary_inv.copy()
    for i in range(1, num_labels):
        w, h = int(stats[i, 2]), int(stats[i, 3])
        if w >= aspect * h and w >= min_len_h:
            no_lines[labels == i] = 0
        elif h >= aspect * w and h >= min_len_v:
            no_lines[labels == i] = 0
    kernel_thick = np.ones((1, 1), np.uint8)
    thickened = cv2.dilate(no_lines, kernel_thick)
    final_output = 255 - thickened
    return final_output


def extract_drawing_from_diary(warped_bgr, diary_type, output_dir="./ocr/img", base_name="drawing"):
    """
    그림일기에서 그림 영역만 추출하여 컬러로 저장.
    warped_bgr: 4점 투영 변환 완료된 BGR 이미지
    diary_type: "horizontal" | "vertical"
    Returns: (img_path: str, drawing_grayscale: np.ndarray)
    """
    h, w = warped_bgr.shape[:2]

    if diary_type == "horizontal":
        # 가로 버전: 왼쪽 페이지만 사용, weight_image_crop_process 크롭
        img_for_crop = warped_bgr[:, : w // 2]
        CROP_TOP, CROP_BOTTOM, CROP_LEFT = 0.20, 0.05, 0.10
        CROP_RIGHT = 0  # 우측 유지
    else:
        # 세로 버전: 전체 사용, length_image_crop_process 크롭
        img_for_crop = warped_bgr
        CROP_TOP, CROP_BOTTOM, CROP_LEFT, CROP_RIGHT = 0.22, 0.35, 0.03, 0.03

    height, width = img_for_crop.shape[:2]
    x_start = int(width * CROP_LEFT)
    x_end = width if diary_type == "horizontal" else int(width * (1 - CROP_RIGHT))
    y_start = int(height * CROP_TOP)
    y_end = int(height * (1 - CROP_BOTTOM))

    cropped_bgr = img_for_crop[y_start:y_end, x_start:x_end].copy()

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{timestamp}.jpg"
    img_path = os.path.join(output_dir, filename)
    cv2.imwrite(img_path, cropped_bgr)

    drawing_grayscale = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY)
    return img_path, drawing_grayscale


def preprocess_diary_image(file_path, show_result=False, use_color_for_ocr=True, return_detection_vis=False):
    """문서 영역 보정 + cleanup. return_detection_vis=True면 원본에 탐지된 문서 영역(4각형) 그린 이미지도 반환."""
    image = cv2.imread(file_path)
    if image is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {file_path}")
    orig = image.copy()
    ratio = image.shape[0] / 500.0
    h, w = 500, int(image.shape[1] / ratio)
    image_resized = cv2.resize(image, (w, h))
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 50, 150)
    kernel_edge = np.ones((5, 5), np.uint8)
    edged = cv2.dilate(edged, kernel_edge, iterations=1)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break
    detection_vis_bgr = None
    if screenCnt is None:
        warped = orig
        if return_detection_vis:
            detection_vis_bgr = orig.copy()
            cv2.putText(detection_vis_bgr, "Document not detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
    else:
        screenCnt = expand_contour(screenCnt, scale=1.03)
        pts_orig = (screenCnt.reshape(4, 2) * ratio).astype(np.int32)
        warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
        if return_detection_vis:
            detection_vis_bgr = orig.copy()
            cv2.polylines(detection_vis_bgr, [pts_orig], True, (0, 255, 0), 3)
            for i, pt in enumerate(pts_orig):
                cv2.circle(detection_vis_bgr, tuple(pt), 8, (0, 0, 255), -1)
                cv2.putText(detection_vis_bgr, str(i + 1), tuple(pt), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    h_warped, w_warped = warped.shape[:2]
    is_horizontal = w_warped > h_warped
    diary_type = "horizontal" if is_horizontal else "vertical"
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")
    img_path, _ = extract_drawing_from_diary(warped, diary_type, output_dir=output_dir, base_name=base_name)

    if is_horizontal:
        left_half = warped[:, :w_warped // 2]
        right_half = warped[:, w_warped // 2:]
        h_left, w_left = left_half.shape[:2]
        left_cropped = left_half[0 : int(h_left * 0.2), 0 : int(w_left * 0.70)]
        left_clean = adaptive_cleanup(left_cropped)
        right_clean = adaptive_cleanup(right_half)
        left_pil = Image.fromarray(cv2.cvtColor(left_clean, cv2.COLOR_GRAY2RGB))
        right_pil = Image.fromarray(cv2.cvtColor(right_clean, cv2.COLOR_GRAY2RGB))
        h_left, w_left = left_clean.shape[:2]
        h_right, w_right = right_clean.shape[:2]
        if h_left != h_right:
            pad_height = h_right - h_left
            left_padded = np.vstack([left_clean, np.ones((pad_height, w_left), dtype=left_clean.dtype) * 255])
            final_bgr = cv2.cvtColor(np.hstack([left_padded, right_clean]), cv2.COLOR_GRAY2BGR)
        else:
            final_bgr = cv2.cvtColor(np.hstack([left_clean, right_clean]), cv2.COLOR_GRAY2BGR)
        if return_detection_vis and detection_vis_bgr is not None:
            cv2.putText(detection_vis_bgr, "Horizontal: left(top20%,left55%) -> right OCR", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
        # if show_result:
        #     imshow("Left (top20%, left55%)", left_clean)
        #     imshow("Right page", right_clean)
        det = detection_vis_bgr if return_detection_vis else None
        return left_pil, right_pil, final_bgr, det, "horizontal", img_path
    else:
        if return_detection_vis and detection_vis_bgr is not None:
            cv2.putText(detection_vis_bgr, "Vertical version: full image", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
        final_result = adaptive_cleanup(warped)
        final_bgr = cv2.cvtColor(final_result, cv2.COLOR_GRAY2BGR)
        pil_image = Image.fromarray(cv2.cvtColor(final_result, cv2.COLOR_GRAY2RGB))
        # if show_result:
        #     imshow("Smart Clean Scan Result", final_result)
        det = detection_vis_bgr if return_detection_vis else None
        if return_detection_vis and detection_vis_bgr is not None:
            return pil_image, final_bgr, detection_vis_bgr, "vertical", img_path
        return pil_image, final_bgr, "vertical", img_path


def _normalize_date(date_str: str) -> str:
    date_str = date_str.strip()
    patterns = [
        (r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일", "%Y-%m-%d"),
        (r"(\d{4})-(\d{1,2})-(\d{1,2})", "%Y-%m-%d"),
        (r"(\d{4})(\d{2})(\d{2})", "%Y-%m-%d"),
    ]
    for pat, _ in patterns:
        m = re.search(pat, date_str.replace(" ", ""))
        if m:
            g = m.groups()
            y, mo, d = int(g[0]), int(g[1]) if len(g) > 1 else 1, int(g[2]) if len(g) > 2 else 1
            return f"{y:04d}-{mo:02d}-{d:02d}"
    return date_str


def _extract_diary_info_gemini(text: str) -> dict:
    api_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
    if not api_key or not api_key.strip():
        raise ValueError("GEMINI_API_KEY 또는 GOOGLE_API_KEY 환경변수를 설정해주세요.")
    if genai_client is None:
        raise ImportError("google-genai 패키지가 필요합니다. pip install google-genai")
    prompt = (
        "다음은 어린이가 쓴 그림일기 텍스트입니다. 띄어쓰기가 없고 오타가 많습니다.\n\n"
        "다음 JSON 형식으로만 답해주세요. 다른 설명 없이 JSON만 출력하세요.\n\n"
        '{"날짜": "YYYY-MM-DD 형식", "제목": "추출된 제목", "내용": "맞춤법과 띄어쓰기를 완벽히 교정한 내용"}\n\n'
        "규칙:\n"
        "- 날짜: 텍스트에서 추출한 날짜를 YYYY-MM-DD 형식으로\n"
        "- 제목: 일기의 제목만 추출\n"
        "- 내용: 본문 내용을 문맥에 맞게 맞춤법 교정 및 띄어쓰기 완벽 수정\n\n"
        "입력 텍스트:\n"
    )
    full_prompt = prompt + text
    client = genai_client.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=full_prompt,
    )
    raw_text = (response.text if hasattr(response, "text") and response.text else "").strip()
    if "```" in raw_text:
        raw_text = re.sub(r"```(?:json)?\n?", "", raw_text)
        raw_text = raw_text.replace("```", "").strip()
    data = json.loads(raw_text)
    data["날짜"] = _normalize_date(data.get("날짜", ""))
    return data


def process_diary_text(text: str) -> dict:
    """OCR 결과를 Gemini로 분석하여 날짜·제목·내용 추출."""
    result = {"원본": text}
    try:
        extracted = _extract_diary_info_gemini(text)
        result["날짜"] = extracted.get("날짜", "")
        result["제목"] = extracted.get("제목", "")
        result["내용"] = extracted.get("내용", "")
    except Exception as e:
        result["오류"] = str(e)
    return result


def _progress(pct: int, msg: str):
    """진행률 출력 (0~100%)."""
    bar_len = 20
    filled = int(bar_len * pct / 100)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r[{bar}] {pct:3d}% | {msg}", end="", flush=True)
    if pct >= 100:
        print()


def run(
    file_path: str = "2page15.jpg",
    max_long_side: int = 512,
    show_preprocess: bool = True,
    show_detection: bool = True,
    save_detection_images: bool = False,
    diary_mode: bool = False,
    show_progress: bool = True,
):
    """그림일기 이미지 경로로 전처리 → OCR → Gemini 후처리 → 결과 출력."""
    global model, processor
    prog = _progress if show_progress else lambda p, m: None

    if model is None or processor is None:
        load_model()
        prog(5, "모델 로드 완료")

    if not os.path.exists(file_path):
        print(f"파일을 찾을 수 없습니다: {file_path}")
        return

    warnings.filterwarnings("ignore", message=".*[Gg]lyph.*missing from font", category=UserWarning)

    try:
        prog(10, "전처리 중 (문서 영역 탐지·보정)...")
        result = preprocess_diary_image(file_path, show_result=show_preprocess, return_detection_vis=True)

        prog(30, "OCR 진행 중...")
        img_path = None
        if len(result) == 6:
            left_pil, right_pil, preprocessed_bgr, detection_vis_bgr, version, img_path = result
            raw_left, text_boxes_left, full_text_left, pil_left_sent = run_varco_ocr(left_pil, max_long_side=max_long_side, diary_mode=diary_mode)
            raw_right, text_boxes_right, full_text_right, pil_right_sent = run_varco_ocr(right_pil, max_long_side=max_long_side, diary_mode=diary_mode)
            full_text = full_text_left + full_text_right
            raw_output = (raw_left or "") + "\n" + (raw_right or "")
            text_boxes = text_boxes_left.copy()
            w_left = pil_left_sent.size[0]
            w_right = pil_right_sent.size[0]
            w_comb = w_left + w_right
            h_comb = max(pil_left_sent.size[1], pil_right_sent.size[1])
            for item in text_boxes_right:
                b = item.get("bbox")
                if b and len(b) >= 4 and max(b) <= 1.0:
                    x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
                    x1_new = (x1 * w_right + w_left) / w_comb
                    x2_new = (x2 * w_right + w_left) / w_comb
                    text_boxes.append({"text": item["text"], "bbox": [x1_new, y1, x2_new, y2]})
                else:
                    text_boxes.append(item)
            pil_image_sent = Image.new("RGB", (w_comb, h_comb), (255, 255, 255))
            pil_image_sent.paste(pil_left_sent, (0, 0))
            pil_image_sent.paste(pil_right_sent, (w_left, 0))
        elif len(result) == 5:
            pil_image, preprocessed_bgr, detection_vis_bgr, version, img_path = result
            raw_output, text_boxes, full_text, pil_image_sent = run_varco_ocr(pil_image, max_long_side=max_long_side, diary_mode=diary_mode)
        elif len(result) == 4:
            pil_image, preprocessed_bgr, version, img_path = result
            detection_vis_bgr = None
            raw_output, text_boxes, full_text, pil_image_sent = run_varco_ocr(pil_image, max_long_side=max_long_side, diary_mode=diary_mode)
        else:
            pil_image, preprocessed_bgr, *rest = result
            detection_vis_bgr = None
            version = rest[0] if len(rest) >= 1 else None
            img_path = rest[1] if len(rest) >= 2 else None
            raw_output, text_boxes, full_text, pil_image_sent = run_varco_ocr(pil_image, max_long_side=max_long_side, diary_mode=diary_mode)

        prog(50, "OCR 완료. 텍스트 후처리 중...")
        print(f"\n전처리 완료: {file_path}" + (f" ({version} version)" if version else ""))
        if img_path:
            print(f"그림 영역 저장: {img_path}")

        # if detection_vis_bgr is not None:
        #     if show_detection:
                # plt.figure(figsize=(12, 12))
                # plt.title("문서 영역 탐지 결과")
                # plt.imshow(cv2.cvtColor(detection_vis_bgr, cv2.COLOR_BGR2RGB))
                # plt.axis("off")
                # plt.show()
            if save_detection_images:
                out_path = file_path.rsplit(".", 1)[0] + "_detection.jpg"
                cv2.imwrite(out_path, detection_vis_bgr)
                print(f"탐지 결과 저장: {out_path}")

        if not text_boxes and (raw_output or "").strip():
            print("텍스트 박스 0개. 모델 원시 출력:\n", (raw_output or "")[:1000])

        min_bbox_area = 0.0003
        text_boxes = [
            item for item in text_boxes
            if item.get("bbox") and len(item["bbox"]) >= 4
            and (item["bbox"][2] - item["bbox"][0]) * (item["bbox"][3] - item["bbox"][1]) >= min_bbox_area
        ]
        full_text = "".join(item["text"] for item in text_boxes)

        display_text = get_diary_text(raw_output, full_text)
        prog(60, "Gemini로 일기 분석 중 (날짜·제목·내용 추출)...")
        diary_result = process_diary_text(display_text)
        if img_path:
            diary_result["그림_저장경로"] = img_path
        prog(80, "Gemini 분석 완료. 결과 정리 중...")
        diary_result_json = json.dumps(diary_result, ensure_ascii=False, indent=2)

        print("\n" + "=" * 50)
        print("일기 분석 결과 (JSON)")
        print("=" * 50)
        print(diary_result_json)
        print("=" * 50)
        print("추출된 텍스트")
        print("=" * 50)
        print(display_text if display_text.strip() else raw_output)
        print("=" * 50)
        if text_boxes:
            print("\n텍스트 박스 (상위 20개)")
            for i, item in enumerate(text_boxes[:20]):
                print(f"  [{i+1}] {item['text']!r} -> bbox: {item['bbox']}")
            if len(text_boxes) > 20:
                print(f"  ... 외 {len(text_boxes) - 20}개")

        vis = cv2.cvtColor(np.array(pil_image_sent), cv2.COLOR_RGB2BGR)
        h_vis, w_vis = vis.shape[:2]
        if text_boxes:
            for item in text_boxes:
                bbox = item.get("bbox")
                if bbox is None or len(bbox) < 4:
                    continue
                x1, y1, x2, y2 = bbox
                if max(x1, y1, x2, y2) <= 1.0:
                    x1, x2 = int(x1 * w_vis), int(x2 * w_vis)
                    y1, y2 = int(y1 * h_vis), int(y2 * h_vis)
                else:
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        prog(90, "시각화 생성 중...")
        # print("\n글자 위치 탐지 결과 (바운딩 박스)")
        # plt.figure(figsize=(12, 12))
        # plt.title("OCR Bounding Boxes")
        # plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        # plt.axis("off")
        # plt.show()

        img_draw = pil_image_sent.copy().convert("RGB")
        draw = ImageDraw.Draw(img_draw)
        w_img, h_img = img_draw.size
        for item in text_boxes or []:
            bbox = item.get("bbox")
            if bbox is None or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = bbox
            if max(x1, y1, x2, y2) <= 1.0:
                x1, x2 = x1 * w_img, x2 * w_img
                y1, y2 = y1 * h_img, y2 * h_img
            box = [(int(x1), int(y1)), (int(x2), int(y1)), (int(x2), int(y2)), (int(x1), int(y2))]
            color = (255, 0, 0) if len(item.get("text", "").strip()) > 1 else (0, 255, 0)
            draw.polygon(box, outline=color, width=5)
        # print("\n탐지 확인 (빨강=여러 글자, 초록=한 글자)")
        # plt.figure(figsize=(12, 12))
        # plt.title("Text boxes (PIL polygon)")
        # plt.imshow(img_draw)
        # plt.axis("off")
        # plt.show()

        # 최종 JSON 형식으로 확인
        print("\n" + "=" * 60)
        print("최종 JSON 결과")
        print("=" * 60)
        print(diary_result_json)
        print("=" * 60)
        # JSON 파일로 저장
        json_path = file_path.rsplit(".", 1)[0] + "_diary_result.json"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(diary_result_json)
        print(f"JSON 저장: {json_path}")

        prog(100, "완료!")
        clean_memory()
    except Exception as e:
        print(f"오류: {e}")
        import traceback
        traceback.print_exc()
        clean_memory()


if __name__ == "__main__":
    # CLI: python diary_ocr_pipeline.py <이미지경로> → 서브프로세스/배치용 (diary_mode=True, show_progress=False)
    if len(sys.argv) >= 2:
        file_path = sys.argv[1]
        run(
            file_path=file_path,
            max_long_side=512,
            show_preprocess=False,
            show_detection=False,
            save_detection_images=False,
            diary_mode=True,
            show_progress=False,
        )
    else:
        file_path = "2page3.jpg"
        run(
            file_path=file_path,
            max_long_side=512,
            show_preprocess=True,
            show_detection=True,
            save_detection_images=False,
            diary_mode=False,
            show_progress=True,
        )


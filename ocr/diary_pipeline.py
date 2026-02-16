"""
그림일기 파이프라인 - Gemini를 이용한 일기 텍스트 추출·교정
"""
import os

os.environ["GEMINI_API_KEY"] = "AIzaSyBEEXDdD6c9IFVJ2-EJ1lwVpPwLmsDRueM"  # 여기에 실제 키 입력

import json
import re
import requests
import pandas as pd
import google.generativeai as genai

# OpenWeatherMap용 위도/경도 (한국 지역)
areaA = "도봉구"
areaB = "강남구"
areaC = "속초시"

AREA_COORDS = {
    "도봉구": (37.6688, 127.0471),
    "강남구": (37.5146, 127.0496),
    "속초시": (38.2070, 128.5918),
}


def get_weather(date: str, location: str) -> dict:
    """날짜와 지역에 맞는 날씨 정보를 반환. API 키가 없거나 실패 시 Mock 데이터 반환."""
    api_key = os.environ.get("OPENWEATHERMAP_API_KEY", "")

    if not api_key or not api_key.strip():
        return {"temp": 5, "description": "맑음", "humidity": 60}

    if location not in AREA_COORDS:
        return {"temp": 5, "description": "맑음", "humidity": 60}

    lat, lon = AREA_COORDS[location]
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric&lang=kr"

    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return {
            "temp": round(data["main"]["temp"]),
            "description": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
        }
    except Exception:
        return {"temp": 5, "description": "맑음", "humidity": 60}


def format_weather_str(weather: dict) -> str:
    """날씨 dict를 DataFrame용 문자열로 변환."""
    return f"{weather['temp']}°C, {weather['description']}"


def normalize_date(date_str: str) -> str:
    """다양한 날짜 형식을 YYYY-MM-DD로 정규화."""
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


def extract_diary_info(raw_text: str) -> dict:
    """어린이 그림일기 텍스트에서 날짜, 제목, 내용을 추출하고 내용을 맞춤법/띄어쓰기 교정."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
    if not api_key or not api_key.strip():
        raise ValueError(
            "GEMINI_API_KEY 또는 GOOGLE_API_KEY 환경변수를 설정해주세요. (Google AI Studio에서 무료 API 키 발급 가능)"
        )

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
    full_prompt = prompt + raw_text
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(full_prompt)
    text = response.text.strip()

    # 마크다운 코드블록 제거
    if "```" in text:
        text = re.sub(r"```(?:json)?\n?", "", text)
        text = text.replace("```", "").strip()

    data = json.loads(text)
    data["날짜"] = normalize_date(data.get("날짜", ""))
    return data


def main():
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", ""))

    raw_diaries = [
        "2026년2월2일제목누나의생일\n오늘은누나의일이었다.그러서연케이크도먹고좋들끼리위식도즐거웠다나는놀이공원에가고싶었는데못가서과식했그래도가족들이행복한시간을보내서좋았다.",
        # "2026년1월31일요일제목보성녹차\n오늘은보성녹차발에갔다녹차아이스크림이맛있었다음엔두개고싶다",
        # "2026년2월2일읽었어요!제목:엄마애빠랑짓구경을갔다\n오늘은엄마랑아빠랑풀구경을했다.너무재미였다!다음에또노르가고싶다!",
    ]

    areas = [areaA, areaB, areaC]

    results = []
    for raw, area in zip(raw_diaries, areas):
        try:
            extracted = extract_diary_info(raw)
            date_str = extracted["날짜"]
            weather = get_weather(date_str, area)
            weather_str = format_weather_str(weather)
            results.append({
                "원본": raw,
                "날짜": date_str,
                "지역": area,
                "날씨": weather_str,
                "제목": extracted["제목"],
                "교정된_내용": extracted["내용"],
            })
        except Exception as e:
            results.append({
                "원본": raw,
                "날짜": "",
                "지역": area,
                "날씨": "오류",
                "제목": "",
                "교정된_내용": f"처리 오류: {e}",
            })

    df = pd.DataFrame(results, columns=["원본", "날짜", "지역", "날씨", "제목", "교정된_내용"])

    # 원본, 교정된_내용 전체 내용 보기 (끝까지 표시)
    pd.set_option("display.max_colwidth", None)
    print(df.to_string())

    print("\n--- 전치 ---")
    df_transposed = df.transpose()
    print(df_transposed.to_string())

    # JSON 형식으로 출력
    print("\n--- JSON 형식 ---")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



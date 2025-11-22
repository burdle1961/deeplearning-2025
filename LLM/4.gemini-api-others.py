import google.generativeai as genai

import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")

genai.configure(api_key=API_KEY)

from google.generativeai.types import HarmCategory, HarmBlockThreshold

def with_parameters():

    print("=== 파라미터 조정 예제 ===")
    
    generation_config = {
        "temperature": 0.3,  # 창의성 조절 (0.0 ~ 2.0)
        "top_p": 0.95,       # 다양성 조절
        "top_k": 40,         # 후보 토큰 수
        "max_output_tokens": 2048,  # 최대 출력 길이
    }

    model = genai.GenerativeModel(
        'gemini-2.5-flash',
        generation_config=generation_config
    )
    
    response = model.generate_content(
        "창의적인 AI 스타트업 아이디어 1가지만 짧게 제안해줘"
    )

    print(response.text)
    print()


def with_safety_settings():

    print("=== 안전 설정 예제 ===")

    # 5가지 차단 레벨
    # HarmBlockThreshold.BLOCK_NONE           # 차단 안 함
    # HarmBlockThreshold.BLOCK_ONLY_HIGH      # 높은 확률만 차단
    # HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE  # 중간 이상 차단
    # HarmBlockThreshold.BLOCK_LOW_AND_ABOVE  # 낮은 확률 이상 차단
    # HarmBlockThreshold.HARM_BLOCK_THRESHOLD_UNSPECIFIED  # 기본값 (보통 MEDIUM)
    safety_settings = {
        # HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }
    
    model = genai.GenerativeModel(
        'gemini-2.5-flash',
        safety_settings=safety_settings
    )
    
    response = model.generate_content("제2차 세계대전 중 나치의 유대인 박해에 대해 설명해줘")
    print(response.text)
    print("\n")

def system_instruction_example():

    print("=== 시스템 명령어 예제 ===")
    
    model = genai.GenerativeModel(
        'gemini-2.5-flash',
        system_instruction="당신은 친절한 파이썬 프로그래밍 튜터입니다. 초보자도 이해하기 쉽게 설명해주세요."
    )
    
    response = model.generate_content("데코레이터가 뭐야?")
    print(response.text)
    print("\n")


def main():

    # system_instruction_example()

    # with_safety_settings()

    with_parameters()
    
if __name__ == "__main__":
    print("=" * 60)
    print("Gemini AI API 파라미터 조정, 안전 설정, 시스템 명령어 예제")
    print("=" * 60)
    print()
    main()
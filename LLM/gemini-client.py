import os
from dotenv import load_dotenv
import google.generativeai as genai

# 환경 변수 로드
load_dotenv()
API_KEY = os.environ.get("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")

# Gemini API 설정
genai.configure(api_key=API_KEY)


def simple_text_generation():
    """기본 텍스트 생성 예제"""
    print("=== 기본 텍스트 생성 ===")
    # gemini-2.5-flash 또는 gemini-pro 사용 가능
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    response = model.generate_content("RAG에 대하여 간단히 설명해줘.")
    print(response.text)
    print("\n")


def chat_conversation():
    """대화형 채팅 예제"""
    print("=== 대화형 채팅 ===")
    model = genai.GenerativeModel('gemini-2.5-flash')
    chat = model.start_chat(history=[])
    
    messages = [
        "RAG가 뭐야?",
        "그럼 벡터 데이터베이스는 어떤 역할을 하는거야?",
        "추천할만한 벡터 DB가 있어?"
    ]
    
    for msg in messages:
        response = chat.send_message(msg)
        print(f"👤 사용자: {msg}")
        print(f"🤖 AI: {response.text}\n")


def streaming_response():
    """스트리밍 응답 예제"""
    print("=== 스트리밍 응답 ===")
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    print("👤 사용자: LangChain에 대해 설명해줘")
    print("🤖 AI: ", end='', flush=True)
    
    response = model.generate_content(
        "LangChain에 대해 설명해줘",
        stream=True
    )
    
    for chunk in response:
        print(chunk.text, end='', flush=True)
    print("\n\n")


def with_parameters():
    """생성 파라미터 조정 예제"""
    print("=== 파라미터 조정 예제 ===")
    
    generation_config = {
        "temperature": 0.7,  # 창의성 조절 (0.0 ~ 2.0)
        "top_p": 0.95,       # 다양성 조절
        "top_k": 40,         # 후보 토큰 수
        "max_output_tokens": 1024,  # 최대 출력 길이
    }
    
    model = genai.GenerativeModel(
        'gemini-2.5-flash',
        generation_config=generation_config
    )
    
    response = model.generate_content(
        "창의적인 AI 스타트업 아이디어 3가지를 제안해줘"
    )
    print(response.text)
    print("\n")


    # # 1. Prompt Feedback 확인 (입력 자체가 차단되었는지)
    # if hasattr(response, 'prompt_feedback'):
    #     print(f"Prompt Feedback: {response.prompt_feedback}")
    #     if response.prompt_feedback.block_reason:
    #         print(f"⚠️ 프롬프트가 차단됨: {response.prompt_feedback.block_reason}")

    # # 2. Candidates 확인
    # print(f"\nCandidates 수: {len(response.candidates)}")

    # if response.candidates:
    #     candidate = response.candidates[0]
        
    #     # Finish Reason
    #     print(f"\nFinish Reason: {candidate.finish_reason}")
    #     print(f"Finish Reason Name: {candidate.finish_reason.name}")
    #     print(f"Finish Reason Value: {candidate.finish_reason.value}")
        
    #     # Safety Ratings
    #     print("\n안전 등급:")
    #     for rating in candidate.safety_ratings:
    #         print(f"  {rating.category.name}: {rating.probability.name} (blocked: {rating.blocked})")
        
    #     # Content 확인
    #     if candidate.content and candidate.content.parts:
    #         print(f"\nContent Parts: {len(candidate.content.parts)}")
    #         for i, part in enumerate(candidate.content.parts):
    #             print(f"  Part {i}: {part.text[:100] if hasattr(part, 'text') else 'No text'}")
    #     else:
    #         print("\n⚠️ Content가 비어있음")

    # # 3. 텍스트 접근 시도
    # print("\n텍스트 접근 시도:")
    # try:
    #     print(response.text)
    # except ValueError as e:
    #     print(f"❌ Error: {e}")
    # except AttributeError as e:
    #     print(f"❌ AttributeError: {e}")

def with_safety_settings():
    """안전 설정 예제"""
    print("=== 안전 설정 예제 ===")
    
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }
    
    model = genai.GenerativeModel(
        'gemini-2.5-flash',
        safety_settings=safety_settings
    )
    
    response = model.generate_content("AI 윤리에 대해 설명해줘")
    print(response.text)
    print("\n")


def system_instruction_example():
    """시스템 명령어 예제"""
    print("=== 시스템 명령어 예제 ===")
    
    model = genai.GenerativeModel(
        'gemini-2.5-flash',
        system_instruction="당신은 친절한 파이썬 프로그래밍 튜터입니다. 초보자도 이해하기 쉽게 설명해주세요."
    )
    
    response = model.generate_content("데코레이터가 뭐야?")
    print(response.text)
    print("\n")


def list_models():
    """사용 가능한 모델 목록 조회"""
    print("=== 사용 가능한 모델 목록 ===")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
    print("\n")


def main():
    """메인 함수"""
    try:
        # 1. 모델 목록 확인
        list_models()
        
        # 2. 기본 텍스트 생성
        simple_text_generation()
        
        # 3. 대화형 채팅
        chat_conversation()
        
        # 4. 스트리밍 응답
        streaming_response()
        
        # 5. 파라미터 조정
        with_parameters()
        
        # 6. 안전 설정
        with_safety_settings()
        
        # 7. 시스템 명령어
        system_instruction_example()
        
    except Exception as e:
        print(f"오류 발생: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Gemini AI API 클라이언트 예제")
    print("=" * 60)
    print()
    main()
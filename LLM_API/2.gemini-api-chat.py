import google.generativeai as genai

import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")

# Gemini 모델 설정
genai.configure(api_key=API_KEY)

    
messages = [
    "RAG가 뭐야?",
    "벡터 데이터베이스는 어떤 역할을 하는거야?",
    "추천할만한 벡터 DB가 있어?"
]

def simple_text_generation(msg):
    
    print (f">>> 질문 : {msg}")
    response = model.generate_content(msg)
    print(f"<<< 답변 : {response.text}")
    print("\n")



def chat_conversation(msgs):

    chat = model.start_chat(history=[])
    
    for msg in msgs:
        response = chat.send_message(msg)
        print(f">>> 사용자: {msg}")
        print(f"<<< AI   : {response.text}\n")


if __name__ == "__main__":

    print ("Loading Model")
    model = genai.GenerativeModel('gemini-2.5-flash')

    print("=== 단일 질문 생성 ===")
    for msg in messages :
        simple_text_generation(msg)    

    print("=== 대화형 채팅 ===")
    chat_conversation(messages)
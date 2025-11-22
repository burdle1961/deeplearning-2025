import google.generativeai as genai

import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")

# Gemini 모델 설정
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

response = model.generate_content("RAG에 대하여 간단히 설명해줘.")
print(response.text)
print("\n")
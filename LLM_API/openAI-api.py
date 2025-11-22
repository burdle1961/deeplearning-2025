from openai import OpenAI

import os
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)     # 또는 환경 변수로 설정 후 api_key 생략 가능

# ChatGPT 호출
response = client.chat.completions.create(
    model="gpt-4o-mini",    # 모델 지정 (gpt-4o, gpt-4o-mini, gpt-3.5-turbo 등 가능)
    messages=[
    {"role": "system", "content": "You are an expert for neo4j."},
    {"role": "user", "content": "neo4j의 APOC에 대하여 설명해줘"}
    ],
    temperature=0.7,        # 창의성 정도
)
# 응답 출력
print("**********************************\n\n")
print(response.choices[0].message.content)

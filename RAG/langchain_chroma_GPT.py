print ("LangChain ChromaDB + ChatGPT RAG Example")
print ("Loading Libraries.... take some time...")

import torch
from transformers import AutoTokenizer, AutoModel
import chromadb
from chromadb.utils import embedding_functions
from chromadb.api import EmbeddingFunction

# from langchain_community.vectorstores import Chroma       # depreciated
from langchain_chroma import Chroma                         # pip install -U langchain-chroma

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# =========================
# 1) HuggingFace 모델 로드
# =========================
print ("VectorDB embedding model Loading...")
model_name = "jhgan/ko-sroberta-multitask"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# =========================
# 2) ChromaDB EmbeddingFunction 구현
# =========================
class LocalHuggingFaceEmbeddingFunction(EmbeddingFunction):
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    # 문서 임베딩 (리스트)
    def embed_documents(self, texts):
        return self._embed(texts)

    # 쿼리 임베딩 (단일 str)
    def embed_query(self, text):
        return self._embed([text])[0]

    # 내부 임베딩 처리
    def _embed(self, texts):
        embeddings = []
        with torch.no_grad():
            for text in texts:
                tokens = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                )
                outputs = self.model(**tokens)
                last_hidden_state = outputs.last_hidden_state
                embedding = last_hidden_state.mean(dim=1).squeeze().numpy()
                embeddings.append(embedding)
        return embeddings

    def name(self):
        return "local_hf_embedding"


# 로컬 임베딩 객체
hf_ef = LocalHuggingFaceEmbeddingFunction(tokenizer, model)

# =========================
# 3) 기존 ChromaDB 로드
# =========================
print ("ChromaDB Loading...")
persist_dir = "C:/Users/burdl/Downloads"

vector_db = Chroma(
    persist_directory=persist_dir,
    embedding_function=hf_ef,
    collection_name="HYWU-RuleDocuments"   # 기존 컬렉션 이름
)

# =========================
# 4) Retriever 생성
# =========================
retriever = vector_db.as_retriever(search_kwargs={"k": 3})  
# =========================
# 5) LLM (ChatGPT API) 연결
# =========================
import os
from dotenv import load_dotenv
load_dotenv()
# openai_api_key = os.environ.get("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# =========================
# 6) 프롬프트 템플릿
# =========================
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "한양여자대학교의 학칙 규정에 대한 내용입니다. "
     "질문과 관련되어 제공된 아래 내용을 참고해서 답변해 주세요  "
    #  "If unsure, say '모르겠습니다'. 간결하게 요약을 먼저 말해주고, 조건이 있다면 조건별로 설명해 주세요. \n\n"
     "Question: {question}\nContext:\n{context}\nAnswer:")
])

# =========================
# 7) RAG 실행 함수
# =========================
def rag_chain(question: str):
    # 유사 문서 검색
    docs = retriever.invoke(question)
    # print(docs)
    # 규정집으로 만든 chromaDB의 내용은 아래와 같음.
    # docs.id
    # docs.metadata
    # docs.page_content

    context = "\n".join([doc.page_content for doc in docs])
    print ("="*20,"RAG 검색 결과", "="*20)
    print (context)
    # 프롬프트 생성
    messages = prompt.invoke({"question": question, "context": context})
    # print (messages)
    print ("------------")
    # LLM 호출
    answer = llm.invoke(messages)
    # print("Token usage:", answer.llm_output["token_usage"])
    print (answer.usage_metadata)
    # usage_metadata['input_tokens']
    # usage_metadata['output_tokens']
    # usage_metadata['total_tokens']
    # input_token_details
    print ("------------")
    return answer.content

# =========================
# 8) 실행 예시
# =========================
if __name__ == "__main__":
    while True :
        q = input("질문을 입력하세요 : ")
        if q == "quit" : break
        print(rag_chain(q))

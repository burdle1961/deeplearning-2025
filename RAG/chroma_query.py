from transformers import AutoTokenizer, AutoModel
import torch
import chromadb
from chromadb.api.types import EmbeddingFunction

# 1) HuggingFace 모델 로드
model_name = "jhgan/ko-sroberta-multitask"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# 2) ChromaDB EmbeddingFunction 클래스 구현
class LocalHuggingFaceEmbeddingFunction(EmbeddingFunction):
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def __call__(self, input):  # ← 여기서 매개변수명은 반드시 input 이어야 함
        embeddings = []
        with torch.no_grad():
            for text in input:
                tokens = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                outputs = self.model(**tokens)
                last_hidden_state = outputs.last_hidden_state
                embedding = last_hidden_state.mean(dim=1).squeeze().numpy()
                embeddings.append(embedding)
        return embeddings

    def name(self):
        return "local_hf_embedding"

# 3) 객체 생성
hf_ef = LocalHuggingFaceEmbeddingFunction(tokenizer, model)
# ChromaDB 클라이언트 연결 (DB 경로에 맞게 수정)
client = chromadb.PersistentClient(path="C:/Users/burdl/Downloads")

# 기존 컬렉션 로드 (컬렉션 이름 확인 필요)
collection = client.get_collection(
    name="HYWU-RuleDocuments",
    embedding_function=hf_ef
)

# 예시 질문
query_text = input("질문을 입력하세요: ")
max_matches = 3

# 벡터 DB 쿼리
result = collection.query(
    query_texts=[query_text],
    n_results=max_matches,
)

# 결과 출력
for i in range(max_matches):
    print("="*10, f"Match {i+1}", "="*10)
    print("ID:", result["ids"][0][max_matches - i - 1])
    print("Distance:", result["distances"][0][max_matches - i - 1])
    print("Document:", result["documents"][0][max_matches - i - 1])
    # 필요하면 메타데이터도 출력
    # print("Metadata:", result["metadatas"][0][max_matches - i - 1])
    print()

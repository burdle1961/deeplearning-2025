# Image 기반의 PDF 문서 (스캔본)은 인식 불가. 그러한 문서는 OCR 처리 필요
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import re

# 1. PDF 로드 및 텍스트 추출 (페이지별)
# document = "C:\\Users\\burdl\\OneDrive\\2025-R&D\\RAG\\HYWU\\54.전문기술석사과정 학사운영규정(2-0-54).pdf"
document = "C:\\Users\\burdl\\OneDrive\\2025-R&D\\RAG\\HYWU\\21.학사학위 전공심화과정 운영 규정(2-0-21).pdf"
# document = "C:\\Users\\burdl\\OneDrive\\2025-R&D\\RAG\\HYWU\\30.조기취업자 학사관리규정(2-0-30).pdf"

loader = PyPDFLoader(document)
docs = loader.load()  # 각 페이지별 Document 리스트

# 2. 문단 단위 텍스트 분할기 설정
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n","\n", "\t", " "], 
    chunk_size=1000,      # 최대 청크 문자 수
    chunk_overlap=50      # 청크 간 중복 문자 수
)

# 3. 각 페이지 문서를 문단 단위로 분할
split_docs = []
for doc in docs:
    splits = text_splitter.split_text(doc.page_content)

    for split in splits:
        split = split.replace("\x00", " ")
        split = split.replace("◎", "")
        #split = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\u200B-\u200D\uFEFF]", "**", split)
        # split = re.sub(r"학사학위 전공심화과정 운영 규정\(2-0-21\)  \d / \d", "", split)
        split = re.sub(r"전문기술석사과정 학사운영규정\(2-0-54\)  \d / \d", "", split)
        # split = re.sub(r"조기취업자 학사관리규정\(2-0-30\) \d / \d", "", split)
        if (len(split) > 0) : 
            split_docs.append(
            {       "text": split, "metadata": doc.metadata}  # 문단, 메타데이터 저장
            )
        # print(doc.metadata)

    # pattern = r'학사학위 전공심화과정 운영 규정(2-0-21)  \d+ / \d+'
    # print (re.split(pattern, split))

parts = []      # 최종 결과.

# 모든 페이지에 대하여 처리
for doc in split_docs:      

    paragraph = doc["text"]
    # print (doc["metadata"])
    # print (paragraph)

    # "제 숫자 장" 패턴을 기준으로 분리 (분리할 때 패턴도 결과에 포함)
    pattern = r'(제\s*\d+\s*장)'    # ""장"으로 분리된 문서
    #pattern = r'(제\s*\d+\s*조)'    # "조"로 분리된 문서

    # split은 패턴이 사라지므로 split 대신 finditer로 위치를 찾아 잘라내기
    matches = list(re.finditer(pattern, paragraph))
    
    if len(matches) == 0 :         # 장 표시가 없는 페이지 (마지막 페이지 처리 포함)
            if parts :
                parts[-1][2] = parts[-1][2] + paragraph.strip()

    for i in range(len(matches)):
        # print (f"**** matchs {i}, {matches[i].start()}")
        start = matches[i].start()

        if i == 0 and start > 0 :           # 앞 페이지 paragraph에 추가할 내용
            # print (paragraph[:start].strip())
            if parts :
                parts[-1][2] = parts[-1][2] + paragraph[:start].strip()

        # 다음 분할 위치 바로 전까지만 슬라이싱
        end = matches[i+1].start() if i+1 < len(matches) else len(paragraph)
        
        # parts.append(paragraph[start:end].strip())
        parts.append([doc["metadata"],matches[i].group(), paragraph[matches[i].end():end]])

        # print ("GROUP : ", matches[i].group())
        # print ("TEXT : ", paragraph[matches[i].end():end])


# para[0] : 문단 제목 (제 x 장)
# para[1] : metadata
# para[2] : 규정 내용 (text body)
for para in parts :
    print (">>", para[1])
    print (type(para[0]), para[0])             # mata data 출력
    # print (para[2])                 # text data 출력

print ("="*20, "PDF 문서 파싱 완료", "="*20)
# chromaDB에 등록
# from langchain.embeddings import HuggingFaceEmbeddings
# from sentence_transformers import SentenceTransformer, CrossEncoder
# import chromadb
# from chromadb.utils import embedding_functions

# #model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
# # s_t = embedding_functions.HuggingFaceEmbeddingFunction(
# #     model_name="paraphrase-multilingual-mpnet-base-v2"   # 성능/속도 균형
# # )
# # hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# hf_ef = embedding_functions.HuggingFaceEmbeddingFunction(
#     model_name="jhgan/ko-sroberta-multitask"
# )
# # Chroma의 embedding_function 래퍼로 감싸기
# def embed_func(texts):
#     return hf_embeddings.embed_documents(texts)

# client = chromadb.PersistentClient(path="d:/VectorDB/HYWU")

# collection = client.get_or_create_collection(
#     name="HYWU-RuleDocuments",
#     embedding_function=hf_ef,
#     metadata={"hnsw:space": "cosine"}
# )
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

# 4) ChromaDB 컬렉션 생성
client = chromadb.PersistentClient(path="C:/Users/burdl/Downloads")

collection = client.get_or_create_collection(
    name="HYWU-RuleDocuments",
    embedding_function=hf_ef,
    metadata={"hnsw:space": "cosine"}
)

print ("="*20, "데이터 준비 완료", "="*20)

# 데이터 준비
# 인덱스
ids = []
# 메타데이터
doc_meta = []
# 벡터로 변환 저장할 텍스트 데이터로 ChromaDB에 Embedding 데이터가 없으면 자동으로 벡터로 변환해서 저장한다.
documents = []

for i, para in enumerate(parts) :

    print (">>>>", para[1])
    ids.append(re.sub(r"\s+", "", document)+ " " + para[1] + " " + str(i))
    doc_meta.append({"meta": str(para[0])})
    documents.append(para[2])

print ("=" * 20, "파싱 결과 --> 벡터 DB(chromaDB) 생성", "=" * 20)

# DB 저장
collection.add(
    documents=documents,
    metadatas=doc_meta,
    ids=ids
)
print ("=" * 20,"벡터 DB 생성 완료", "=" * 20)

# # DB 쿼리
# max_matches = 3
# result = collection.query(
#     query_texts=["전공 심화 과정의 수업은 어떻게 진행되나요?"],
#     n_results=max_matches,
# )

# for i in range(0,max_matches) :
#     print (result["ids"][0][i])
#     print (result["distances"][0][i])
#     # print (result["metadatas"][0][i])
#     print (result["documents"][0][i])

#     print ()
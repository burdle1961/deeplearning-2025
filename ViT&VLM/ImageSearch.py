#2-1. 이미지 리사이즈 후 MongoDB에 바이너리 저장
import os
import pymongo
from PIL import Image
from bson.binary import Binary

# MongoDB 클라이언트 연결
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
db = mongo_client["image_db"]
image_col = db["images"]

print ("MongoDB connected")

# 리사이즈 & 저장 함수
def save_resized_to_mongo(img_path, size=(320, 240)):
    img = Image.open(img_path).convert("RGB")
    resized = img.resize(size)
    # 바이너리 변환
    from io import BytesIO
    buffer = BytesIO()
    resized.save(buffer, format="JPEG")
    binary_data = buffer.getvalue()
    doc = {
        "filename": os.path.basename(img_path),
        "data": Binary(binary_data)
    }
    result = image_col.insert_one(doc)
    return str(result.inserted_id), resized

print("DINO embedding started.")

#2-2. DINO 임베딩 및 ChromaDB 등록
from transformers import AutoImageProcessor, AutoModel
import torch
import chromadb
from chromadb.config import Settings

# DINO 모델 준비

print ("DINO model preparing...", end="")
image_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base', use_fast=False)
image_encoder = AutoModel.from_pretrained('facebook/dinov2-base')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_encoder.to(device)
image_encoder.eval()

print ("loaded")

def get_dino_embedding(pil_img):
    inputs = image_processor(images=pil_img, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = image_encoder(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
    return embedding.tolist()

# ChromaDB server 설정으로 실행
client = chromadb.Client(Settings(
    chroma_server_host="localhost",
    chroma_server_http_port=8000
))
print("ChromaDB server runs")

# ChromaDB 클라이언트
client = chromadb.PersistentClient(path="chroma_db_test")
print("ChromaDB client runs")

class DinoEmbeddingFunction:
    def __call__(self, input):
        return [get_dino_embedding(img) for img in input]
    def name(self):
        return "dino"  # 임의의 함수명(식별자) 문자열 지정
    
dino_fn = DinoEmbeddingFunction()
#client.delete_collection("images")
collection = client.get_or_create_collection("images", embedding_function=dino_fn)

#2-3. 샘플 이미지 일괄 저장 및 인덱싱
from glob import glob
import numpy as np

image_paths = glob("image_folder/*.jpg")  # 샘플 이미지 경로
mongo_ids = []
metadatas = []
image_ids = []
image_documents = []

for idx, path in enumerate(image_paths):
    print(path)
    mongo_id, resized_img = save_resized_to_mongo(path)
    image_documents.append(np.array(resized_img))  # (O) numpy array로 변환 후 저장
    image_ids.append(f"img_{idx}")
    metadatas.append({
        "orig_path": path,
        "mongo_id": mongo_id
    })

collection.add(
    images=image_documents,
    ids=image_ids,
    metadatas=metadatas
)

#3-0. MongoDB에서 결과 이미지 가져오기 및 저장

from bson import ObjectId
from io import BytesIO
import matplotlib.pyplot as plt

def retrieve_and_imshow(mongo_id):
    doc = image_col.find_one({"_id": ObjectId(mongo_id)})
    img_bytes = BytesIO(doc["data"])
    pil_img = Image.open(img_bytes)
    plt.imshow(pil_img)
    plt.axis('off')
    plt.show()


def retrieve_and_save(mongo_id, out_filename):
    doc = image_col.find_one({"_id": ObjectId(mongo_id)})
    with open(out_filename, "wb") as f:
        f.write(doc["data"])

#3-1. 쿼리 이미지와 유사한 이미지 검색
# 쿼리 이미지는 반드시 320x240으로 resize해서 PIL로 로딩 (동일 전처리)
# 유사도 결과에 따라 filtering하는 방법도 가능

while True :
    query_img_path = input("Sample Image (q to Quit): ")
    if (query_img_path == "q") : break
    query_pil_img = Image.open(query_img_path).convert("RGB").resize((320, 240))
    query_np_img = np.array(query_pil_img)
    results = collection.query(
        query_images=[query_np_img],
        n_results=5     # 상위 5 개
    )


    # original image를 가져옴.
    for i, (result_id, meta) in enumerate(zip(results['ids'][0], results['metadatas'][0])):
        mongo_id = meta['mongo_id']
        out_file = f"retrieved_{result_id}.jpg"

        similarity = results['distances'][0][i]  # 유사도 또는 거리 값
        print(f"복원 이미지: {out_file}, 유사도(L2 거리): {similarity}")
        #retrieve_and_save(mongo_id, out_file)
        retrieve_and_imshow(mongo_id)

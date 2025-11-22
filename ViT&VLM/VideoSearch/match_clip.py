import os
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np

# CLIP 모델 및 프로세서 초기화
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 폴더명, 목록 파일명, 질의 입력
path = input("폴더명: ")
list_file = "unique_frames_dino.txt"        # 이 파일을 읽어서 CLIP으로 유사도 검색 (DINO dedup 결과)
text_query = input("여행지 분위기 설명: ")

# 지정된 목록 파일에서 파일명만 추출
with open(os.path.join(path, list_file), "r") as f:
    target_files = [line.strip() for line in f if line.strip()]

print(f"{list_file}에는 {len(target_files)}개의 파일명이 있습니다.")

# 텍스트 임베딩(L2 정규화) 추출
text_inputs = processor(text=[text_query], return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    text_emb = model.get_text_features(**text_inputs)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

similarity_scores = []

for img_file in target_files:
    image_path = os.path.join(path, img_file)
    if not os.path.exists(image_path):
        print(f"{img_file}가 폴더에 존재하지 않습니다.")
        continue
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Failed to open {img_file}: {e}")
        continue

    # 이미지 임베딩(L2 정규화) 추출
    image_inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_emb = model.get_image_features(**image_inputs)
        image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)

    # 코사인 유사도 계산
    similarity = (text_emb @ image_emb.T).squeeze().item()
    similarity_scores.append((img_file, similarity))
    print (img_file, similarity)

# === 통계 정보 출력 ===
if similarity_scores:
    similarity_values = np.array([score for _, score in similarity_scores])
    total_images = len(similarity_scores)
    mean_similarity = similarity_values.mean()
    std_similarity = similarity_values.std()
    max_similarity = similarity_values.max()

    print("\n=== 유사도 통계 ===")
    print(f"총 이미지 수: {total_images}")
    print(f"평균 유사도: {mean_similarity:.4f}")
    print(f"유사도 표준편차: {std_similarity:.4f}")
    print(f"최대 유사도: {max_similarity:.4f}")
else:
    print("유사도 정보를 계산할 수 없습니다. 유효한 이미지가 없습니다.")

# 유사도 기준 필터 및 정렬(선택, 옵션)
threshold = np.percentile(similarity_values, 90)    # 상위 10%를 기준으로 해당되는 이미지를 선택
filtered = [item for item in similarity_scores if item[1] >= threshold]

print(f"'{text_query}'와 코사인 유사도가 {threshold} 이상인 이미지 수: {len(filtered)}")
print("유사한 이미지 목록:")
for fn, score in filtered:
    print(f"{fn}: {score:.4f}")

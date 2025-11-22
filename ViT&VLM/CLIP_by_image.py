from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import glob
import os
import matplotlib.pyplot as plt

# 사용할 한글 폰트 경로 지정 (윈도우: 'malgun.ttf', 나눔고딕 등)
import matplotlib.font_manager as fm
from matplotlib import rc
font_path = "C:/Windows/Fonts/malgun.ttf"  # 또는 원하는 한글 폰트 ttf 경로
font_name = fm.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

# 찾고자하는 이미지를 설명하는 Text Label 정보
text_labels = ["apples",
               "a red apple",
               "a photo of bananas",
               "a beautiful scenery with sunset, an ocean view",
               "a small island in the west sea in Korea",
               "a famous backpacking spot in the island",
               "archery player",
               "a picture of cat",
               "a dog",
               "a photo of a puppy",
               "Paris and Effel Tower",
               "Paris",
               "Tokyo with Fuji mountain",
               "a city of Seoul",
               "olympic games",
               "Sports Utility Vehicle(SUV), Land Rover",
               "Sports Utility Vehicle(SUV), G-Wagen",
               "a picture of Seoul, Lotte Tower and Han River",
               "a picture of Seoul, a palace",
               "beach, island",
               "a boat in the ocean"]

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 폴더 경로 설정
folder_path = "C:\\Users\\burdl\\OneDrive\\2025-R&D\\ViT&VLM\\image_folder2"

# 모든 jpg 파일 찾기
jpg_files = glob.glob(os.path.join(folder_path, "*.jpg"))

# 각 이미지 파일 처리
threshold = 0.5
for img_path in jpg_files:
    #print(f"Processing: {os.path.basename(img_path)}")
    
    # 이미지 로드
    image = Image.open(img_path)

    inputs = processor(text=text_labels, images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    # Top 1 결과 추출
    top1_index = torch.argmax(probs[0])
    top1_value = probs[0][top1_index]
    confidence = top1_value.item()

    label = text_labels[top1_index.item()]
    #print(f"Prediction: {label} ({top1_value.item():.4f})")

    if confidence >= threshold :

        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"{os.path.basename(img_path)}\n{label}\nConfidence: {confidence:.4f}", 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()

    #print(logits_per_image)
    #print("Label probs:", probs)

    top3_values, top3_indices = torch.topk(probs[0], k=3)
    print(f"Predictions for {img_path}")        
    for i, (value, idx) in enumerate(zip(top3_values, top3_indices)):
        label = text_labels[idx.item()]
        #if (value.item() < 0.8) : break # threshold = 0.8
        print(f"\t{i+1}. {label}: {value.item():.4f}")

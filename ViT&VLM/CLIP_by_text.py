from transformers import CLIPProcessor, CLIPModel
import glob
import os
from PIL import Image
import torch
import matplotlib.pyplot as plt

# 사용할 한글 폰트 경로 지정 (윈도우: 'malgun.ttf', 나눔고딕 등)
import matplotlib.font_manager as fm
from matplotlib import rc
font_path = "C:/Windows/Fonts/malgun.ttf"  # 또는 원하는 한글 폰트 ttf 경로
font_name = fm.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

# KoCLIP 모델 사용 예시 (한국어 지원 모델) : 아래 repo 의 내용이 변경될 수 있음
# from transformers import AutoProcessor, AutoModel
# repo = "Bingsu/clip-vit-large-patch14-ko"
# model = AutoModel.from_pretrained(repo)
# processor = AutoProcessor.from_pretrained(repo)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def visualize_text_to_images(folder_path, text_query, model, processor, k=3):
    """
    텍스트와 유사한 이미지들을 찾아서 시각화
    """
    # 이미지들 찾기
    top_images = find_top_k_images_for_text(folder_path, text_query, model, processor, k)
    
    if not top_images:
        print("No images found!")
        return
    
    # 시각화
    cols = 3
    rows = (len(top_images) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    fig.suptitle(f"'{text_query}'와 유사한 이미지들", fontsize=16, fontweight='bold')
    
    for i, (image, filename, score) in enumerate(top_images):
        if i < len(axes):
            axes[i].imshow(image)
            axes[i].axis('off')
            axes[i].set_title(f"{filename}\n{score:.4f}", fontsize=10)
    
    # 남은 subplot 숨기기
    for i in range(len(top_images), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def find_top_k_images_for_text(folder_path, text_query, model, processor, k=3):
    """
    텍스트와 가장 유사한 상위 K개 이미지 찾기
    """
    # 이미지 로드 (위와 동일)
    jpg_files = glob.glob(os.path.join(folder_path, "*.jpg"))
    images = []
    file_names = []
    
    for img_path in jpg_files:
        try:
            image = Image.open(img_path)
            images.append(image)
            file_names.append(os.path.basename(img_path))
        except:
            continue
    
    # CLIP 추론
    inputs = processor(text=text_query, images=images, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_text = outputs.logits_per_text
    probs = logits_per_text.softmax(dim=1)[0]
    
    # Top-K 이미지 찾기
    top_k_values, top_k_indices = torch.topk(probs, k=min(k, len(images)))
    
    print(f"'{text_query}'와 유사한 상위 {k}개 이미지:")
    top_images = []
    for i, (value, idx) in enumerate(zip(top_k_values, top_k_indices)):
        img_idx = idx.item()
        score = value.item()
        filename = file_names[img_idx]
        print(f"{i+1}. {filename}: {score:.4f}")
        top_images.append((images[img_idx], filename, score))
    
    return top_images

folder_path = "C:\\Users\\burdl\\OneDrive\\2025-R&D\\ViT&VLM\\image_folder"
while True :
    text_query = input("검색할 문장 ('quit' to finish): ")
    if (text_query == 'quit') : break
    visualize_text_to_images(folder_path, text_query, model, processor, k=6)
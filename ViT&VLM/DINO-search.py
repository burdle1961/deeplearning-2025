import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import glob

# 사용할 한글 폰트 경로 지정 (윈도우: 'malgun.ttf', 나눔고딕 등)
import matplotlib.font_manager as fm
from matplotlib import rc
font_path = "C:/Windows/Fonts/malgun.ttf"  # 또는 원하는 한글 폰트 ttf 경로
font_name = fm.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

class DINOImageSearcher:
    def __init__(self):
        # 1. Load pretrained DINO model
        self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
        self.model.eval()
        
        # 2. Image preprocessing
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])
        
        self.image_features = {}
        self.image_paths = []
    
    def extract_features(self, image_path):
        """단일 이미지에서 DINO 특징을 추출"""
        img = Image.open(image_path).convert("RGB")
        inp = self.transform(img).unsqueeze(0)
        
        with torch.no_grad():
            feat = self.model(inp)
            return feat[0].cpu().numpy()
    
    def load_database(self, folder_path, extensions=['*.jpg', '*.jpeg', '*.png', '*.bmp']):
        """폴더에서 모든 이미지를 로드하고 특징을 추출"""
        print("데이터베이스 이미지들을 로딩 중...")
        
        # 지원하는 확장자의 모든 이미지 파일 찾기
        all_images = []
        for ext in extensions:
            # all_images.extend(glob.glob(os.path.join(folder_path, ext)))
            all_images.extend(glob.glob(os.path.join(folder_path, ext.upper())))
        
        self.image_paths = all_images
        
        for i, img_path in enumerate(self.image_paths):
            try:
                features = self.extract_features(img_path)
                self.image_features[img_path] = features
                print(f"처리됨 ({i+1}/{len(self.image_paths)}): {os.path.basename(img_path)}")
            except Exception as e:
                print(f"오류 발생 {img_path}: {e}")
        
        print(f"총 {len(self.image_features)}개 이미지 로드 완료!")
    
    def search(self, query_image_path, top_k=5):
        """쿼리 이미지와 가장 유사한 상위 k개 이미지 검색"""
        if not self.image_features:
            print("먼저 load_database()를 실행하세요!")
            return
        
        # 쿼리 이미지 특징 추출
        query_features = self.extract_features(query_image_path)
        
        # 모든 데이터베이스 이미지와의 코사인 유사도 계산
        similarities = []
        valid_paths = []
        
        for img_path, db_features in self.image_features.items():
            # 코사인 유사도 계산
            similarity = cosine_similarity([query_features], [db_features])[0][0]
            similarities.append(similarity)
            valid_paths.append(img_path)
        
        # 유사도 순으로 정렬
        sorted_indices = np.argsort(similarities)[::-1]
        
        # 상위 k개 결과 반환
        results = []
        for i in range(min(top_k, len(sorted_indices))):
            idx = sorted_indices[i]
            results.append({
                'path': valid_paths[idx],
                'similarity': similarities[idx],
                'filename': os.path.basename(valid_paths[idx])
            })
        
        return results
    
    def display_results(self, query_image_path, results):
        """검색 결과를 시각화"""
        num_results = len(results)
        fig, axes = plt.subplots(2, max(2, num_results // 2), figsize=(15, 8))
        fig.suptitle('이미지 검색 결과 (DINO + Cosine Similarity)', fontsize=16)
        
        # 쿼리 이미지 표시
        query_img = Image.open(query_image_path)
        axes[0, 0].imshow(query_img)
        axes[0, 0].set_title(f'쿼리 이미지\n{os.path.basename(query_image_path)}', fontsize=10)
        axes[0, 0].axis('off')
        
        # 검색 결과 표시
        for i, result in enumerate(results):
            row = i // 2
            col = i % 2
            
            try:
                img = Image.open(result['path'])
                axes[row, col].imshow(img)
                axes[row, col].set_title(
                    f"순위 {i+1}: {result['filename']}\n"
                    f"유사도: {result['similarity']:.4f}", 
                    fontsize=9
                )
                axes[row, col].axis('off')
            except Exception as e:
                print(f"이미지 로드 오류 {result['path']}: {e}")
        
        # 빈 서브플롯 숨기기
        for i in range(num_results + 1, len(axes.flat)):
            axes.flat[i].axis('off')
        
        plt.tight_layout()
        plt.show()

# 사용 예시
if __name__ == "__main__":
    # 검색 시스템 초기화
    searcher = DINOImageSearcher()
    
    # 데이터베이스 폴더 경로 설정
    database_folder = "image_folder"  # 여기에 실제 폴더 경로 입력
    
    # 데이터베이스 로드
    searcher.load_database(database_folder)
    
    # 검색할 쿼리 이미지 경로
    query_image = input("검색 이미지 : ")
    
    # 검색 실행 (상위 5개 결과)
    results = searcher.search(query_image, top_k=4)
    
    # 결과 출력
    print("\n=== 검색 결과 ===")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['filename']} - 유사도: {result['similarity']:.4f}")
    
    # 결과 시각화
    searcher.display_results(query_image, results)

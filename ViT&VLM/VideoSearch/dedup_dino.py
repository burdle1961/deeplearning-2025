import os
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import argparse
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import re

# 경고 메시지 억제
warnings.filterwarnings('ignore')

class VideoFrameDeduplicator:
    def __init__(self, folder_path, similarity_threshold=0.95, model_name='dino_vits16', 
                 device=None, window_size=5):
        """
        비디오 프레임 중복 제거 클래스 (인접 프레임만 비교)
        
        Args:
            folder_path (str): 이미지가 있는 폴더 경로
            similarity_threshold (float): 유사도 임계값 (0.0-1.0, 높을수록 엄격)
            model_name (str): DINO 모델 이름
            device (str): 디바이스 ('cuda', 'cpu', None=자동선택)
            window_size (int): 비교할 인접 프레임 윈도우 크기 (기본값: 5)
        """
        self.folder_path = Path(folder_path)
        self.similarity_threshold = similarity_threshold
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.window_size = window_size
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
        print(f"사용 디바이스: {self.device}")
        print(f"DINO 모델: {model_name}")
        print(f"인접 프레임 윈도우 크기: {window_size}")
        
        # DINO 모델 로드
        self.model = self._load_dino_model()
        
        # 이미지 전처리 변환
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def _load_dino_model(self):
        """DINO 모델을 로드합니다."""
        try:
            print("DINO 모델을 로드하는 중...")
            model = torch.hub.load('facebookresearch/dino:main', self.model_name)
            model = model.to(self.device)
            model.eval()
            print("DINO 모델 로드 완료!")
            return model
        except Exception as e:
            print(f"DINO 모델 로드 실패: {e}")
            print("인터넷 연결을 확인하고 다시 시도해주세요.")
            raise
    
    def extract_frame_number(self, filename):
        """파일명에서 프레임 번호를 추출합니다."""
        # 일반적인 패턴들: frame_001.jpg, img_123.png, 001.jpg, video_001_frame_456.jpg 등
        patterns = [
            r'frame_(\d+)',
            r'img_(\d+)', 
            r'image_(\d+)',
            r'(\d+)(?=\.[^.]*$)',  # 확장자 직전의 숫자
            r'_(\d+)_',
            r'_(\d+)$'
        ]
        
        filename_lower = filename.lower()
        for pattern in patterns:
            matches = re.findall(pattern, filename_lower)
            if matches:
                return int(matches[-1])  # 마지막 매치된 번호 사용
        
        return 0  # 번호를 찾을 수 없으면 0 반환
    
    def get_image_files(self):
        """폴더에서 이미지 파일 목록을 가져와서 프레임 순서대로 정렬합니다."""
        image_files = []
        for ext in self.image_extensions:
            image_files.extend(self.folder_path.glob(f'*{ext}'))
            #image_files.extend(self.folder_path.glob(f'*{ext.upper()}'))
        
        # 프레임 번호로 정렬 (번호가 같으면 파일명으로 정렬)
        def sort_key(path):
            frame_num = self.extract_frame_number(path.stem)
            return (frame_num, path.name)
        
        sorted_files = sorted(image_files, key=sort_key)
        
        # 정렬 결과 확인용 출력 (처음 몇 개만)
        #for i, file_path in enumerate(sorted_files[:5]) :
        #    frame_num = self.extract_frame_number(file_path.stem)
        #    print(f"  {i+1}. {file_path.name} (frame: {frame_num})")
         
        return sorted_files
    
    def extract_features(self, image_path):
        """DINO 모델을 사용하여 이미지 특성을 추출합니다."""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(image_tensor)
            
            features = features.cpu().numpy().flatten()
            features = features / np.linalg.norm(features)
            
            return features
            
        except Exception as e:
            print(f"특성 추출 실패 {image_path}: {e}")
            return None
    
    def calculate_similarity(self, features1, features2):
        """두 특성 벡터 간의 코사인 유사도를 계산합니다."""
        if features1 is None or features2 is None:
            return 0.0
        
        similarity = cosine_similarity([features1], [features2])[0][0]
        similarity = (similarity + 1) / 2  # 0~1로 정규화
        
        return float(similarity)
    
    def get_image_info(self, image_path):
        """이미지의 기본 정보를 가져옵니다."""
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                file_size = image_path.stat().st_size
                
                return {
                    'width': width,
                    'height': height,
                    'file_size': file_size,
                    'resolution': width * height
                }
        except Exception as e:
            print(f"이미지 정보 가져오기 실패 {image_path}: {e}")
            return None
    
    def choose_better_image(self, img1_path, img2_path):
        """두 유사한 이미지 중 더 나은 이미지를 선택합니다."""
        info1 = self.get_image_info(img1_path)
        info2 = self.get_image_info(img2_path)
        
        if info1 is None:
            return img2_path
        if info2 is None:
            return img1_path
        
        # 해상도가 높은 것을 우선
        if info1['resolution'] != info2['resolution']:
            return img1_path if info1['resolution'] > info2['resolution'] else img2_path
        
        # 파일 크기가 큰 것을 우선
        return img1_path if info1['file_size'] > info2['file_size'] else img2_path
    
    def process_images_sequential(self):
        """인접한 프레임들만 비교하여 중복을 제거합니다."""
        image_files = self.get_image_files()
        
        if not image_files:
            print("처리할 이미지 파일이 없습니다.")
            return [], []
        
        if len(image_files) < 2:
            print("비교할 이미지가 충분하지 않습니다.")
            return image_files, []
        
        print(f"총 {len(image_files)}개의 비디오 프레임을 발견했습니다.")
        print("인접 프레임 비교를 시작합니다...")
        
        unique_images = []
        duplicate_groups = []
        comparison_count = 0
        
        i = 0
        while i < len(image_files):
            print(f"\n프레임 {i+1} ({image_files[i].name}) 처리 중...")
            
            # 현재 프레임 (n번째) 특성 추출
            current_image = image_files[i]
            current_features = self.extract_features(current_image)
            
            # 연속된 유사한 프레임들 찾기
            consecutive_duplicates = []
            j = i + 1
            
            # n+1, n+2, n+3... 순차적으로 비교
            while j < len(image_files):
                next_image = image_files[j]
                next_features = self.extract_features(next_image)
                
                # 비교 횟수 증가
                comparison_count += 1
                
                # 현재 프레임(n)과 다음 프레임(n+k) 비교
                similarity = self.calculate_similarity(current_features, next_features)
                
                print(f"  → 프레임 {j+1} 비교: {similarity:.4f}")
                
                if similarity >= self.similarity_threshold:
                    # 유사한 프레임 발견 - 중복 목록에 추가
                    consecutive_duplicates.append(next_image)
                    j += 1
                else:
                    break
            
            # 현재 프레임을 대표로 선택 (품질 체크 후)
            if consecutive_duplicates:
                # 중복 그룹에서 최고 품질 찾기
                all_in_group = [current_image] + consecutive_duplicates
                best_image = current_image
                
                for img in consecutive_duplicates:
                    if self.choose_better_image(best_image, img) != best_image:
                        best_image = img
                
                unique_images.append(best_image)
                
                print(f"그룹 대표 이미지 : {best_image.name} (총 {len(all_in_group)}개 프레임 그룹)")
                
                # 중복 그룹 정보 저장 (유사도는 간소화)
                duplicate_groups.append({
                    'selected': str(best_image),
                    'duplicates': [str(img) for img in all_in_group if img != best_image],
                    'group_size': len(all_in_group),
                    'frame_range': f"{self.extract_frame_number(all_in_group[0].stem)}-"
                                 f"{self.extract_frame_number(all_in_group[-1].stem)}"
                })
            else:
                # 중복이 없는 독립 프레임
                unique_images.append(current_image)
                print(f"독립 프레임: {current_image.name}")
            
            # 다음 처리할 프레임으로 이동 (중복된 프레임들은 모두 건너뛰기)
            if j < len(image_files):
                i = j  # 유사하지 않은 프레임부터 다시 시작
                #print(f"다음 시작점: 프레임 {i+1}")
            else:
                i = len(image_files)  # 모든 프레임 처리 완료
        
        print(f"\n처리 완료! 총 비교 횟수: {comparison_count}번")
        print(f"   원본 프레임: {len(image_files)}개")
        print(f"   고유 프레임: {len(unique_images)}개")
        print(f"   제거된 프레임: {len(image_files) - len(unique_images)}개")
        
        return unique_images, duplicate_groups
    
    def save_results(self, unique_images, duplicate_groups):
        """결과를 파일로 저장합니다."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 고유 이미지 목록을 텍스트 파일로 저장
        unique_list_path = self.folder_path / f"unique_frames_dino.txt"
        with open(unique_list_path, 'w', encoding='utf-8') as f:
            #f.write(f"# DINO 기반 비디오 프레임 중복 제거 후 고유 이미지 목록\n")
            #f.write(f"# 생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            #f.write(f"# 모델: {self.model_name}\n")
            #f.write(f"# 유사도 임계값: {self.similarity_threshold}\n")
            #f.write(f"# 인접 프레임 윈도우 크기: {self.window_size}\n")
            #f.write(f"# 총 {len(unique_images)}개 프레임\n\n")
            
            for img_path in unique_images:
                frame_num = self.extract_frame_number(img_path.stem)
                #f.write(f"{img_path.name}\t# frame: {frame_num}\n")
                f.write(f"{img_path.name}\n")
        
        # 2. 중복 정보를 JSON 파일로 저장
        duplicate_info_path = self.folder_path / f"duplicate_frames_info.json"
        
        total_original_frames = len(unique_images) + sum(group['group_size'] - 1 for group in duplicate_groups)
        compression_ratio = len(unique_images) / total_original_frames if total_original_frames > 0 else 1.0
        
        duplicate_info = {
            'summary': {
                'total_unique_frames': len(unique_images),
                'total_original_frames': total_original_frames,
                'total_duplicate_groups': len(duplicate_groups),
                'total_duplicates_removed': sum(group['group_size'] - 1 for group in duplicate_groups),
                'compression_ratio': compression_ratio,
                'model_name': self.model_name,
                'similarity_threshold': self.similarity_threshold,
                'window_size': self.window_size,
                'device_used': self.device,
                'processed_at': datetime.now().isoformat()
            },
            'duplicate_groups': duplicate_groups
        }
        
        with open(duplicate_info_path, 'w', encoding='utf-8') as f:
            json.dump(duplicate_info, f, ensure_ascii=False, indent=2)
        
        # 3. 상세 보고서 생성
        report_path = self.folder_path / f"frame_deduplication_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("DINO 기반 비디오 프레임 중복 제거 보고서\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"처리 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"폴더 경로: {self.folder_path}\n")
            f.write(f"DINO 모델: {self.model_name}\n")
            f.write(f"사용 디바이스: {self.device}\n")
            f.write(f"유사도 임계값: {self.similarity_threshold}\n")
            f.write(f"처리 방식: 인접 프레임 순차 비교\n\n")
            
            f.write("요약\n")
            f.write("-" * 20 + "\n")
            f.write(f"원본 프레임 수: {total_original_frames}\n")
            f.write(f"고유 프레임 수: {len(unique_images)}\n")
            f.write(f"중복 그룹 수: {len(duplicate_groups)}\n")
            f.write(f"제거된 중복 프레임 수: {sum(group['group_size'] - 1 for group in duplicate_groups)}\n")
            f.write(f"압축율: {compression_ratio:.2%}\n\n")
            
            if duplicate_groups:
                f.write("중복 프레임 그룹 상세 정보\n")
                f.write("-" * 40 + "\n")
                for i, group in enumerate(duplicate_groups, 1):
                    f.write(f"\n그룹 {i} (연속 프레임 {group['frame_range']}, 총 {group['group_size']}개):\n")
                    f.write(f"  선택된 프레임: {group['selected']}\n")
                    f.write(f"  제거된 프레임들:\n")
                            
        return unique_list_path, duplicate_info_path, report_path

def main():
    parser = argparse.ArgumentParser(description='비디오 프레임 중복 제거 도구 (인접 프레임 비교)')
    parser.add_argument('folder_path', help='비디오 프레임 이미지가 있는 폴더 경로')
    parser.add_argument('--threshold', type=float, default=0.75,
                        help='유사도 임계값 (0.0-1.0, 기본값: 0.75)')
    parser.add_argument('--model', type=str, default='dino_vits16',
                        choices=['dino_vits16', 'dino_vits8', 'dino_vitb16', 'dino_vitb8'],
                        help='DINO 모델 선택 (기본값: dino_vits16)')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cuda', 'cpu'],
                        help='사용할 디바이스 (기본값: 자동선택)')
    parser.add_argument('--window-size', type=int, default=5,
                        help='인접 프레임 윈도우 크기 (기본값: 5)')
    
    args = parser.parse_args()
    
    # 폴더 존재 확인
    if not os.path.exists(args.folder_path):
        print(f"오류: 폴더를 찾을 수 없습니다: {args.folder_path}")
        return
    
    try:
        # 중복 제거 실행
        deduplicator = VideoFrameDeduplicator(
            args.folder_path, 
            args.threshold, 
            args.model,
            args.device,
            args.window_size
        )
        
        unique_images, duplicate_groups = deduplicator.process_images_sequential()
        
        if unique_images:
            # 결과 저장
            unique_list_path, duplicate_info_path, report_path = deduplicator.save_results(
                unique_images, duplicate_groups
            )
            
            total_original = len(unique_images) + sum(group['group_size'] - 1 for group in duplicate_groups)
            compression_ratio = len(unique_images) / total_original if total_original > 0 else 1.0
            
            print(f"\n🎬 비디오 프레임 중복 제거 완료!")
            print(f"원본 프레임 수: {total_original}")
            print(f"고유 프레임 수: {len(unique_images)}")
            print(f"제거된 중복 프레임 수: {sum(group['group_size'] - 1 for group in duplicate_groups)}")
            print(f"압축율: {compression_ratio:.2%}")
            print(f"비교 횟수: 약 {len(unique_images) + sum(group['group_size'] - 1 for group in duplicate_groups) - 1}번 (순차 비교)")
            print(f"\n생성된 파일:")
            print(f"  - 고유 프레임 목록: {unique_list_path}")
            print(f"  - 중복 정보: {duplicate_info_path}")
            print(f"  - 상세 보고서: {report_path}")
        else:
            print("처리할 수 있는 이미지가 없습니다.")
            
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        print("필요한 라이브러리가 설치되어 있는지 확인해주세요.")

if __name__ == "__main__":
    main()
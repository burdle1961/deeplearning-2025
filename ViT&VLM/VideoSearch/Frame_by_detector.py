from scenedetect import detect, ContentDetector, split_video_ffmpeg
import cv2
import os

def extract_scene_frames_new_api(video_path, threshold=27.0, output_dir="."):
    """
    PySceneDetect 최신 API를 사용한 장면 변화 감지 및 프레임 추출
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 장면 감지 (새로운 API)
    scene_list = detect(video_path, ContentDetector(threshold=threshold))
    
    print(f"감지된 장면 수: {len(scene_list)}")
    
    # 각 장면의 시작 프레임 추출
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    for i, (start_time, end_time) in enumerate(scene_list):
        # 시작 시간을 프레임 번호로 변환
        start_frame = int(start_time.get_frames())
        start_seconds = start_time.get_seconds()
        
        # 해당 프레임으로 이동
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, frame = cap.read()
        
        if ret:
            filename = f"{output_dir}/scene_{i+1:04d}_frame_{start_frame:06d}_{start_seconds:.2f}s.jpg"
            cv2.imwrite(filename, frame)
            print(f"저장됨: {filename}")
    
    cap.release()
    return scene_list

# 사용 예시
scene_list = extract_scene_frames_new_api("video2.mp4", threshold=25.0, output_dir='video2')
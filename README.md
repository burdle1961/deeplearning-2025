Flask
- 0.http_jpg : jpg 파일을 읽어 back-end로 요청하는 python client
- 0.http_jpg_bbox : jpg 파일을 읽어 back-end로 요청하고,  return 값을 이용하여 Bounding Box를 그려주는 cliengt(flask_server_json을 back-end로 사용)
- 1.flask_server : YOLO로 인식한 class 결과를 return
- 2.http_camera :  web camera의 frame을 back-end로 요청
- 3.flask_server_json : YOLO로 인식한 결과를 json 으로 return (class 및 bbox 정보 모두)
- 4.http_bbox : web camera의 인식결과를 받아 bbox 출력 (flask_server_json을 back-end로 사용)
- streamlibYOLO : streamlit 을 이용한 예제
Flask_front
- teamplate folder : jinja의 index.html과 같은 html template를 저장하는 폴더
- flaskFront.py : flask로 구성한 front-end 예제
- 3.flask_server_html.py : flask로 구성한 back-end 예제 (YOLO 인식 결과를 to_html() 메소드로 전송)
openCV
- 1.OpenCV_basic.py : openCV 기본 예제
- 2.YOLO_basic.py : YOLO 기본 예제
- 3.OpenCV+YOLO.py : openCV와 YOLO를 통합한 예제.
ViT&VLM
- DINO-search.py : DINO 를 이용한 이미지로 이미지 앨범 검색하기
- CLIP_by_image.py : CLIP을 이용하여 이미지에 적합한 설명 text 검색
- CLIP_by_text.py : CLIP을 이용하여 문장으로 이미지 검색하기
- ImageSearch.py : CLIP, ChromaDB, mongoDB를 이용한 간단한 이미지 검색 시스템 예제
- VideoSearch 폴더
    - Frame_by_detector.py : 동영상의 장면 전환을 검출하여 frame으로 출력
    - dedup_dino.py : DINO를 이용하여 장면전환 frame 중에서 유사한 이미지를 제거 (대표 이미지만 남김)
    - match_clip.py : 최종으로 남은 FRAME을 이용하여 검색 질문과의 유사도를 판별
YOLO
- 1.model_summary.py : YOLO model 구조 출력
- 2.model_train.py: YOLO train() 예제
- 3.model_predict.py : YOLO predict() 예제
- 4.type_of_source.py : mp4, webcamera 등을 이용하는 YOLO 예제 (다양한 이미지 입력 방법)

from ultralytics import YOLO

model = YOLO ('runs/detect/train/weights/best.pt')

result = model.predict(source='./test/images/', show=False, verbose=False)

# source에서 입력받은 이미지 갯수 만큼 결과가 ultralytics에서 정의한 object로 출력됨
print (len(result))

# 첫번재 이미지에 대한 결과
one = result[0]

# object 내의 속성을 출력 --> 실졔 predict의 결과를 사용하기 속성 이름 알아보기
print (one.__dict__.keys())
# dict_keys(['orig_img', 'orig_shape', 'boxes', 'masks', 'probs', 'keypoints', 'obb', 'speed', 'names', 'path', 'save_dir', '_keys'])

# orig_img :  입력 이미지 데이터 (2차원)
# orig_shape : 입력 이미지 사이즈 (640 x 640)
# boxes : predict로 검출된 객체의 bounding box (검출된 객체 수 만큼의 배열)
# names : 학습 시에 지정된 각 class 이름.
# 기타...

# boxes 속성이 가지고 있는 속성 이름 알라보기
print (one.boxes.__dict__.keys())
# dict_keys(['data', 'orig_shape', 'is_track'])

print (one.names)

# data 속성값 출력 : 실제 인식 결과
print (one.boxes.data.cpu().numpy())
# [[     196.48      252.85      397.24      400.53       0.984           0] <-- 0 번 class ()
#  [          0      489.67      185.67         640     0.98331           6]]

print (one.to_json())

print (one.to_html())
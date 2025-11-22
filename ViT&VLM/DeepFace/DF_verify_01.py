from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

folder = '/venv/deepface/images/'
sample = 'kke+asy.jpg'
target = 'asy+father.jpeg'

dir = os.listdir(folder)

rows = 1
cols = 2
    
def verify(img1_path,img2_path):

    img1= cv2.imread(folder + img1_path)
    img2= cv2.imread(folder + img2_path)

    h,w,c = img2.shape              # 대상 이미지 사이즈 가져오기

    output = DeepFace.verify(folder + img1_path, folder + img2_path, enforce_detection=False)
    bbox1 = output['facial_areas']['img1']
    bbox2 = output['facial_areas']['img2']

    if (bbox2['h'] == h and bbox2['w'] == w) :
        print ("Face Detect failed in  ", img2_path)
        return

    #print(output)
    verification = output['verified']

   
    if verification:
       print('They are same with distance = ', output['distance'])
    else:
       print('The are not same, distance = ', output['distance'])

    fig = plt.figure()

    ax1 = fig.add_subplot(rows, cols, 1)
    ax1.imshow(img1[:,:,::-1])
    rect = patches.Rectangle((bbox1['x'], bbox1['y']), bbox1['w'], bbox1['h'], linewidth=1, edgecolor='r', facecolor='none')
    ax1.add_patch(rect)
    ax1.set_title(sample)
    ax1.axis("off")

    ax2 = fig.add_subplot(rows, cols, 2)
    ax2.imshow(img2[:,:,::-1])
    rect = patches.Rectangle((bbox2['x'], bbox2['y']), bbox2['w'], bbox2['h'], linewidth=1, edgecolor='r',  facecolor='none')
    ax2.add_patch(rect)
    ax2.set_title(target)
    ax2.axis("off")
    
    plt.show()
    plt.close()


verify(sample,target)

from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt


def verify(img1_path,img2_path):
    img1= cv2.imread(img1_path)
    img2= cv2.imread(img2_path)
    
    plt.imshow(img1)
    plt.show()
    plt.imshow(img2)
    plt.show()
    output = DeepFace.verify(img1_path,img2_path)
    print(output)
    verification = output['verified']
    if verification:
       print('They are same')
    else:
       print('The are not same')


verify('img36.jpg','img37.jpg')

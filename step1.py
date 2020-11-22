import face_recognition
import cv2
import numpy as np

imgShi = face_recognition.load_image_file('image/Shi-Hailong.jpg')
imgShi = cv2.cvtColor(imgShi, cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('image/Shi-Hailong4.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)


cv2.imshow("imgShi", imgShi)
cv2.imshow("imgTest", imgTest)

cv2.imwrite("step1_Shi.jpg",imgShi)
cv2.imwrite("step1_imgTest.jpg",imgTest)

cv2.waitKey(0)



import face_recognition
import cv2
import numpy as np

imgElon = face_recognition.load_image_file('image/Elon-Musk.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('image/Elon-Test.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)


cv2.imshow("imgElon", imgElon)
cv2.imshow("imgTest", imgTest)
cv2.waitKey(0)



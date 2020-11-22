import face_recognition
import cv2
import numpy as np

imgShi = face_recognition.load_image_file('image/Shi-Hailong.jpg')
imgShi = cv2.cvtColor(imgShi, cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('image/Bill-Gates.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgShi)[0]
encodeShi = face_recognition.face_encodings(imgShi)[0]
cv2.rectangle(imgShi, (faceLoc[3],faceLoc[0]), (faceLoc[1],faceLoc[2]),(255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3],faceLocTest[0]), (faceLocTest[1],faceLocTest[2]),(255, 0, 255), 2)

results = face_recognition.compare_faces([encodeShi], encodeTest)
faceDis = face_recognition.face_distance([encodeShi], encodeTest)
cv2.putText(imgTest, f'{results} {round(faceDis[0],2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 3)

cv2.imshow("imgTest", imgTest)
cv2.imwrite("step3_false.jpg", imgTest)
cv2.waitKey(0)



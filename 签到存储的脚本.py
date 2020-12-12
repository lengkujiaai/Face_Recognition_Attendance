import face_recognition
import cv2
import numpy as np
import os
import cv2
from datetime import datetime

class Record:
    def __init__(self):
        path = 'ImagesAttendance'
        myList = os.listdir(path)
        self.images = []     # 存储所有图片的list
        self.className = []    # 存储对应的类名
        self.encodeListKnown = []

        for x,cl in enumerate(myList):
            curImg = cv2.imread(f'{path}/{cl}')
            self.images.append(curImg)
            self.className.append(os.path.splitext(cl)[0])

        deleted_idx = 'no'
        found = False
        deleted = False
        for item in self.className:
            if item == '.ipynb_checkpoints':
                deleted_idx = self.className.index(item)
                found = True
        if found == True and deleted == False:
            del(self.className[deleted_idx])
            del(self.images[deleted_idx])
            deleted = True
        print(self.className)
        self.findEncodings()

    def findEncodings(self):
        encodeList = []
        for img in self.images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        self.encodeListKnown =  encodeList

    def gstreamer_pipeline(self, capture_width=1080, capture_height=720, display_width=1080, display_height=720, framerate=30, flip_method=0,):
        return (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,
            )
        )

    def video_show(self):
        video = cv2.VideoCapture(self.gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
        while True:
            ret,frame = video.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(frame)
            encodesCurFrame = face_recognition.face_encodings(frame, facesCurFrame)
            if len(facesCurFrame) != 0:
                print("find face and process it")
                for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
                    matches = face_recognition.compare_faces(self.encodeListKnown, encodeFace)
                    faceDis = face_recognition.face_distance(self.encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)
                print(self.className[matchIndex])
                if matches[matchIndex]:
                    name = self.className[matchIndex].upper()
                    y1,x2,y2,x1=faceLoc
                    y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 1)
                if faceDis[matchIndex]< 0.50:
                    name = self.className[matchIndex].upper()
                else: name = 'Unknown'
                self.markAttendance(name)

            cv2.imshow("video",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video.release()
        cv2.destroyAllWindows()

    def markAttendance(self, name):
        with open('Attendance.csv','r+') as f:
            myDataList = f.readlines()
            nameList =[]
            for line in myDataList:
                entry = line.split(',')
                nameList.append(entry[0])
            if name not in  line:
                now = datetime.now()
                dt_string = now.strftime("%H:%M:%S")
                f.writelines(f'\n{name},{dt_string}')



if __name__ == '__main__':
    r = Record()
    #print(r.encodeList)
    r.video_show()


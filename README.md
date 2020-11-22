# Face-Recognition
jupyterlab notebook中用python和opencv实现刷脸签到

网络上有很多已经实现的基于opencv或其他算法的人脸识别代码，但多数是基于英文的。我选择一个并将其整理成了中文，代码也基于设备和不同功能有改动

# 已经在nvidia 的jetson nano上验证了代码的正确性

作者：lengkujiaai@126.com

目录及文件的介绍：

1、images是基础知识.ipynb里用到的图片，基础知识由3步组成

2、其中step1.py和step2.py是属于基础知识，和基础知识.ipynb内容基本一样，知识py文件可以用cv2.imshow()显示图片，而ipynb不能用该方法

3、其中step3_false.py和step3_true.py也属于基础知识

4、ImagesAttendance是签到项目用到的图片，Attendance.csv是签到项目存储图片名字的文件



# 用opencv实现人脸识别

我们将要学习如何利用opencv准确的识别人脸。首先，我们简单的学习一下理论和基本的实现；接着，我们会用网络摄像头探测并记录人脸后记录到excel表格中做为一个出席会议的项目。人脸识别在初学者和在计算机视觉领域经验丰富的人中是一个很流行的话题。因为人脸识别在众多的应用中都非常有用。

## 安装：
原文中推荐的安装是visual studio，还有cmake、dlib、fance_recognition、numpy、opencv-python。

本代码是在ubuntu上运行于jupyterlab下，需要安装一些依赖项。本代码运行的硬件环境是nvidia jetson nano

## 理解问题:
尽管opencv的人脸识别算法已经被开发出来好多年了，但是算法的运行速度和准确率一直不能很好的配合。不过，最近的一些改进露出了曙光。Facebook是一个很好的例子，他们只需要训练少数的图片就能够标记你和你的朋友，并且准确率高达98%。这是如何实现的呢，我们今天将使用Adam Geitgey开发的人脸识别库来复现类似的结果。让我们先看看他在论文（ https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78 ）中解释的四个问题。

## 人脸识别是若干问题组成的：

1、首先，查看一张图片并找到上面所有的人脸

2、第二，聚焦于不同的脸，即便这张脸是偏转的方向或不正常的光照，能识别出这是同一个人

3、第三，能够从脸上找到唯一的特征来区别于其他的脸----比如眼睛的大小，脸的长度等

4、最后，根据这张脸唯一的特征和你已知的所有人脸来确定其名字

参考英文链接：https://www.murtazahassan.com/face-recognition-opencv/

## 人脸识别：

首先，引入相关库

Import face_recognition

Import cv2

Import numpy as np

报错：no module named ‘face_recognition’

解决(请稍等31分钟，如果网络不好，可能要多等一会)：

sudo pip3 install face_recognition

因为要在notebook中运行并显示图片，这里选择用matplotlib，因此要安装一下：

sudo pip3 install matplotlib

识别可以分成三步。

# 第一步，加载图片并转换成RGB格式

人脸识别包中有个加载图片的函数，被引入的图片必须是RGB格式的。

imgShi = face_recognition.load_image_file('ImagesBasic/Shi-Hailong.jpg')

imgShi = cv2.cvtColor(imgShi,cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('ImagesBasic/Shi-Hailong4.jpg')

imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

原图：

![image](https://github.com/lengkujiaai/Face-Recognition/blob/main/readmeImages/step1_Shi.jpg)

测试图片：

![image](https://github.com/lengkujiaai/Face-Recognition/blob/main/readmeImages/step1_imgTest.jpg)

# 第二步，发现脸的位置并编码
在第二步，将使用人脸识别库的真正功能。首先发现图片中所有的人脸，这个通过在后端运行HOG实现。找到人脸后，变换成需要角度的图片。接着把这张图片送给预训练的神经网络，该网络会输出128个参数作为人脸的唯一标识。这些参数是模型在训练时自己学习的，所以我也不知道参数的具体含义。庆幸的是这些工作只需要两行代码。我们有了脸的位置信息和编码参数，就可以用矩形把脸框起来。

faceLoc = face_recognition.face_locations(imgShi)[0]

encodeShi = face_recognition.face_encodings(imgShi)[0]

cv2.rectangle(imgShi,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2) # top, right, bottom, left
 
faceLocTest = face_recognition.face_locations(imgTest)[0]

encodeTest = face_recognition.face_encodings(imgTest)[0]

cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

![image](https://github.com/lengkujiaai/Face-Recognition/blob/main/readmeImages/step2_Shi.jpg)

![image](https://github.com/lengkujiaai/Face-Recognition/blob/main/readmeImages/step2_Test.jpg)

# 第三步，人脸比较并发现差距
有个两张脸的编码参数后，就可以通过比较这两张脸的128个参数来发现相似性。用最常用的机器学习方法中的线性SVM分类器来比较这些参数。可以用compare_faces函数来计算人脸的相似性，该函数会返回True或False。同理，可以用face_distance函数来计算某张脸和其他脸的相似性。当有很多脸需要比较时，这个就很有用。

results = face_recognition.compare_faces([encodeShi], encodeTest)

faceDis = face_recognition.face_distance([encodeShi], encodeTest)

cv2.putText(imgTest,f'{results} {round(faceDis[0],2)} ',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),3)

如果运行测试图片，会得到返回值True,提示发现了人脸 Shi。比较人脸的差距是0.26，差距越小，相似度越高。

判断为正确的图片：

![image](https://github.com/lengkujiaai/Face-Recognition/blob/main/readmeImages/step3_true.jpg)

换成另外一张测试图片，这次是用的Bill Gates。可以看到结果是False，差距比之前高很多说明匹配的不好。

判断为错误的图片：

![image](https://github.com/lengkujiaai/Face-Recognition/blob/main/readmeImages/step3_false.jpg)

## 签到项目
我们用上面的方法开发一个签到系统，当摄像头探测到用户的脸时自动记录下来。这里会把对应人第一次出现的时间作为他的名字。

## 引入图片
像前面引入一样，使用face_recognition.load_image_file()函数引入我们需要的图片。当有很多图片时，引入他们会导致混乱。我们会写一个脚本从指定的文件夹中一次引入所有的图片。这里需要引入os库来操作文件夹。把所有的图片存到一个list中，把对应的名字存到另外一个list中。

import face_recognition

import cv2

import numpy as np

import os

path = 'ImagesAttendance'

images = []     # LIST CONTAINING ALL THE IMAGES

className = []    # LIST CONTAINING ALL THE CORRESPONDING CLASS Names

myList = os.listdir(path)

print("Total Classes Detected:",len(myList))

for x,cl in enumerate(myList):

        curImg = cv2.imread(f'{path}/{cl}')
        
        images.append(curImg)
        
        className.append(os.path.splitext(cl)[0])

## 计算编码
现在有一些图片在list中，可以把它们生成已知人名的编码。这里通过定义一个函数来实现该功能。像前面一样，先转换成RGB在转成对应的编码，最后添加到定义的list中。

def findEncodings(images):

    encodeList = []
    
    for img in images:
    
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        encode = face_recognition.face_encodings(img)[0]
        
        encodeList.append(encode)
        
    return encodeList

现在可以容易的调用这个函数，把图片list作为输入参数。

encodeListKnown = findEncodings(images)

print('Encodings Complete')

## While 循环

While训练是用来运行网络相机的。在运行while循环之前必须先创建捕获视频的对象，该对象用来从网络相机中抓取图片帧。

cap = cv2.VideoCapture(0)

## 网络相机图片
先读取图片并把它缩放到四分之一大小，这样是为了提升系统的速度。即便使用的是原图四分之一的大小，但显示的还是原图大小。再转换成RGB。

while True:

    success, img = cap.read()
    
    imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

## 网络相机编码
得到网络相机的图片后，接着找到图片上所有的脸，face_locations函数就是做这件事的。接着用face_encodings。

facesCurFrame = face_recognition.face_locations(imgS)

encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

## 匹配运算
现在可以通过把当前图片的编码和已知图片的编码进行比较来发现是谁。这里也会计算相似性，以防在一次比较中发现多个人，这样可以选择最相似的那个人。

for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):

    matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
    
    faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

一旦得到人脸差距的list后，就可以发现差距最小的那个，这个就是最佳匹配。

matchIndex = np.argmin(faceDis)

现在就可以根据索引值确定显示的图片对应的人名。

if matches[matchIndex]:

    name = className[matchIndex].upper()
    
    y1,x2,y2,x1=faceLoc
    
    y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
    
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
    
    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

# 标记出席者
下面添加自动出席的代码。先写一个唯一输入值是用户名字的函数。先打开csv格式的出席文件，接着读取所有的行并循环执行。接着用逗号分隔，这样就能得到第一个元素是用户的名字。如果用户已经存在于文件中，就什么都不发生。如果用户不在文件中，用户和对应是时间戳就存放到文件中。可以使用data time包中的datetime类得到当前时间。

def markAttendance(name):

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

# 把陌生脸也标记上
如果发现陌生的脸，就用下面的方法：

if matches[matchIndex]:

    name = classNames[matchIndex].upper()
    
    #print(name)
    
    y1,x2,y2,x1 = faceLoc
    
    y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
    
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
    
    cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
    
    cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    
    markAttendance(name)

if faceDis[matchIndex]< 0.50:

    name = classNames[matchIndex].upper()
    
    markAttendance(name)
    
else: name = 'Unknown'

#print(name)

y1,x2,y2,x1 = faceLoc

y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4

cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)

cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)


做这些是检查和对应脸的相似的是否小于0.5。如果大于0.5，意味着这个人是陌生人，并把名字标记成unknowm，并且标记出席

# 结论
opencv的人脸识别方法是面部识别中最简单和最快速的。


# 另：

1、2020-11-22，现供职于北京中电科卫星导航系统有限公司，本部门为研发中心。

2、公司在淘宝销售nvidia jetson 系列的产品，包括jetson nano，     TX1,     TX2,    AGX XAVIER,        XAVIER NX产品

3、我们属于提供技术支持的，本项目就是一老师要求的功能。

4、复制链接：   

    2.0fυィ直信息₰gyi7clU3sNj₤回t~bao或點几url链 https://m.tb.cn/h.4WAPC9j?sm=19844c 至浏览er【北京中电科卫星导航公司】
    
后打开淘宝即可

# 技术支持：

![image](https://github.com/lengkujiaai/Face-Recognition/blob/main/readmeImages/%E5%85%AC%E5%8F%B8%E4%BA%A7%E5%93%81.png)

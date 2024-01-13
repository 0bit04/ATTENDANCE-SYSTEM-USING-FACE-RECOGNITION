import numpy as np
import cv2
import face_recognition
import os
from datetime import datetime
from insert_data import insert_data
import pandas as pd

path = "Img"
names = []
className = []
mylist = os.listdir(path)
for cls in mylist:
    curImg = cv2.imread(f'{path}/{cls}')
    names.append(curImg)
    className.append(os.path.splitext(cls)[0])
# print(className)

def findEncodings(names):
    encodeL = []
    for img in names:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeL.append(encode)
    return encodeL

def markAtt(number):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nList = []
        for line in myDataList:
            entry = line.split(',')
            nList.append(entry[0])
            now = datetime.now()
            dtString = now.strftime('%H %M %S')
        if number in nList:
            for i, line in enumerate(myDataList):
                entry = line.split(',')
                if entry[0] == str(number):
                    myDataList[i] = f'{number},{dtString}\n'
                    break
            insert_data(number)
            f.seek(0)
            f.writelines(myDataList)

        if number not in nList:
            now = datetime.now()
            dtString = now.strftime('%H %M %S')
            f.writelines(f'\n{number},{dtString}')
            print(number)


encodeListKnown = findEncodings(names)
# print(encodeListKnown)

cap = cv2.VideoCapture(0)

while True:
    success,img = cap.read()
    imgs = cv2.resize(img,(0,0),None,0.25,0.25)
    imgs = cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)
    FaceCurF = face_recognition.face_locations(imgs)
    encodeCurF = face_recognition.face_encodings(imgs,FaceCurF)

    for encodeF , faceL in zip(encodeCurF,FaceCurF):
        matches = face_recognition.compare_faces(encodeListKnown,encodeF)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeF)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name1 = className[matchIndex].upper()
            y1,x2,y2,x1 = faceL
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name1,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAtt(name1)

    cv2.imshow('Webcam',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

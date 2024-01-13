import cv2
import face_recognition

imgRob = face_recognition.load_image_file('Img/robert_.jpg')
imgRob = cv2.cvtColor(imgRob,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('Img/test.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

FaceLoc = face_recognition.face_locations(imgRob)[0]
encode = face_recognition.face_encodings(imgRob)[0]
cv2.rectangle(imgRob,(FaceLoc[3],FaceLoc[0]),(FaceLoc[1],FaceLoc[2]),(255,0,0),2)

FaceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(FaceLocTest[3],FaceLocTest[0]),(FaceLocTest[1],FaceLocTest[2]),(255,0,0),2)

result = face_recognition.compare_faces([encode],encodeTest)
FacDis = face_recognition.face_distance([encode],encodeTest)
cv2.putText(imgTest,f'{result} {round(FacDis[0],3)}',(50,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,0),2)


cv2.imshow('Robert',imgRob)
cv2.imshow('Test',imgTest)
cv2.waitKey(0)
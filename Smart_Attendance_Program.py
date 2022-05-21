import cv2
import numpy as np
import face_recognition as face_rec
import os
from datetime import datetime
import pyttsx3 as textspeech  #imported libraries

engine = textspeech.init()

path = 'Students'   # specified the path where the images are store of the students
studentImg = []
studentName = []
myList = os.listdir(path)
#print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}\{cl}')#(student\Ayushi.jpg)
    studentImg.append(curImg)#add on curr img
    studentName.append(os.path.splitext(cl)[0])#to get rid of the .jpg at the end of the name ann add int in studentname list

#print(studentName)

#To find encoding of each and every image
def FindEncoding(images):
    imgEncoding = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeimg = face_rec.face_encodings(img)[0]
        imgEncoding.append(encodeimg)
    return imgEncoding

def MarkAttendance(name):
    with open('attendance.csv','r+') as f:# recongnise the student only once
        Datalist = f.readlines()
        nameList =[]
        for line in Datalist:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            timestr = now.strftime('%H: %M')
            date = now.strftime('%d-%B-%Y')
            f.writelines(f'\n{name}, {date}, {timestr}, Present')
            engine.say('welcome to class' + name)
            engine.runAndWait()

#to get the frame of a video
encode_list = FindEncoding(studentImg)
vid = cv2.VideoCapture(0)
while True:
    success, frame = vid.read()
    frames = cv2.resize(frame, (0,0), None, 0.25, 0.25)
   # frames = cv2.cvtcolor(frames, cv2.COLOR_BGR2RGB)
    faces_in_frame = face_rec.face_locations(frames)# to encode all the faces in the frame
    encode_Faces_in_frame = face_rec.face_encodings(frames, faces_in_frame)# to encode all the faces


    for encodeFace, faceloc  in zip(encode_Faces_in_frame, faces_in_frame):
        matches = face_rec.compare_faces(encode_list, encodeFace)
        facedis = face_rec.face_distance(encode_list, encodeFace)
        print(facedis)
        matchIndex = np.argmin(facedis)

        if matches[matchIndex]: #if matches are found to the image in student file we want the square around the with name
            name = studentName[matchIndex].upper()
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)
            cv2.rectangle(frame, (x1, y2-25), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
            MarkAttendance(name)

    cv2.imshow('video',frame)
    cv2.waitKey(1)




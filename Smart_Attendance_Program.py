import cv2
import numpy as np
import face_recognition as face_rec
import os
from datetime import datetime
import time
import pyttsx3 as textspeech  #imported libraries

engine = textspeech.init()

path = 'Students'  # specified the path where the images are store of the students
studentImg = []
studentName = []
#to get the frame of a video
encode_list = []
vid = cv2.VideoCapture(0)
def init():
    global studentName,studentImg,encode_list,vid
    studentImg = []
    studentName = []
    myList = os.listdir(path)

    for cl in myList:
        curImg = cv2.imread(f'{path}\{cl}')  # (student\Ayushi.jpg)
        studentImg.append(curImg)  # add on curr img
        studentName.append(os.path.splitext(cl)[0])  # to get rid of the .jpg at the end of the name ann add int in studentname list
    # to get the frame of a video
    encode_list = FindEncoding(studentImg)
    vid = cv2.VideoCapture(0)


#To find encoding of each and every image
def FindEncoding(images):
    imgEncoding = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeimg = face_rec.face_encodings(img)[0]
        imgEncoding.append(encodeimg)
    return imgEncoding



def register():
    name,frame = face_matching()
    if name != -1:
        print("Registration already done")
        return
    camera = cv2.VideoCapture(0)
    input('Press Enter to capture')
    return_value, image = camera.read()
    del camera
    with open('records.csv', 'r+') as a:
        Datalist1 = a.readlines()
        nameList1 = []
        for line in Datalist1:
            entry1 = line.split(',')
            nameList1.append(entry1[0])
        nam = input("Name: ")
        if nam not in nameList1:
            reg_number = input("Registration_number: ")
            id = input("Roll_Number: ")
            phone = input("Phone_number: ")
            cls = input("Standard: ")
            add = input("Address: ")
            a.writelines(f'\n{nam}, {reg_number}, {id}, {phone}, {cls}, {add}')
    pic = cv2.imwrite('Students/' + nam + '.jpg', image)
    print("Registration Completed")


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
            print("Attendance updated")



def attendance():
    name,frame = face_matching()
    if name != -1:
        MarkAttendance(name)

    cv2.imshow('video',frame)
    time.sleep(5)
    cv2.destroyAllWindows()

def face_matching():
    success, frame = vid.read()
    frames = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    faces_in_frame = face_rec.face_locations(frames)  # to encode all the faces in the frame
    encode_Faces_in_frame = face_rec.face_encodings(frames, faces_in_frame)  # to encode all the faces

    for encodeFace, faceloc in zip(encode_Faces_in_frame, faces_in_frame):
        matches = face_rec.compare_faces(encode_list, encodeFace)
        facedis = face_rec.face_distance(encode_list, encodeFace)
        matchIndex = np.argmin(facedis)

        if matches[matchIndex]:  # if matches are found to the image in student file we want the square around the with name
            name = studentName[matchIndex].upper()
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.rectangle(frame, (x1, y2 - 25), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            return name,frame

    return -1,-1


while True:
    init()
    ch = input("Press 0(attendance) or 1(registration):")
    if int(ch) == 0:
        attendance()
    else:
        register()


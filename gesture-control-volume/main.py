import cv2 
import mediapipe as mp
from math import hypot
from ctypes import cast, POINTER
from subprocess import call
import numpy as np 

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands 
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

valid = False

def calculate_volume(length):
    length = int(int((length*2)/5)/10)*10
    if length<0:
        return 0
    elif length>100:
        return 100
    else:
        return length

while True:
    success,img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    lmList = []
    if results.multi_hand_landmarks:
        for handlandmark in results.multi_hand_landmarks:
            for id,lm in enumerate(handlandmark.landmark):
                h,w,_ = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                lmList.append([id,cx,cy]) 
            mpDraw.draw_landmarks(img,handlandmark,mpHands.HAND_CONNECTIONS)
    if lmList != []:
        x1,y1 = lmList[4][1],lmList[4][2]
        x2,y2 = lmList[8][1],lmList[8][2]
        cv2.circle(img,(x1,y1),4,(255,0,0),cv2.FILLED)
        cv2.circle(img,(x2,y2),4,(255,0,0),cv2.FILLED)
        cv2.line(img,(x1,y1),(x2,y2),(255,0,0),3)
        length = hypot(x2-x1,y2-y1)
        volume=calculate_volume(length)
        print(length,volume)
        try:
            volume = int(volume)
            if (volume <= 100) and (volume >= 0):
                call(["amixer", "-D", "pulse", "sset", "Master", str(volume)+"%"])
                valid = True

        except ValueError:
            pass

        

    cv2.imshow('Image',img)
    if cv2.waitKey(1) & 0xff==ord('q'):
        break

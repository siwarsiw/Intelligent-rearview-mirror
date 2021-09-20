import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time


mixer.init()
sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')
mouth_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_mcs_mouth.xml')
if mouth_cascade.empty():
  raise IOError('Unable to load the mouth cascade classifier xml file')
  
lbl=['Close','Open']

model = load_model('models/cnncat2.h5')
model2 = load_model('models/yawnModel.h5')

path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]

mpred=[99]
ds_factor = 0.5

while(True):
    ret, frame = cap.read()
    cv2.imshow("frame",frame)
    frame2 = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)


    height,width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)
    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )


    mouth_rects = mouth_cascade.detectMultiScale(gray2, 1.7, 11)
    for (x,y,w,h) in mouth_rects:
    
        y = int(y - 0.15*h)
        mouth=frame2[y:y+h,x:x+w]
        mouth=cv2.cvtColor(mouth,cv2.COLOR_BGR2GRAY)
        mouth= cv2.resize(mouth, (60, 60))
        mouth= mouth/255
        mouth=mouth.reshape(-1,60,60)
        mouth = np.expand_dims(mouth,axis=0)
        mpred = model.predict_classes(mouth)
        
        if (mpred[0]==0):
            print('yawn')
        if (mpred[0]==1):
            print('normal')
        break
            
    c = cv2.waitKey(1)
    if c == 27:
        break
cap.release()
cv2.destroyAllWindows()

# -*- coding: utf-8 -*-
'''
@author: kalyan, fredrik
'''
import cv2
import os.path
from os import walk

# Training Cascaded Classifier 
#http://docs.opencv.org/3.1.0/dc/d88/tutorial_traincascade.html#gsc.tab=0

# Paths to pre-trained cascade specifications
#cascades_base_path = "/usr/local/share/OpenCV/haarcascades/"
cascades_path = "/usr/share/opencv/haarcascades/"

face_path = os.path.join(cascades_path, "haarcascade_frontalface_default.xml")
eye_path = os.path.join(cascades_path, "haarcascade_eye.xml")
smile_path = os.path.join(cascades_path, "haarcascade_smile.xml")
assert os.path.exists(face_path)
assert os.path.exists(eye_path)
assert os.path.exists(smile_path)

# Set up cascades
face_cascade  = cv2.CascadeClassifier(face_path)
eye_cascade   = cv2.CascadeClassifier(eye_path)
smile_cascade = cv2.CascadeClassifier(smile_path)

# Check input and output images
cwd      = os.path.dirname(os.path.realpath(__file__))
faces_dir = os.path.join(cwd, 'faces')
_,_, imageList = walk(faces_dir).next()

res_dir = os.path.join(cwd, 'results')
try:
        os.stat(res_dir)
except:
        os.mkdir(res_dir)

# Look over images and detect stuff
for imgName in imageList:
    
    srcName,extn = imgName.split('.')
    
    # Cascades were trained on gray scale data
    img = cv2.imread(os.path.join(faces_dir, imgName))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#    faces = face_cascade.detectMultiScale(gray, 1.03, 3)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop over faces
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        
        smiles = smile_cascade.detectMultiScale(roi_gray)
        for (sx,sy,sw,sh) in smiles:
            cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)
    
    dstName = srcName+'_face.'+extn
    cv2.imwrite(os.path.join(res_dir, dstName),img)

'''
Created on Mar 21, 2016

@author: kalyan
'''
import cv2

cap = cv2.VideoCapture('asterix02.mp4')

fourcc   = cv2.VideoWriter_fourcc('X','2','6','4')
width = 1024
height = 640
outVideo = cv2.VideoWriter('output.mp4',fourcc, 20.0, (width,height))
while(1):
    ret ,frame = cap.read()
    if ret == True:
        #frame = cv2.flip(frame,0)
        res = cv2.resize(frame,(width,height), interpolation = cv2.INTER_CUBIC)
        outVideo.write(res)

    else:
        break

cap.release()
outVideo.release()
cv2.destroyAllWindows()

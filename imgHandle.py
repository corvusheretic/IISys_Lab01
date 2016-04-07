'''
Created on Mar 21, 2016

@author: kalyan
'''
import sys
import numpy as np
import cv2

def rgb2grayFunc(fileName,
                 op_write='y'):
    img = cv2.imread(fileName)
    
    if(len(img.shape)==3):
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    
    if(op_write=='y'):
        cv2.imwrite('rgb2gray.png',img)
        
    print('rgb2grayFunc exit.')
    return img

def binaryFunc(fileName):
    
    img     = rgb2grayFunc(fileName,'n')
    _,thres = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    cv2.imwrite('binary.png',thres)
    print('binaryFunc exit.')
    return thres

# Use the link below for online help on geometric transforms 
# http://docs.opencv.org/3.1.0/da/d6e/tutorial_py_geometric_transformations.html#gsc.tab=0

def resizeFunc(fileName,
               scale=2):
    
    img     = rgb2grayFunc(fileName,'n')
    
    #res = cv2.resize(img,None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC) 
    # OR
    height, width = img.shape[:2]
    res = cv2.resize(img,(int(scale*width), int(scale*height)), interpolation = cv2.INTER_CUBIC)
    
    cv2.imwrite('resize.png',res)
    print('resizeFunc exit.')
    return res

def affineTransform(fileName):
    
    img     = rgb2grayFunc(fileName,'n')
    
    rows,cols = img.shape
    
    pts1 = np.float32([[50,50],[200,50],[50,200]])
    pts2 = np.float32([[10,100],[200,50],[100,250]])
    
    M = cv2.getAffineTransform(pts1,pts2)
    
    res = cv2.warpAffine(img,M,(cols,rows))
    
    cv2.imwrite('skew.png',res)
    print('affineTransform exit.')
    return res

if __name__ == '__main__':
    task     = sys.argv.pop()
    fileName = sys.argv.pop()
    
    if(task == 'c2g'):
        rgb2grayFunc(fileName)
    
    if(task == 'bw'):
        binaryFunc(fileName)
    
    if(task == 'resz'):
        resizeFunc(fileName,scale=1.5)
        
    if(task == 'skew'):
        affineTransform(fileName)
        
    print('Main exit.')

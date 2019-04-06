# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 16:22:23 2019

@author: Nicole_Xlp
"""
import cv2
from numpy import random
import numpy as np

img_color=cv2.imread('C:\\Users\\Nicole_Xlp\\Desktop\\image\\Nicole.jpg')
'''cv2.imshow('Nicole',img_color)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()
print(img_color)
print(img_color.dtype)
print(img_color.shape)'''
    
'''img_gray=cv2.imread('C:\\Users\\Nicole_Xlp\\Desktop\\image\\IMG20150425164612.jpg',0)
cv2.imshow('Nicole',img_gray)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()
print(img_gray)
print(img_gray.dtype)
print(img_gray.shape)'''

'''img_crop = img_color[0:224,0:224,:]
cv2.imshow('Nicole',img_crop)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()'''
    
'''img_resize = cv2.resize(img_color,(224,224))
cv2.imshow('Nicole',img_resize)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()
cv2.imwrite('Nicole.jpg',img_resize)'''

'''B,G,R = cv2.split(img_color)
b_rand = random.randint(-50,50)
if b_rand == 0:
    pass
elif b_rand > 0:
    lim = 255 - b_rand
    B[B>lim] = 255
    B[B<lim] = (B[B<lim] + b_rand).astype(img_color.dtype)
if b_rand < 0:
    lim = 0 - b_rand
    B[B>lim] = (B[B>lim] + b_rand).astype(img_color.dtype)
    B[B<lim] = 0
img_change_color = cv2.merge((B,G,R))
cv2.imshow('Nicole',img_color)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()
cv2.imshow('Nicole',img_change_color)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()'''
    
'''img_rotation = cv2.getRotationMatrix2D((img_color.shape[0]/2,img_color.shape[1]/2),30,9)
img_rotate = cv2.warpAffine(img_color, img_rotation, (img_color.shape[1], img_color.shape[0]))
cv2.imshow('Nicole',img_rotate)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()'''
    
'''def gamma_trans(img,gamma):
    table = [np.power(x / 255.0,gamma) * 255.0 for x in range(256)]
    table = np.array(table).astype("uint8")
    return cv2.LUT(img,table)
img_gamma = gamma_trans(img_color,3.4)
cv2.imshow('Nicole',img_gamma)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()'''
    
'''img_rotation = cv2.getRotationMatrix2D((img_color.shape[0]/2,img_color.shape[1]/2),30,9)
img_rotate = cv2.warpAffine(img_color, img_rotation, (img_color.shape[1], img_color.shape[0]))
cv2.imshow('Nicole',img_rotate)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()'''
    
'''rows, cols, ch = img_color.shape
pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.7, rows * 0.1], [cols * 0.1, rows * 0.9]]) 
M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img_color, M, (cols, rows))
cv2.imshow('Nicole', dst)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()'''
    
'''def random_warp(img):
    height, width, channels = img.shape
    random_margin = 70
    x1 = random.randint(-random_margin, random_margin)
    y1 = random.randint(-random_margin, random_margin)
    x2 = random.randint(width - random_margin - 1, width - 1)
    y2 = random.randint(-random_margin, random_margin)
    x3 = random.randint(width - random_margin - 1, width - 1)
    y3 = random.randint(height - random_margin - 1, height - 1)
    x4 = random.randint(-random_margin, random_margin)
    y4 = random.randint(height - random_margin - 1, height - 1)

    dx1 = random.randint(-random_margin, random_margin)
    dy1 = random.randint(-random_margin, random_margin)
    dx2 = random.randint(width - random_margin - 1, width - 1)
    dy2 = random.randint(-random_margin, random_margin)
    dx3 = random.randint(width - random_margin - 1, width - 1)
    dy3 = random.randint(height - random_margin - 1, height - 1)
    dx4 = random.randint(-random_margin, random_margin)
    dy4 = random.randint(height - random_margin - 1, height - 1)

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
    M_warp = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, M_warp, (width, height))
    return M_warp, img_warp
M_warp, img_warp = random_warp(img_color)
cv2.imshow('Nicole', img_warp)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()'''
	
    












    
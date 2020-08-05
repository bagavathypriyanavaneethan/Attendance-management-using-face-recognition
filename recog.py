# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 23:30:17 2019

@author: DELL
"""

import cv2

camera = cv2.VideoCapture(0)
i = 0
while i < 5000:
    print(r'Press Enter to capture')
    return_value, image = camera.read()
    cv2.imshow('img',image)
    cv2.imwrite(r'D:/Education/Project/Face recognition/Dataset/baggu'+str(i)+'.png', image)
    i += 1
    
#del(camera)
camera.release()
cv2.destroyAllWindows()
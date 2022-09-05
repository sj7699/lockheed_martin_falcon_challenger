import cv2 
import numpy as np
import pyzbar.pyzbar as pyzbar
image=cv2.imread('multiple_qr.png')
cv2.imshow('here',image)
qrs=pyzbar.decode(image)
print(len(qrs))
print(qrs[0].data.decode('utf-8'))
k="2*2-1*3+1-2"
print(eval(k))
cv2.waitKey(0)
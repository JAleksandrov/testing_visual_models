# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import imutils
import cv2

import imutils
import time
import os



hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


directory = '/root/human_detection/sample_images/'
directory = '/root/human_detection/negative_samples/'

img_count = 0

human_count = 0

below_013 = 0
below_030 = 0
below_070 = 0
higher_070 = 0

time_taken_analyse = time.time()


for image in os.listdir(directory):
    img_count = img_count+1
    imagePath = directory+image
    frame = cv2.imread(imagePath)
    bounding_box_cordinates, weights =  hog.detectMultiScale(frame, winStride = (4, 4), padding = (8, 8), scale = 1.03)
    for i, (x,y,w,h) in enumerate(bounding_box_cordinates):
        if weights[i] < 0.13:
            below_013 += 1
        elif weights[i] < 0.3 and weights[i] > 0.13:
            below_030 += 1
        if weights[i] < 0.7 and weights[i] > 0.3:
            below_070 += 1
        if weights[i] > 0.7:
            higher_070 += 1


        print(human_count)
        human_count += 1
           
print("Images Analysed: " + str(img_count))
print("Total humans detected: " + str(human_count))
print("Time taken to analyse %s seconds" % (time.time() - time_taken_analyse))
print("Per second images analysed: " + str((img_count / (time.time() - time_taken_analyse))))
print("Below 13%: " + str(below_013))
print("Between 13% and 30%: " + str(below_030))
print("Between 30% and 70%: " + str(below_070))
print("Higher 70%: " + str(higher_070))





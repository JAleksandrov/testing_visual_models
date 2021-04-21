import numpy as np
import argparse
import imutils
import time
import cv2
import os


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")



#directory = '/root/human_detection/sample_images/'
directory = '/root/human_detection/negative_samples/'
img_count = 0
human_count = 0
results = []
time_taken_analyse = time.time()
for image in os.listdir(directory):
    img_count += 1

    frame = cv2.imread(directory+image)
    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]
    size = (w, h)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if (int(detections[0, 0, i, 1])) == 15:
            if confidence > 0.3:
                human_count += 1
                print(confidence)
                results.append(confidence)
directory = '/root/human_detection/sample_images/'
for image in os.listdir(directory):
    img_count += 1

    frame = cv2.imread(directory+image)
    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]
    size = (w, h)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if (int(detections[0, 0, i, 1])) == 15:
            if confidence > 0.3:
                human_count += 1
                print(confidence)
                results.append(confidence)
for image in os.listdir(directory):
    img_count += 1

    frame = cv2.imread(directory+image)
    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]
    size = (w, h)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if (int(detections[0, 0, i, 1])) == 15:
            if confidence > 0.3:
                human_count += 1
                print(confidence)
                results.append(confidence)
directory = '/root/human_detection/sample_images/'
for image in os.listdir(directory):
    img_count += 1

    frame = cv2.imread(directory+image)
    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]
    size = (w, h)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if (int(detections[0, 0, i, 1])) == 15:
            if confidence > 0.3:
                human_count += 1
                print(confidence)
                results.append(confidence)
for image in os.listdir(directory):
    img_count += 1

    frame = cv2.imread(directory+image)
    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]
    size = (w, h)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if (int(detections[0, 0, i, 1])) == 15:
            if confidence > 0.3:
                human_count += 1
                print(confidence)
                results.append(confidence)
directory = '/root/human_detection/sample_images/'
for image in os.listdir(directory):
    img_count += 1

    frame = cv2.imread(directory+image)
    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]
    size = (w, h)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if (int(detections[0, 0, i, 1])) == 15:
            if confidence > 0.3:
                human_count += 1
                print(confidence)
                results.append(confidence)
for image in os.listdir(directory):
    img_count += 1

    frame = cv2.imread(directory+image)
    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]
    size = (w, h)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if (int(detections[0, 0, i, 1])) == 15:
            if confidence > 0.3:
                human_count += 1
                print(confidence)
                results.append(confidence)
directory = '/root/human_detection/sample_images/'
for image in os.listdir(directory):
    img_count += 1

    frame = cv2.imread(directory+image)
    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]
    size = (w, h)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if (int(detections[0, 0, i, 1])) == 15:
            if confidence > 0.3:
                human_count += 1
                print(confidence)
                results.append(confidence)
for image in os.listdir(directory):
    img_count += 1

    frame = cv2.imread(directory+image)
    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]
    size = (w, h)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if (int(detections[0, 0, i, 1])) == 15:
            if confidence > 0.3:
                human_count += 1
                print(confidence)
                results.append(confidence)
directory = '/root/human_detection/sample_images/'
for image in os.listdir(directory):
    img_count += 1

    frame = cv2.imread(directory+image)
    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]
    size = (w, h)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if (int(detections[0, 0, i, 1])) == 15:
            if confidence > 0.3:
                human_count += 1
                print(confidence)
                results.append(confidence)
for image in os.listdir(directory):
    img_count += 1

    frame = cv2.imread(directory+image)
    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]
    size = (w, h)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if (int(detections[0, 0, i, 1])) == 15:
            if confidence > 0.3:
                human_count += 1
                print(confidence)
                results.append(confidence)
directory = '/root/human_detection/sample_images/'
for image in os.listdir(directory):
    img_count += 1

    frame = cv2.imread(directory+image)
    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]
    size = (w, h)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if (int(detections[0, 0, i, 1])) == 15:
            if confidence > 0.3:
                human_count += 1
                print(confidence)
                results.append(confidence)

print("Images Analysed: " + str(img_count))
print("Total humans detected: " + str(human_count))
print("Time taken to analyse %s seconds" % (time.time() - time_taken_analyse))
print("Per second images analysed: " + str((img_count / (time.time() - time_taken_analyse))))
print("Average Result: " + str(np.mean(results)))




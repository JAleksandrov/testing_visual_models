import numpy as np
import time
import cv2
import os
import imutils


labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')


#directory = '/root/human_detection/sample_images/'
directory = '/root/human_detection/negative_samples/'
img_count = 0
human_count = 0
results = []
time_taken_analyse = time.time()

boxes = []
confidences = []
classIDs = []

for image in os.listdir(directory):
    print(image)
    img_count += 1

    image = cv2.imread(directory+image)
    (H, W) = image.shape[:2]
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)



    for output in layerOutputs:
	    for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if classID==0 and confidence > 0.1:
                    print(confidence)

                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
                    #results.append(confidence)
                    #human_count += 1
                    print(human_count)
                        

indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

for i in indices:
    i = i[0]
    results.append(confidences[i])
    human_count += 1


print("Images Analysed: " + str(img_count))
print("Total humans detected: " + str(human_count))
print("Time taken to analyse %s seconds" % (time.time() - time_taken_analyse))
print("Per second images analysed: " + str((img_count / (time.time() - time_taken_analyse))))
print("Average Result: " + str(np.mean(results)))

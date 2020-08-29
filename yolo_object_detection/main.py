import cv2
import numpy as np

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
'''Loading the Algo'''
with open("mohanad.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layerNames = net.getLayerNames()
outputLayers = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size = (len(classes), 3))

# Loading Img
img = cv2.imread("test.jpg")
img = cv2.resize(img, None, fx=0.4,fy=0.4) # fx-width fy-height
height,width,channels = img.shape

# Extract features (Detecting Objs )
blob = cv2.dnn.blobFromImage(img, 0.00392,(416, 416), (0, 0, 0), True, crop=False) # 416,416 the standard size
'''
for b in blob:
    for n,imgBlob in enumerate(b):
        cv2.imshow(str(n), imgBlob)
'''

net.setInput(blob)# pass blob in the algo
outs = net.forward(outputLayers) # we need to forward the output layer to get the result
#print(outs)

# Display borders of the detected objs  on the screen
confidences = []
class_ids = []
boxes = []


for out in outs:
    for detection in out:
        #Confidence
        scores = detection[5:]
        class_id = np.argmax(scores) # asscoiated with classes
        confidence = scores[class_id]
        if confidence > 0.5:
            centerX = int(detection[0] * width)
            centerY = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            #cv2.circle(img,(centerX, centerY), 10,(0,255,0),2) for testing locating the detected objs

            # Drawing Rect
            x = int(centerX - w /2)
            y = int(centerY - h /2)
            #cv2.rectangle(img, (x,y), (x+w,y+h),(0,255,0),2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

print(len(boxes))
objsDetected = len(boxes)
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4) # Non maximum suppresion and standard Threshold (for filtering the duplicates)
print(indexes)
font = cv2.FONT_HERSHEY_DUPLEX
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y+30), font, 1, color, 2)
cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
Blob it’s used to extract feature from the image and to resize them. YOLO accepts three sizes:

320×320 it’s small so less accuracy but better speed
609×609 it’s bigger so high accuracy and slow speed
416×416 it’s in the middle and you get a bit of both.
'''



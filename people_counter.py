from utils.centroidtracker import CentroidTracker
from utils.trackableobject import TrackableObject
from utils.nms import non_max_suppression
import numpy as np
import cv2 as cv
import sys

W = None
H = None

lines = list()
curLine = list()
drawMode = False
lastPosition = ()
def onMouseClick(event, x, y, flags, param):
    global curLine        
    global drawMode
    global lastPosition
    if event == cv.EVENT_LBUTTONUP:                
        drawMode = True
        if len(curLine) == 1:
            curLine.append((x,y))
            lines.append(Line(curLine[0], curLine[1]))
            curLine = list()
            drawMode = False
        else:
            curLine.append((x,y))
    elif event == cv.EVENT_RBUTTONUP:
        drawMode = False
        curLine = list()
    elif event == cv.EVENT_MOUSEMOVE:
        lastPosition = (x,y)

def drawLines(frame):     
    for line in lines:
        cv.line(frame,line.p1,line.p2,(255,0,0),5)
        cv.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    if len(curLine) == 1 and len(lastPosition) == 2:
        cv.line(frame,curLine[0],lastPosition,(255,0,0),5)                

class Line:
    def __init__(self, p1, p2 = None):
        self.p1 = p1
        self.p2 = p2

    def intersection(self, line):
        xdiff = (self.p1[0] - self.p2[0], line.p1[0] - line.p2[0])
        ydiff = (self.p1[1] - self.p2[1], line.p1[1] - line.p2[1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            return False

        d = (det(self.p1, self.p2), det(line.p1, line.p2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y

    def crossProduct(self, pt):
        return (self.p2[0] - self.p1[0]) * (pt[1] - self.p1[1]) - (self.p2[1] - self.p1[1])* (pt[0] - self.p1[0])
  

# load the COCO class labels our YOLO model was trained on
labelsPath = "./trained_model/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")
weightsPath = "./trained_model/yolov3.weights"
configPath = "./trained_model/yolov3.cfg"
net = cv.dnn.readNetFromDarknet(configPath, weightsPath)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
def getPeopleInFrame_YOLO(frame):
    W = len(frame[0])
    H = len(frame)
    bBoxLimit = (W,H,W,H)
    rects = []
    confidences = []
    blob = cv.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)           
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)

            if LABELS[classID] != "person":
                continue
		    
            confidence = scores[classID]
            if confidence > 0.6:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x1 = int(centerX - (width / 2))
                y1 = int(centerY - (height / 2))
                x2 = int(centerX + (width / 2))
                y2 = int(centerY + (height / 2))

                confidences.append(float(confidence))

                # Keep BoundingBoxes within range and covnert types into PythonTypes from openCV
                rects.append(tuple(map(lambda x:x.item(), np.clip((x1,y1,x2,y2), 0, bBoxLimit))))

    idxs = cv.dnn.NMSBoxes(rects, confidences, 0.4, 0.65)

    if len(idxs) > 0:
        rects = [rects[i] for i in idxs.flatten()]        
    return rects

# Init OpenCV Objects
cap = cv.VideoCapture('./videos/example_03.mp4')
#cap = cv.VideoCapture(0)
cv.namedWindow('frame')
cv.setMouseCallback('frame', onMouseClick)
mulTracker = None
trackers = []

cTracker = CentroidTracker(maxDisappeared=40, maxDistance=100)
objects = []
trackableObjects = {}
rects = []
skip_frames = 15
frame_counter = 0
out = None

while(True):
    e1 = cv.getTickCount()
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    if W is None or H is None:
        (H, W) = frame.shape[:2]
        out = cv.VideoWriter('./output/output.avi',cv.VideoWriter_fourcc('M','J','P','G'), 10, (W,H))

    r = 400 / float(W)
    dim = (400, int(H * r))
    frame = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
    rects = [] 




    if frame_counter % skip_frames == 0:
        mulTracker = cv.MultiTracker_create()

        rects = getPeopleInFrame_YOLO(frame)

        for (xA, yA, xB, yB) in rects:
            tracker = cv.TrackerKCF_create()
            mulTracker.add(tracker, frame, (xA, yA ,xB, yB))
    else:    
        success, bBoxes = mulTracker.update(frame)
        rects = bBoxes
                                
    objects = cTracker.update(rects)

    for (objectID, centroid) in objects.items():    
        to = trackableObjects.get(objectID, None)
    
        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)            
        else:
            if not to.counted:                
                l = Line(to.centroids[0], centroid)                    
                for line in lines:
                    intersection = l.intersection(line)
                    if intersection and min(l.p1[0], l.p2[0]) < intersection[0] < max(l.p1[0], l.p2[0]):
                        to.counted = True
                        print(f"WORKS - {objectID} - {line.crossProduct(centroid)}")
                        break
                
                to.centroids.append(centroid)

        trackableObjects[objectID] = to
    # Draws Tracking Data
    for (objectID, centroid) in objects.items():
        text = "ID {}".format(objectID)
        cv.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)        
        cv.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    for (xA, yA, xB, yB) in rects:
        cv.rectangle(frame, (int(xA), int(yA)), (int(xB),int(yB)), (0, 255, 0), 2)        

    # Display the resulting frame
    drawLines(frame)
    out.write(frame)
    cv.imshow('frame',frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
    e2 = cv.getTickCount()
    t = (e2 - e1)/cv.getTickFrequency()
    #print(t)
    frame_counter = (frame_counter + 1) % skip_frames

# When everything done, release the capture
cap.release()
out.release()
cv.destroyAllWindows()
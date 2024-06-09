import cv2

thres=0.5
cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

classNames=[]
classFile='coco.names'
with open(classFile,'rt') as f:
    classNames=f.read().rstrip('\n').split('\n')

configPath= 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath='frozen_inference_graph.pb'

net=cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)
while True:
    success,img=cap.read()
    classIds,confs,bbox=net.detect(img,confThreshold=thres)
    print(classIds,bbox)
    if len(classIds)!=0:
        for classId,confidence,box in zip(classIds.flatten(),confs.flatten(),bbox): 
            cv2.rectangle(img,box,color=(0,255,0),thickness=2)
            label = f"{classNames[classId-1].upper()} {confidence:.2f}"
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y1 = max(box[1], labelSize[1] + 10)
            cv2.rectangle(img, (box[0], y1 - labelSize[1] - 10), (box[0] + labelSize[0], y1 + baseLine - 10), (255, 255, 255), cv2.FILLED)
            cv2.putText(img, label, (box[0], y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imshow("Output",img)
    cv2.waitKey(1)

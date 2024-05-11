import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *


model=YOLO('yolov8s.pt')


area1=[(312,388),(289,390),(474,469),(497,462)]

area2=[(279,392),(250,397),(423,477),(454,469)]
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        #print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('input\shopping store.mp4')

# Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1020, 500))

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 

###############################################################################

print("About Project: ")
print("Unique Counts: A person enter or exit the region should be counted as 1 (in:1 or out:1)")
print("Same Counts: Same person enters and exit the region should be counted as 2 (in:1, out:1)")
print("Press 1: Unique Counts")
print("Press 2: Same Counts")
choice = int(input('Enter Your Choice : '))

####################################################################################

count=0
entering = set()
exiting = set()
people_entering = {}
people_exiting = {}
tracker = Tracker()
while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 2 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
    results=model.predict(frame)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
    list=[]
             
    for index,row in px.iterrows(): 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'person' in c:
           list.append([x1,y1,x2,y2])
    bbox_id = tracker.update(list)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox

        # Case 1: When People Entering
        results = cv2.pointPolygonTest(np.array(area2, np.int32),((x4,y4)),False)
        if results >=0:  
            people_entering[id] = (x4, y4)
            cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),2)

        if id in people_entering:
            results1 = cv2.pointPolygonTest(np.array(area1, np.int32),((x4,y4)),False)
            if results1 >=0:
                cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
                cv2.circle(frame, (x4, y4), 4, (255,0,255), -1)
                cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,(1),(0,0,255),1)
                entering.add(id)

        # Case 2: When People Exiting
        results2 = cv2.pointPolygonTest(np.array(area1, np.int32),((x4,y4)),False)
        if results2 >=0:  
            people_exiting[id] = (x4, y4)
            cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,255),2)

        if id in people_exiting:
            results3 = cv2.pointPolygonTest(np.array(area2, np.int32),((x4,y4)),False)
            if results3 >=0:
                cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,255),2)
                cv2.circle(frame, (x4, y4), 4, (255,0,255), -1)
                cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,(1),(0,0,255),1)
                exiting.add(id)
       
    cv2.polylines(frame,[np.array(area1,np.int32)],True,(255,0,0),2)
    cv2.putText(frame,str('1'),(504,471),cv2.FONT_HERSHEY_COMPLEX,(0.5),(0,0,0),1)

    cv2.polylines(frame,[np.array(area2,np.int32)],True,(255,0,0),2)
    cv2.putText(frame,str('2'),(466,485),cv2.FONT_HERSHEY_COMPLEX,(0.5),(0,0,0),1)

    if choice == 1:
        case_1 = "Unique Counts: A person enter or exit the region should be counted as 1 (in:1 or out:1)"
        cv2.putText(frame, case_1, (50, 80), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, "    In: " + str(len(entering)), (550, 380), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "   Out: " + str(len(exiting)), (550, 420), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Unique: " + str(len(entering)-len(exiting)), (550, 460), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 125, 0), 2, cv2.LINE_AA)

        cv2.imshow("RGB", frame)

        # Write the frame to the output video file
        #out.write(frame)
        if cv2.waitKey(1)&0xFF==27:
            break

    if choice == 2:
        case_2 = "Same Counts: Same person enters and exit the region should be counted as 2 (in:1, out:1)"
        cv2.putText(frame, case_2, (50, 80), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, "People: " + str(len(entering)+len(exiting)), (550, 420), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("RGB", frame)
        # Write the frame to the output video file
        #out.write(frame)
        if cv2.waitKey(1)&0xFF==27:
            break

cap.release()
cv2.destroyAllWindows()


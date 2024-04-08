import HandTrackingModule as HTM
import cv2 as cv
import time
import numpy as np
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


cap = cv.VideoCapture(0)
#Set dimension of webcam
wCam, hCam = 648, 488

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)

volume = interface.QueryInterface(IAudioEndpointVolume)
volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
# volume.SetMasterVolumeLevel(-20.0, None)
# print(volRange)
minVol = volRange[0]
maxVol = volRange[1]
vol=0
volBar=400
volPercent=0

cap.set(3,wCam)
cap.set(4,hCam)
pTime=0
detector = HTM.handDetector(detectionCon = 0.75)

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw=False)
    if len(lmlist)!=0:
        # print(lmlist)
        x1, y1 = lmlist[4][1], lmlist[4][2]
        x2, y2 = lmlist[8][1], lmlist[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2
        cv.circle(img, (x1,y1), 15, (255,0,255),cv.FILLED)
        cv.circle(img, (x2, y2), 15, (255, 0, 255), cv.FILLED)
        cv.line(img, (x1,y1), (x2,y2), (255,0,255), 2)

        length = math.hypot(x2-x1, y2-y1)
        print(length)

        #Hand range: 50 - 210
        #Vol range: -65 - 0
        vol = np.interp(length, [50,260], [minVol,maxVol])
        volBar = np.interp(length, [50, 260], [400, 150])
        volPercent = np.interp(length, [50, 260], [0, 100])
        volume.SetMasterVolumeLevel(vol, None)

        if length <= 50:
            cv.circle(img, (cx,cy), 15, (0,255,0),cv.FILLED)

    cv.rectangle(img, (50,150),(85,400), (0,255,0),2)
    cv.rectangle(img, (50, int(volBar)), (85, 400), (255,0,0), cv.FILLED)
    cv.putText(img, f"{int(volPercent)}%", (50,430), cv.FONT_HERSHEY_PLAIN, 1, (255,0,0),2)

    cTime=time.time()
    fps = 1/(cTime-pTime)
    pTime=cTime
    cv.putText(img, f"FPS:{int(fps)}", (10,30), cv.FONT_HERSHEY_PLAIN, 2, (255,0,0),2)
    cv.imshow("Volume Control", img)
    cv.waitKey(1)
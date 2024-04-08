import cv2 as cv
import time
import HandTrackingModule as htm
# import os

wCam, hCam = 648, 488

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime=0
cTime=0
path = "Finger Images"
myList = os.listdir(path)
overlayList = []
for imPath in myList:
    image = cv.imread(f"{path}/{imPath}")
    overlayList.append(image)

tipIds=[4,8,12,16,20]

detector = htm.handDetector()

while True:
    success, img = cap.read()
    # img[0:200, 0:200] = overlayList[0]
    img = detector.findHands(img, draw=True)
    lmlist = detector.findPosition(img, draw=False)
    if len(lmlist)!=0:
        fingers = []
        if lmlist[tipIds[id]][1] < lmlist[tipIds[id-2]][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1,5):
            if lmlist[tipIds[id]][2]> lmlist[tipIds[id-2]][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers = fingers.count(1)
        cv.putText(img, f"{totalFingers}", (45, 375), cv.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 20)

        w, h, c = overlayList[totalFingers-1].shape
        img[0:w, 0:h]= overlayList[totalFingers-1]


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv.putText(img, f"FPS:{int(fps)}", (10, 30), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv.imshow("Finger Counter", img)
    cv.waitKey(1)

import cv2 as cv
import mediapipe as mp
import time

class poseDetector():

    def __init__(self, mode=False, complex=1, smooth=True,enableSegment=False,
                 smoothSegment = True, detectionCon=0.5, trackingCon=0.5):
        self.mode = mode
        self.complex = complex
        self.smooth = smooth
        self.enableSegment = enableSegment
        self.smoothSegment = smoothSegment
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode,self.complex,self.smooth,
                                     self.enableSegment,self.smoothSegment,self.detectionCon,self.trackingCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self,img,draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self,img,draw=True):
        lmlist = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                w, h, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id,cx,cy])
                if draw:
                    cv.circle(img, (cx, cy), 5, (255, 0, 255), cv.FILLED)
        return lmlist

def main():
    cap = cv.VideoCapture(0)
    cTime = 0
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmlist = detector.findPosition(img)
        if len(lmlist)!=0:
            print(lmlist[14])
            cv.circle(img, (lmlist[14][1],lmlist[14][2]),15,(255,0,0),cv.FILLED)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv.imshow("Video", img)

        cv.waitKey(1)

if __name__ == "__main__":
    main()

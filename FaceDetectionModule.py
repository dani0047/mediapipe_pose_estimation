import cv2 as cv
import mediapipe as mp
import time

class faceDetector:

    def __init__(self, minDetectionCon = 0.5, model = 0):
        self.min_detection_confidence = minDetectionCon
        self.model_selection = model
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.min_detection_confidence)

    def findFaces(self,img, draw =True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxes =[]
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # mpDraw.draw_detection(img,detection)
                # print(detection.location_data.relative_bounding_box)
                bboxC = detection.location_data.relative_bounding_box
                w, h, c = img.shape
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                bboxes.append([id,bbox,detection.score])
                if draw:
                    img= self.fancyDraw(img,bbox)
                    cv.putText(img, str(f"{int(detection.score[0]*100)}%"), (bbox[0], bbox[1] - 20), cv.FONT_HERSHEY_PLAIN, 1,
                                (255, 0, 255), 2)

        return img, bboxes

    def fancyDraw(self,img,bbox,l=30,t=10):
        x, y, w, h = bbox
        x1, y1 = x+w, y+h
        cv.rectangle(img, bbox, (255,0,255), 2)
        #Top left
        cv.line(img,(x,y), (x+l, y), (255,0,255), t)
        cv.line(img,(x,y), (x, y+l), (255,0,255), t)
        #Top right
        cv.line(img,(x+w,y), (x+w-l, y), (255,0,255), t)
        cv.line(img,(x+w,y), (x+w, y+l), (255,0,255), t)
        #Bottom left
        cv.line(img,(x,y+h), (x+l, y+h), (255,0,255), t)
        cv.line(img,(x,y+h), (x, y+h-l), (255,0,255), t)
        #Bottom right
        cv.line(img,(x+w,y+h), (x+w-l, y+h), (255,0,255), t)
        cv.line(img,(x+w,y+h), (x+w, y+h-l), (255,0,255), t)


        return img


def main():
    cap = cv.VideoCapture(0)
    cTime = 0
    pTime = 0
    detector = faceDetector()
    while True:
        success, img = cap.read()
        img, bboxes = detector.findFaces(img)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(img, str(f"FPS:{int(fps)}"), (10, 70), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        cv.imshow("Face Detection", img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()
import cv2 as cv
import mediapipe as mp
import time


class FaceMeshDetector():

    def __init__(self, mode=False, maxFaces = 2,landmarks=False, detectionCon=0.5, trackingCon=0.5):
        self.static_image_mode = mode
        self.max_num_faces = maxFaces
        self.refine_landmarks = landmarks
        self.min_detection_confidence = detectionCon
        self.min_tracking_confidence = trackingCon

        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.static_image_mode,self.max_num_faces,
                                                 self.refine_landmarks,self.min_detection_confidence,self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(color=(0, 0, 255),thickness=1,circle_radius=2)


    def findFaceMesh(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    w, h, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # print(id, cx, cy)
                    face.append([id,cx,cy])

                faces.append(face)
        return img, faces

def main():
    cap = cv.VideoCapture(0)
    cTime = 0
    pTime = 0
    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()
        cTime = time.time()
        img, faces = detector.findFaceMesh(img)
        if len(faces)!=0:
            print(len(faces))
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(img, f"FPS:{int(fps)}", (10, 30), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        cv.imshow("Face Mesh", img)
        cv.waitKey(1)

if __name__ == "__main__":
    main()

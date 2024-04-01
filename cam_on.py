"""face detection using opencv caffe ssd model"""
import cv2
import numpy as np
import logging

logging.basicConfig(format='%(asctime)s | %(levelname)-8s | %(message)s', level=logging.INFO)
log = logging.getLogger(__name__)

PROTOTXT = "deploy.prototxt.txt"
CAFFEMODEL = "res10_300x300_ssd_iter_140000.caffemodel"
CONF_THRESHOLD = 0.7


class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("cannot open webcam")

        self.net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)
        self.face = None

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        self.face = None
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < CONF_THRESHOLD:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            self.face = frame[y1:y2, x1:x2]
            break  # take highest confidence only

        return frame

    def run(self):
        log.info("press q to quit")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                log.error("frame read failed")
                break
            cv2.imshow("Caffe SSD Face Detection", self.process_frame(frame))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    def release_resources(self):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        cam = Camera()
        cam.run()
    except IOError as e:
        log.error(e)
    finally:
        if "cam" in locals():
            cam.release_resources()

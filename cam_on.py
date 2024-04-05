"""switched to opencv yunet detector - way simpler than caffe blob preprocessing"""
import cv2
import numpy as np
import logging

logging.basicConfig(format='%(asctime)s | %(levelname)-8s | %(message)s', level=logging.INFO)
log = logging.getLogger(__name__)

YUNET_MODEL = "face_detection_yunet_2023mar.onnx"
CONF_THRESH = 0.7


class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("cannot open webcam")
        self.face = None
        self.face_box = None
        self.detector = None
        self._load_detector()

    def _load_detector(self):
        import os
        if not os.path.exists(YUNET_MODEL):
            log.warning(f"yunet model not found: {YUNET_MODEL}")
            return
        try:
            self.detector = cv2.FaceDetectorYN.create(YUNET_MODEL, "", (320, 240))
            log.info("YuNet detector loaded")
        except Exception as e:
            log.error(f"yunet load failed: {e}")

    def process_frame(self, frame):
        if self.detector is None:
            return frame
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, dets = self.detector.detect(frame)
        self.face = None
        self.face_box = None
        if dets is None or len(dets) == 0:
            return frame
        best = max(dets, key=lambda d: d[-1])
        if best[-1] < CONF_THRESH:
            return frame
        x, y, bw, bh = [int(v) for v in best[:4]]
        x2, y2 = min(w, x + bw), min(h, y + bh)
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{best[-1]*100:.1f}%", (x, y - 8 if y > 20 else y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        self.face = frame[y:y2, x:x2]
        self.face_box = best
        return frame

    def run(self):
        log.info("press q to quit")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                log.error("frame read failed")
                break
            cv2.imshow("Video", self.process_frame(frame))
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

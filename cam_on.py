"""trying facenet for face recognition - tf graph loading is messy"""
import cv2
import numpy as np
import logging
import os

logging.basicConfig(format='%(asctime)s | %(levelname)-8s | %(message)s', level=logging.INFO)
log = logging.getLogger(__name__)

PROTOTXT = "deploy.prototxt.txt"
CAFFEMODEL = "res10_300x300_ssd_iter_140000.caffemodel"
CONF_THRESH = 0.7
FACENET_MODEL = "facenet_model.pb"


class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("cannot open webcam")
        self.net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)
        self.face = None
        self.facenet = None
        self._load_facenet()

    def _load_facenet(self):
        if not os.path.exists(FACENET_MODEL):
            log.warning(f"facenet pb not found: {FACENET_MODEL}")
            return
        try:
            # cv2.dnn.readNetFromTensorflow chokes on the frozen pb
            # tried tf.compat.v1.Session + GraphDef too but all the node names
            # from tools/freeze_graph.py are internal and not documented clearly
            # leaving this for now, looking for alternatives
            self.facenet = cv2.dnn.readNetFromTensorflow(FACENET_MODEL)
            log.info("facenet loaded (output layer still unknown)")
        except cv2.error as e:
            # this keeps failing with "FAILED: ReadProtoFromBinaryFile"
            # the .pb from davidsandberg/facenet is TF1 format, opencv cant read it
            log.error(f"opencv cannot parse frozen pb: {e}")
        except Exception as e:
            log.error(f"facenet load error: {e}")

    def get_embedding(self, face_img):
        if self.facenet is None:
            return None
        resized = cv2.resize(face_img, (160, 160))
        blob = cv2.dnn.blobFromImage(resized, 1.0/128.0, (160, 160), (127.5, 127.5, 127.5))
        self.facenet.setInput(blob)
        # tried: 'embeddings', 'InceptionResnetV1/Logits/AvgPool_1a_8x8/AvgPool', 'output'
        # all throw cv2.error: Unknown layer type: Switch
        return self.facenet.forward().flatten()

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        dets = self.net.forward()
        self.face = None
        for i in range(dets.shape[2]):
            conf = dets[0, 0, i, 2]
            if conf < CONF_THRESH:
                continue
            box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            self.face = frame[y1:y2, x1:x2]
            break
        return frame

    def run(self):
        log.info("press q to quit")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                log.error("frame read failed")
                break
            cv2.imshow("FaceNet Test", self.process_frame(frame))
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

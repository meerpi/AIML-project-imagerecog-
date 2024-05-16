# Model Download Instructions

The ONNX model files are not included in the repository (they are binary and large).
Download them manually and place them in the project root directory.

## YuNet Face Detector

File: `face_detection_yunet_2023mar.onnx`

```bash
wget https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

## SFace Face Recognizer

File: `face_recognizer_fast.onnx`

```bash
wget https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognizer_fast.onnx
```

Both models are from the [OpenCV Model Zoo](https://github.com/opencv/opencv_zoo).

## After Download

Verify both files are in the project root:

```
face_detection_yunet_2023mar.onnx   (~380 KB)
face_recognizer_fast.onnx           (~38 MB)
```

Then run the app normally.

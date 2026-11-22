import cv2
import numpy as np
import time
from picamera2 import Picamera2

class VisionSubsystem:
    def __init__(self, onnx_path="yolo11n.onnx", names_path="coco.names"):
        with open(names_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
            
        self.net = cv2.dnn.readNetFromONNX(onnx_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        self.input_size = 640
        self.conf_threshold = 0.5
        self.nms_threshold = 0.4

    def _process_tensor(self, frame):
        image_height, image_width = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (self.input_size, self.input_size), swapRB=True, crop=False)
        self.net.setInput(blob)
        preds = self.net.forward()

        outputs = cv2.transpose(preds[0])
        boxes_raw = outputs[:, :4]
        scores_raw = outputs[:, 4:]

        class_ids = np.argmax(scores_raw, axis=1)
        scores = np.max(scores_raw, axis=1)
        mask = scores > self.conf_threshold

        boxes_raw = boxes_raw[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        boxes = []
        x_factor = image_width / self.input_size
        y_factor = image_height / self.input_size

        for box in boxes_raw:
            x, y, w, h = box
            left = int((x - w / 2) * x_factor)
            top = int((y - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            boxes.append([left, top, width, height])

        indices = cv2.dnn.NMSBoxes(boxes, scores.tolist(), self.conf_threshold, self.nms_threshold)

        last_class_ids = []
        if len(indices) > 0:
            for i in indices.flatten():
                last_class_ids.append(class_ids[i])

        detected_names = set([self.classes[cls_id] for cls_id in last_class_ids])
        
        if detected_names:
            return ", ".join(detected_names)
        else:
            return "nothing"

    def detect_objects_webcam(self):
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return "error camera unavailable"

            for _ in range(15):
                cap.read()

            success, frame = cap.read()
            cap.release()

            if not success:
                return "error frame capture failed"

            return self._process_tensor(frame)
        except Exception:
            return "error webcam initialization failed"

    def detect_objects(self):
        try:
            picam2 = Picamera2()
            config = picam2.create_preview_configuration(main={"format": "BGR888", "size": (640, 480)})
            picam2.configure(config)
            picam2.start()
            
            time.sleep(2)
            
            frame = picam2.capture_array()
            
            picam2.stop()
            picam2.close()
            
            return self._process_tensor(frame)
        except Exception:
            return "error picamera initialization failed"

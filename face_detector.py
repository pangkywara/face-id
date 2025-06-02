"""
Real-time face detection, anti-spoofing, and recognition pipeline.

Based on processing steps from:
- https://github.com/biubug6/Pytorch_Retinaface (for RetinaFace detection)
- Integrates FaceRecognizer and AntiSpoofing modules.
"""

import cv2
import time
import argparse
import numpy as np
# from PIL import Image # PIL Image not directly used in the class version, cv2 handles images
from itertools import product
from math import ceil
import os
import threading # Used by CameraManager, FaceRecognizer, AntiSpoofing indirectly
# from queue import Queue, Empty # Used by old WebcamStream, not directly by FaceDetector class

# --- Project-specific imports ---
from camera_manager import CameraManager
from face_recognizer import FaceRecognizer
from anti_spoofing import AntiSpoofing
from utils import logger # Assuming utils.py has a configured logger

# --- Post-processing Utilities & Config (Module Level) ---
cfg_mnet = {
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'image_size': 480 # Default, detector input size can be different
}

class PriorBoxNumpy(object):
    def __init__(self, cfg, image_size):
        super(PriorBoxNumpy, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        # image_size is expected to be (height, width)
        self.image_size_h_w = image_size 
        self.feature_maps = [[ceil(self.image_size_h_w[0]/step), ceil(self.image_size_h_w[1]/step)] for step in self.steps]

    def forward(self):
        anchors = []
        for k, f_map_h_w in enumerate(self.feature_maps):
            min_sizes_for_feature_map = self.min_sizes[k]
            for i, j in product(range(f_map_h_w[0]), range(f_map_h_w[1])): # i for height, j for width
                for min_size in min_sizes_for_feature_map:
                    # s_kx and s_ky are normalized by image width and height respectively
                    s_kx = min_size / self.image_size_h_w[1] # Normalize by width
                    s_ky = min_size / self.image_size_h_w[0] # Normalize by height
                    
                    dense_cx = [(x + 0.5) * self.steps[k] / self.image_size_h_w[1] for x in [j]]
                    dense_cy = [(y + 0.5) * self.steps[k] / self.image_size_h_w[0] for y in [i]]
                    
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
        output = np.array(anchors).reshape(-1, 4)
        if self.clip:
            output = np.clip(output, 0, 1)
        return output

def decode_boxes_numpy(loc, priors, variances):
    boxes = np.concatenate((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def decode_landm_numpy(pre, priors, variances): # Landmarks not actively used in pipeline yet
    landms = np.concatenate((
        priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
        ), axis=1)
    return landms

def py_cpu_nms(dets, thresh):
    if dets.shape[0] == 0:
        return []
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

# Global tflite module to be populated by __main__ or class instance
# This is a placeholder; the class will handle its own tflite import.
# _tflite_runtime = None 

class FaceDetector:
    def __init__(self, detector_model_path,
                 conf_threshold=0.7, nms_threshold=0.4, detector_num_threads=None,
                 use_full_tf=False, camera_index=0, camera_resolution=(640,480),
                 detector_input_size=480):
        
        self.detector_model_path = detector_model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.detector_num_threads = detector_num_threads
        self.use_full_tf = use_full_tf
        self.camera_index = camera_index
        self.camera_resolution = camera_resolution
        self.detector_input_size = detector_input_size # Expected input size (e.g., 480 for 480x480 model)

        self._tflite = None # To be initialized based on use_full_tf
        self._import_tflite()

        # Output tensor indices for RetinaFace (verify if using a different model version)
        self.LOC_INDEX = 568
        self.CONF_INDEX = 583
        self.LANDMS_INDEX = 597

        self._load_detector_model()
        self._initialize_modules()
        
        self.frame_count = 0
        self.fps = 0
        self.fps_start_time = time.time()
        self.detector_inference_time_ms = 0 # Initialize timing variables

    def _import_tflite(self):
        if self.use_full_tf:
            logger.info("FaceDetector: Using full TensorFlow library for TFLite.")
            import tensorflow as tf
            self._tflite = tf.lite
        else:
            try:
                logger.info("FaceDetector: Using tflite_runtime library.")
                import tflite_runtime.interpreter as interpreter_module
                self._tflite = interpreter_module # The module itself, Interpreter is called as self._tflite.Interpreter
            except ImportError:
                logger.error("FaceDetector: tflite_runtime not found. Install it or use full TensorFlow.")
                raise

    def _load_detector_model(self):
        if not os.path.exists(self.detector_model_path):
            logger.error(f"FaceDetector Error: Model file not found at {self.detector_model_path}")
            raise FileNotFoundError(f"Detector model not found: {self.detector_model_path}")

        logger.info(f"FaceDetector: Loading model: {self.detector_model_path}")
        try:
            self.detector_interpreter = self._tflite.Interpreter(
                model_path=self.detector_model_path, 
                num_threads=self.detector_num_threads
            )
        except Exception as e:
            logger.error(f"FaceDetector: Error loading TFLite model: {e}")
            raise
        
        input_details = self.detector_interpreter.get_input_details()
        self.detector_input_index = input_details[0]['index']
        
        # NCHW format for many RetinaFace TFLite models
        desired_shape = [1, 3, self.detector_input_size, self.detector_input_size]
        logger.info(f"FaceDetector: Resizing input tensor {self.detector_input_index} to shape: {desired_shape}")
        try:
            self.detector_interpreter.resize_tensor_input(self.detector_input_index, desired_shape)
            self.detector_interpreter.allocate_tensors()
        except Exception as e:
            logger.error(f"FaceDetector: Error resizing/allocating input tensor: {e}")
            raise

        # Get details again after resize
        self.detector_input_details = self.detector_interpreter.get_input_details()
        # For NCHW, shape is [batch, channels, height, width]
        self.detector_input_height = self.detector_input_details[0]['shape'][2]
        self.detector_input_width = self.detector_input_details[0]['shape'][3]
        self.is_detector_floating_model = self.detector_input_details[0]['dtype'] == np.float32
        logger.info(f"FaceDetector: Input details AFTER resize: {self.detector_input_details}")
        logger.info(f"FaceDetector: Model expects float32 input: {self.is_detector_floating_model}")

        logger.info("FaceDetector: Generating prior boxes...")
        # Pass (height, width) to PriorBoxNumpy
        priorbox_gen = PriorBoxNumpy(cfg=cfg_mnet, image_size=(self.detector_input_height, self.detector_input_width))
        self.priors = priorbox_gen.forward()
        logger.info(f"FaceDetector: Generated {self.priors.shape[0]} prior boxes.")
        
        # Verify output indices exist
        output_details = self.detector_interpreter.get_output_details()
        output_indices_from_model = {detail['index'] for detail in output_details}
        required_indices = {self.LOC_INDEX, self.CONF_INDEX, self.LANDMS_INDEX}
        if not required_indices.issubset(output_indices_from_model):
            logger.error(f"FaceDetector Error: One or more specified output indices ({required_indices}) not found in the model!")
            logger.error(f"Available output details: {output_details}")
            # This is a critical error, consider raising an exception or specific handling
            # For now, it will likely fail later when get_tensor is called.
            # raise ValueError("Detector model output indices mismatch")
        else:
            logger.info(f"FaceDetector: Using output indices - Loc: {self.LOC_INDEX}, Conf: {self.CONF_INDEX}, Landms: {self.LANDMS_INDEX}")


    def _initialize_modules(self):
        logger.info("FaceDetector: Initializing Face Recognizer...")
        self.face_recognizer = FaceRecognizer()
        logger.info("FaceDetector: Initializing Anti-Spoofing module...")
        self.anti_spoofing = AntiSpoofing()
        logger.info("FaceDetector: Initializing CameraManager...")
        self.cam_manager = CameraManager(camera_index=self.camera_index, resolution=self.camera_resolution)

    def start_pipeline(self):
        """Starts the camera and the main processing loop. Displays output in a window."""
        self.cam_manager.start()
        logger.info("FaceDetector: Waiting for camera to be ready...")
        while not self.cam_manager.is_ready():
            if hasattr(self.cam_manager, 'placeholder_frame'): # Display placeholder if available
                placeholder = self.cam_manager.placeholder_frame.copy()
                cv2.putText(placeholder, "Initializing, please wait...", 
                            (50, self.camera_resolution[1] // 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                cv2.imshow('Face Detection Pipeline', placeholder)
                cv2.waitKey(100) # ms
            else:
                time.sleep(0.1)
        logger.info("FaceDetector: Camera is ready. Starting pipeline loop.")

        try:
            while True:
                frame = self.cam_manager.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                processed_frame, detections = self.process_single_frame(frame)
                
                cv2.imshow('Face Detection Pipeline', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.stop_pipeline()

    def process_single_frame(self, frame):
        """Processes a single frame for face detection, anti-spoofing, and recognition."""
        img_height, img_width, _ = frame.shape
        
        # --- Preprocessing for Face Detector (RetinaFace) ---
        # RetinaFace often expects BGR input with specific BGR mean subtraction.
        img_resized_detector_bgr = cv2.resize(frame, (self.detector_input_width, self.detector_input_height))
        input_data_detector = np.expand_dims(img_resized_detector_bgr, axis=0)
        input_data_detector = input_data_detector.transpose((0, 3, 1, 2))  # HWC to NCHW
        input_data_detector = input_data_detector.astype(np.float32)
        
        # BGR channel means for NCHW format (104, 117, 123)
        channel_means = np.array([[[[104]], [[117]], [[123]]]], dtype=np.float32)
        if self.is_detector_floating_model:
            input_data_detector -= channel_means
        
        self.detector_interpreter.set_tensor(self.detector_input_index, input_data_detector)

        # --- Face Detection Inference ---
        inference_start_time = time.time()
        self.detector_interpreter.invoke()
        self.detector_inference_time_ms = (time.time() - inference_start_time) * 1000

        # --- Post-processing Detections ---
        try:
            loc = np.squeeze(self.detector_interpreter.get_tensor(self.LOC_INDEX))
            conf = np.squeeze(self.detector_interpreter.get_tensor(self.CONF_INDEX))
            # landms_raw = np.squeeze(self.detector_interpreter.get_tensor(self.LANDMS_INDEX))
        except Exception as e:
            logger.error(f"FaceDetector: Error getting output tensors: {e}")
            # Draw error on frame and return
            cv2.putText(frame, "DETECTOR OUTPUT ERROR", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame, []

        boxes = decode_boxes_numpy(loc, self.priors, cfg_mnet['variance'])
        boxes = boxes * np.array([img_width, img_height, img_width, img_height])
        scores = conf[:, 1]

        valid_idx = np.where(scores > self.conf_threshold)[0]
        boxes = boxes[valid_idx]
        scores = scores[valid_idx]
        # landms_decoded_scaled = landms_decoded_scaled[valid_idx] # if landmarks are processed

        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep_indices = py_cpu_nms(dets, self.nms_threshold)
        dets = dets[keep_indices, :]
        # landms_filtered = landms_decoded_scaled[keep_indices] # if landmarks are processed

        detected_faces_info = []

        for i, det_box in enumerate(dets):
            x1, y1, x2, y2 = map(int, det_box[0:4])
            detection_score = det_box[4]

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_width - 1, x2)
            y2 = min(img_height - 1, y2)

            if x1 >= x2 or y1 >= y2:
                continue

            cropped_face = frame[y1:y2, x1:x2]
            if cropped_face.size == 0:
                continue

            is_real, spoof_type = self.anti_spoofing.check(cropped_face)
            
            recognized_name = "Unknown"
            recognition_confidence = 0.0
            if is_real:
                recognized_name, recognition_confidence = self.face_recognizer.recognize(cropped_face)
                if recognized_name is None: recognized_name = "Unknown"
            
            detected_faces_info.append({
                'box': (x1, y1, x2, y2),
                'detection_score': detection_score,
                'is_real': is_real,
                'spoof_type': spoof_type,
                'recognized_name': recognized_name,
                'recognition_confidence': recognition_confidence
            })

            # --- Drawing Results ---
            box_color = (0,0,255) # Default Red (spoof or unknown)
            if is_real:
                if recognized_name != "Unknown" and recognition_confidence > 0.5: # Adjust threshold as needed
                    box_color = (0, 255, 0) # Green for known, real
                else:
                    box_color = (0, 255, 255) # Yellow for unknown, real

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            label_y_offset = y1 - 7
            text_info = []
            if is_real:
                text_info.append(f"ID: {recognized_name} ({recognition_confidence:.2f})")
            text_info.append(f"Spoof: {spoof_type} ({detection_score:.2f})")
            
            for i, text_line in enumerate(reversed(text_info)):
                cv2.putText(frame, text_line, (x1, label_y_offset - (i*18)), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, box_color, 1)

        # --- FPS Calculation & Display ---
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.fps_start_time
        if elapsed_time >= 1.0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.fps_start_time = current_time

        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Detect Infer: {self.detector_inference_time_ms:.1f} ms", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame, detected_faces_info

    def stop_pipeline(self):
        logger.info("FaceDetector: Stopping camera stream and closing windows...")
        if self.cam_manager:
            self.cam_manager.stop()
        cv2.destroyAllWindows()
        # CameraManager.release_all() # This is a class method, call if appropriate for app lifecycle
        logger.info("FaceDetector: Cleanup complete.")

    def __enter__(self):
        self.cam_manager.start()
        logger.info("FaceDetector: Waiting for camera to be ready (context manager)...")
        while not self.cam_manager.is_ready(): time.sleep(0.1)
        logger.info("FaceDetector: Camera is ready (context manager).")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_pipeline()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Real-time face detection, anti-spoofing, and recognition pipeline.")
    parser.add_argument(
        '-m', '--model_path',
        default='models/480-float16.tflite',
        help='Path to the .tflite face detector model file (e.g., RetinaFace variant)')
    parser.add_argument(
        '--conf_thresh', type=float, default=0.7,
        help='Confidence threshold for filtering detections')
    parser.add_argument(
        '--nms_thresh', type=float, default=0.4,
        help='Non-Maximum Suppression (NMS) threshold')
    parser.add_argument(
        '--num_threads', type=int, default=None,
        help='Number of threads for TFLite detector interpreter')
    parser.add_argument(
        '--use_full_tf', action='store_true',
        help='Use full TensorFlow library instead of tflite_runtime')
    parser.add_argument(
        '--cam_idx', type=int, default=0, help='Camera index to use.')
    parser.add_argument(
        '--cam_res_w', type=int, default=640, help='Camera resolution width.')
    parser.add_argument(
        '--cam_res_h', type=int, default=480, help='Camera resolution height.')
    parser.add_argument(
        '--detector_input_size', type=int, default=480, help='Input size (square) for the detector model (e.g., 480 for 480x480).')

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        logger.error(f"Specified model path {args.model_path} not found. Please check the path.")
        # Try to find it in common locations relative to script if it's a default name
        script_dir = os.path.dirname(__file__)
        potential_path = os.path.join(script_dir, args.model_path) # handles if model_path is 'models/...' or just '480-float16.tflite'
        if os.path.exists(potential_path):
            args.model_path = potential_path
            logger.info(f"Found model at {args.model_path}")
        else:
            logger.error("Exiting as detector model is crucial.")
            exit(1)
    
    try:
        face_detector_pipeline = FaceDetector(
            detector_model_path=args.model_path,
            conf_threshold=args.conf_thresh,
            nms_threshold=args.nms_thresh,
            detector_num_threads=args.num_threads,
            use_full_tf=args.use_full_tf,
            camera_index=args.cam_idx,
            camera_resolution=(args.cam_res_w, args.cam_res_h),
            detector_input_size=args.detector_input_size
        )
        face_detector_pipeline.start_pipeline()
    except Exception as e:
        logger.error(f"Failed to initialize or run FaceDetector pipeline: {e}", exc_info=True)
    finally:
        CameraManager.release_all() # Ensure all camera resources are freed on exit
        logger.info("Application finished.")

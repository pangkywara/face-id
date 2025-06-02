# p:\Python\PROJECT PKM_KC\face_new\main.py
import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import messagebox, ttk, simpledialog
from PIL import Image, ImageTk
import time
import threading
import queue
import asyncio
import concurrent.futures
import logging

from face_detector import FaceDetector
from anti_spoofing import AntiSpoofing
from face_recognizer import FaceRecognizer
from utils import preprocess_face, logger

# Global model cache to prevent reloading
MODEL_CACHE = {}

class CameraManager:
    """Manages camera operations with error handling and optimization"""
    
    # Class-level camera cache to avoid repeated initialization
    _camera_cache = {}
    
    def __init__(self, camera_index=0, resolution=(640, 480), fps=30):
        self.camera_index = camera_index
        self.resolution = resolution
        self.fps = fps
        self.cap = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=2)  # Buffer for latest frames
        self.lock = threading.Lock()
        self.initialization_started = False
        self.prewarm_complete = False
        
        # Cache key for this camera configuration
        self.cache_key = f"cam_{camera_index}_{resolution[0]}x{resolution[1]}"
        
        # Add a placeholder frame for faster UI display
        self.placeholder_frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
        cv2.putText(self.placeholder_frame, "Starting camera...", (resolution[0]//4, resolution[1]//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        self.frame_queue.put(self.placeholder_frame)
        
    def start(self):
        """Start camera with minimal overhead and quick return"""
        if self.is_running:
            logger.debug("Camera already running, skipping initialization")
            return True
            
        # Check if we already have the camera in the cache
        if self.cache_key in CameraManager._camera_cache:
            logger.info(f"Using cached camera {self.camera_index}")
            with self.lock:
                self.cap = CameraManager._camera_cache[self.cache_key]
                if self.cap and self.cap.isOpened():
                    self.is_running = True
                    self.prewarm_complete = True  # Consider it already warmed up
                    
                    # Start frame capture thread if not already running
                    if not hasattr(self, 'capture_thread') or not self.capture_thread.is_alive():
                        self.capture_thread = threading.Thread(target=self._capture_frames)
                        self.capture_thread.daemon = True
                        self.capture_thread.start()
                    return True
            
        # Start camera in background thread if not initialized yet
        if not self.initialization_started:
            self.initialization_started = True
            camera_thread = threading.Thread(target=self._start_camera_thread)
            camera_thread.daemon = True
            camera_thread.start()
        
        # Return immediately - camera will be ready when it's ready
        return True
        
    def _optimize_usb_polling(self, cap):
        """Optimize USB polling for better performance (Windows-specific)"""
        if os.name == 'nt':
            try:
                # Windows-specific optimizations for DirectShow
                # Set larger buffer size
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                
                # For DirectShow: try to set these properties which might improve USB polling
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                
                # Disable windows wait for device ready (faster but may be unstable on some devices)
                cap.set(cv2.CAP_PROP_SETTINGS, 0)
            except:
                pass
        return cap
        
    def _start_camera_thread(self):
        """Background thread to initialize camera with optimizations"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Always use DSHOW backend on Windows for faster startup
                backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
                
                with self.lock:
                    # First try to open with minimal properties (pre-warming)
                    self.cap = cv2.VideoCapture(self.camera_index, backend)
                    
                    if not self.cap.isOpened():
                        logger.warning(f"Failed to open camera on attempt {attempt+1}")
                        time.sleep(0.1)  # Very short wait before retry
                        continue
                    
                    # Optimize USB polling
                    self.cap = self._optimize_usb_polling(self.cap)
                    
                    # Disable auto focus, auto exposure, and auto white balance for faster startup
                    self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                    self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = manual mode
                    self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
                    
                    # Start with very low resolution for ultra-quick initialization
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)  # Even lower resolution for faster startup
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
                    self.cap.set(cv2.CAP_PROP_FPS, 15)
                    
                    # Skip most frames during warmup - just grab a few frames to ensure camera is active
                    frames_to_discard = 3  # Reduced from 10
                    for _ in range(frames_to_discard):
                        self.cap.grab()  # Just grab without decoding (faster)
                    
                    self.prewarm_complete = True
                    logger.info("Camera pre-warming complete")
                    
                    # Now set to the desired resolution
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                    self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                    
                    # Just grab one frame at target resolution
                    self.cap.grab()
                    
                    if self.cap.isOpened():
                        # Store in cache for reuse
                        CameraManager._camera_cache[self.cache_key] = self.cap
                        self.is_running = True
                        
                        # Start frame capture thread
                        self.capture_thread = threading.Thread(target=self._capture_frames)
                        self.capture_thread.daemon = True
                        self.capture_thread.start()
                        logger.info("Camera started successfully")
                        return
                        
            except Exception as e:
                logger.error(f"Camera initialization attempt {attempt+1} failed: {e}")
                if self.cap:
                    self.cap.release()
                    self.cap = None
                time.sleep(0.1)  # Shorter wait between retries
        
        logger.error("Failed to initialize camera after multiple attempts")
        
    def _capture_frames(self):
        """Optimized frame capture with minimal processing and smart recovery"""
        last_error_time = 0
        error_count = 0
        skip_frame_count = 0
        
        while self.is_running:
            try:
                with self.lock:
                    if not self.cap or not self.cap.isOpened():
                        time.sleep(0.05)
                        continue
                    
                    # Skip frames if system is under load (every 5th frame)
                    skip_frame_count += 1
                    if skip_frame_count % 5 == 0:
                        # Just grab the frame without decoding for frame sync
                        self.cap.grab()
                        continue
                    
                    ret, frame = self.cap.read()
                
                if ret:
                    # Reset error count on successful frame
                    error_count = 0
                    
                    # Replace old frame instead of emptying queue
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.frame_queue.put(frame)
                else:
                    error_count += 1
                    current_time = time.time()
                    
                    # Only log errors occasionally to avoid flooding logs
                    if current_time - last_error_time > 5.0:
                        logger.warning(f"Failed to read frame, error count: {error_count}")
                        last_error_time = current_time
                        
                    # If we have consistent errors, try to recover the camera
                    if error_count > 10:
                        logger.warning("Attempting to recover camera connection")
                        error_count = 0
                        with self.lock:
                            if self.cap:
                                # Try to reset the camera without fully releasing
                                for _ in range(3):
                                    self.cap.grab()  # Try to refresh the stream
                                
                                # If still not working, reinitialize
                                if not self.cap.grab():
                                    self.cap.release()
                                    
                                    # Reopen camera with same settings
                                    backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
                                    self.cap = cv2.VideoCapture(self.camera_index, backend)
                                    self._optimize_usb_polling(self.cap)
                                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                    
                    time.sleep(0.01)
            except Exception as e:
                logger.error(f"Frame capture error: {e}")
                time.sleep(0.05)
        
    def get_frame(self):
        """Get the latest frame, or None if no frame is available"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            # If we're pre-warming but not fully running yet, return a blank frame
            # so the UI doesn't appear frozen
            if self.initialization_started and not self.is_running:
                return np.zeros((240, 320, 3), dtype=np.uint8)
            return None
    
    def is_ready(self):
        """Check if camera is fully initialized and ready"""
        return self.is_running and self.prewarm_complete
            
    def stop(self):
        """Stop the camera safely but keep it in cache if possible"""
        self.is_running = False
        
        # Don't release if in cache - just stop the thread
        if self.cache_key not in CameraManager._camera_cache:
            with self.lock:
                if self.cap and self.cap.isOpened():
                    self.cap.release()
                    self.cap = None
                
    def __del__(self):
        """Ensure camera is released when object is destroyed"""
        self.stop()
        
    @classmethod
    def release_all(cls):
        """Release all cached cameras when application exits"""
        for key, cap in cls._camera_cache.items():
            if cap and cap.isOpened():
                cap.release()
        cls._camera_cache.clear()

class FaceAuthSystem:
    def __init__(self, root, title="Face Authentication System"):
        self.root = root
        self.root.title(title)
        self.root.geometry("800x600")
        
        # Initialize settings dictionary first
        self.settings = {} # Ensure settings attribute exists
        # TODO: Load actual settings here if they come from a file or other source
        # For example: self.load_app_settings()

        # Create a flag for showing face detection overlay
        self.show_detection = False
        
        # Set lazy loading flags - nothing loaded by default
        self.camera_initialized = False
        self.detector_ready = False
        self.recognizer_ready = False
        self.spoof_ready = False
        
        # Create placeholders for components that will be loaded on demand
        self.camera = None
        self.face_detector = None
        self.face_recognizer = None
        self.anti_spoofing = None
        
        # UI setup - do this first for faster UI loading
        self.setup_ui()
        
        # Control variables
        self.running = False
        self.current_user = None
        self.current_image = None
        self.video_panel_height = 450  # Default height for video panel
        
        # Create a placeholder image to show immediately
        self._create_placeholder_image()
        
        # Start UI updates immediately, much faster than waiting for camera
        self.update_frame()
        
        # Start non-blocking initialization in separate threads with priorities:
        # 1. Camera (needed for UI to show something real)
        # 2. Face detector (needed to find faces)
        # 3. Face recognizer and anti-spoofing (needed for actual authentication)
        threading.Thread(target=self._initialize_camera, daemon=True).start()
        threading.Thread(target=self._initialize_models, daemon=True).start()
    
    def _create_placeholder_image(self):
        """Create a placeholder image to show until camera is ready"""
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add text showing app is starting
        cv2.putText(placeholder, "Starting camera...", (180, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        self.placeholder_frame = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
        self.current_image = ImageTk.PhotoImage(Image.fromarray(self.placeholder_frame))
    
    def _initialize_camera(self):
        """Initialize camera in background thread with highest priority"""
        try:
            # Create and start camera - defer other operations
            self.camera = CameraManager()
            self.camera.start()
            self.camera_initialized = True
            self.running = True
            self.log_message("Camera ready")
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
    
    def _initialize_models(self):
        """Initialize all machine learning models with error handling."""
        try:
            logger.info("Initializing detection models...")
            
            if not hasattr(self, 'settings') or not self.settings:
                logger.warning("FaceAuthSystem.settings not found or empty, using hardcoded defaults for FaceDetector.")
                settings_conf_thresh = 0.7
                settings_nms_thresh = 0.4
                settings_cam_index = 0
                settings_cam_res = (640, 480)
                settings_detector_input_size = 480
                # Default to trying tflite_runtime first for low-end compatibility
                settings_use_full_tf = False 
            else:
                settings_conf_thresh = self.settings.get('detection_confidence_threshold', 0.7)
                settings_nms_thresh = self.settings.get('detection_nms_threshold', 0.4)
                settings_cam_index = self.settings.get('camera_index', 0)
                settings_cam_res = self.settings.get('camera_resolution', (640,480))
                settings_detector_input_size = self.settings.get('detector_input_size', 480)
                # Allow overriding via settings, but default False (try tflite_runtime first)
                settings_use_full_tf = self.settings.get('use_full_tensorflow', False) 

            detector_model_file = "models/480-float16.tflite" 
            base_dir = os.path.dirname(__file__)
            default_detector_path = os.path.join(base_dir, detector_model_file)

            if not os.path.exists(default_detector_path):
                logger.warning(f"Default detector model {default_detector_path} not found. Using relative path {detector_model_file}")
            else:
                 detector_model_file = default_detector_path

            self.face_detector = FaceDetector(
                detector_model_path=detector_model_file,
                conf_threshold=settings_conf_thresh,
                nms_threshold=settings_nms_thresh,
                camera_index=settings_cam_index,
                camera_resolution=settings_cam_res,
                detector_input_size=settings_detector_input_size,
                use_full_tf=settings_use_full_tf 
            )
            self.detector_ready = True 
            logger.info(f"Face detector initialized successfully (using {'full TF' if settings_use_full_tf else 'tflite_runtime'} attempt).")
        except ModuleNotFoundError as mnfe: # Specifically catch ModuleNotFoundError for tflite_runtime
             logger.error(f"Error initializing face detector: {mnfe}. Likely missing 'tflite_runtime'.", exc_info=True)
             self.face_detector = None 
             self.detector_ready = False
             messagebox.showerror("Dependency Error", f"Failed to initialize face detector: {mnfe}.\n\nPlease install 'tflite_runtime' or ensure TensorFlow is used.")
        except Exception as e:
            logger.error(f"Error initializing face detector models (using {'full TF' if settings_use_full_tf else 'tflite_runtime'} attempt): {e}", exc_info=True) 
            self.face_detector = None 
            self.detector_ready = False
            messagebox.showerror("Model Initialization Error", f"Failed to initialize face detector. Please check logs. Error: {type(e).__name__}")

    def _init_recognizer(self):
        """Initialize recognizer in separate thread"""
        try:
            self.face_recognizer = FaceRecognizer()
            self.recognizer_ready = True
            self.log_message("Face recognition ready")
        except Exception as e:
            logger.error(f"Error initializing face recognizer: {e}")
    
    def _init_anti_spoofing(self):
        """Initialize anti-spoofing in separate thread"""
        try:
            self.anti_spoofing = AntiSpoofing()
            self.spoof_ready = True
            self.log_message("Anti-spoofing ready")
        except Exception as e:
            logger.error(f"Error initializing anti-spoofing: {e}")

    def setup_ui(self):
        """Create UI elements and initialize state variables"""
        # Main frame for the application
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Video panel - shows camera feed
        video_frame = tk.Frame(main_frame, bg="black")
        video_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create video panel as a label
        self.video_panel = tk.Label(video_frame, bg="black")
        self.video_panel.pack(fill=tk.BOTH, expand=True)
        
        # Status frame for messages and controls
        status_frame = tk.Frame(main_frame)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Status label
        self.status_label = tk.Label(status_frame, text="Initializing system...", font=("Arial", 12))
        self.status_label.pack(side=tk.LEFT, pady=5)
        
        # Buttons frame
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Login button
        self.login_button = tk.Button(button_frame, text="Login", command=self.login, width=15)
        self.login_button.pack(side=tk.LEFT, padx=5)
        
        # Register button
        self.register_button = tk.Button(button_frame, text="Register", command=self.register_new_user, width=15)
        self.register_button.pack(side=tk.LEFT, padx=5)
        
        # Test Mode button - for unlimited login attempts
        self.test_mode_button = tk.Button(button_frame, text="Test Detection", command=self.start_test_mode, width=15)
        self.test_mode_button.pack(side=tk.LEFT, padx=5)
        
        # Settings button
        self.settings_button = tk.Button(button_frame, text="Settings", command=self.open_settings, width=15)
        self.settings_button.pack(side=tk.RIGHT, padx=5)
        
    def update_frame(self):
        """Update the video frame with the latest camera capture."""
        try:
            display_frame_for_ui = None # Initialize with a default

            if not self.camera_initialized or not self.camera or not self.camera.is_running:
                if hasattr(self, 'current_image') and self.current_image: # Check if placeholder exists
                    self.video_panel.config(image=self.current_image)
                    self.video_panel.image = self.current_image
                
                status_img = None
                if hasattr(self, 'placeholder_frame') and self.placeholder_frame is not None:
                    status_img = self.placeholder_frame.copy()
                    y_pos = 280
                    status_texts = []
                    if not self.camera_initialized: status_texts.append("Camera: Initializing...")
                    # Check self.detector_ready AFTER checking self.face_detector exists
                    if not (hasattr(self, 'face_detector') and self.face_detector and self.detector_ready): status_texts.append("Detector: Loading...")
                    if not (hasattr(self, 'face_recognizer') and self.face_recognizer and self.recognizer_ready): status_texts.append("Recognition: Loading...")
                    if not (hasattr(self, 'anti_spoofing') and self.anti_spoofing and self.spoof_ready): status_texts.append("Security: Loading...")
                    
                    for text in status_texts:
                        cv2.putText(status_img, text, (180, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                        y_pos += 30
                    display_frame_for_ui = status_img
            else:
                frame = self.camera.get_frame()
                
                if frame is not None:
                    # If detector is ready, use it to process the frame fully.
                    # process_single_frame returns the frame with all drawings.
                    if hasattr(self, 'face_detector') and self.face_detector and self.detector_ready:
                        processed_frame, _ = self.face_detector.process_single_frame(frame.copy()) # Pass a copy
                        display_frame_for_ui = processed_frame
                    else:
                        # Detector not ready, just display the raw frame
                        display_frame_for_ui = frame.copy()

                    # If other models are not ready, overlay their status on the current display_frame_for_ui
                    if display_frame_for_ui is not None and not (hasattr(self, 'face_detector') and self.face_detector and self.detector_ready):
                        status_texts_overlay = []
                        if not (hasattr(self, 'face_detector') and self.face_detector and self.detector_ready): status_texts_overlay.append("Detector: Loading")
                        # if not (hasattr(self, 'face_recognizer') and self.face_recognizer and self.recognizer_ready): status_texts_overlay.append("Recognizer: Loading") # Removed
                        # if not (hasattr(self, 'anti_spoofing') and self.anti_spoofing and self.spoof_ready): status_texts_overlay.append("Anti-Spoof: Loading") # Removed
                        
                        y_pos_overlay = 30
                        for text in status_texts_overlay:
                            # Adjusted color to be less intrusive, e.g., yellow or orange
                            cv2.putText(display_frame_for_ui, text, (10, y_pos_overlay), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2) # Orange text
                            y_pos_overlay += 25
                else: # frame is None
                    if hasattr(self, 'placeholder_frame') and self.placeholder_frame is not None:
                        display_frame_for_ui = self.placeholder_frame.copy()
                        cv2.putText(display_frame_for_ui, "No camera frame", (180, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

            # Convert to format suitable for tkinter and update panel if a frame was determined
            if display_frame_for_ui is not None:
                display_frame_rgb = cv2.cvtColor(display_frame_for_ui, cv2.COLOR_BGR2RGB)
                img_tk = ImageTk.PhotoImage(Image.fromarray(display_frame_rgb))
                self.video_panel.config(image=img_tk)
                self.video_panel.image = img_tk  
            
            update_interval = 30 # Consistent update interval
            self.root.after(update_interval, self.update_frame)
            
        except Exception as e:
            logger.error(f"Error updating frame: {e}", exc_info=True)
            self.root.after(100, self.update_frame) # Attempt to recover

    def on_successful_login(self, username):
        """Handle actions after successful login"""
        # Update UI for logged-in state
        if hasattr(self, 'login_button'):
            self.login_button.config(state=tk.DISABLED)
            
        if hasattr(self, 'logout_button'):
            self.logout_button.config(state=tk.NORMAL)
            
        # Show welcome message
        welcome_window = tk.Toplevel(self.root)
        welcome_window.title("Welcome")
        welcome_window.geometry("300x200")
        welcome_window.transient(self.root)
        welcome_window.grab_set()
        
        # Welcome message
        tk.Label(welcome_window, text=f"Welcome, {username}!", font=("Arial", 16)).pack(pady=20)
        tk.Label(welcome_window, text="You have been successfully authenticated.").pack(pady=10)
        
        # Close button
        tk.Button(welcome_window, text="OK", command=welcome_window.destroy, width=10).pack(pady=20)

    def _process_frame(self, frame):
        """Process a frame for display with optional overlays"""
        try:
            # Make a copy to avoid modifying the original
            display_frame = frame.copy()
            
            # Add detection overlay if enabled
            if self.show_detection:
                # Get detected face with angle information
                _, face_rect, angle = self._get_detected_face_with_angle(frame, detection_only=True)
                
                # Draw detection rectangle if a face was found
                if face_rect is not None:
                    x, y, w, h = face_rect
                    # Draw rectangle around face
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Add angle information
                    if angle is not None:
                        cv2.putText(display_frame, f"Angle: {angle:.1f}°", 
                                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return display_frame
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return frame

    def _get_detected_face_with_angle(self, frame, detection_only=False):
        """
        Get detected face with head angle estimation - simplified for robustness
        Returns: (face_img, face_rect, angle)
        """
        try:
            # Ensure we have a valid frame
            if frame is None or frame.size == 0:
                return None, None, None
                
            # Try to detect faces using the detector
            faces = self.face_detector.detect_faces(frame)
            
            # Check if any faces were found
            if isinstance(faces, np.ndarray) and faces.shape[0] > 0:
                # If we have multiple faces, use the largest one
                if faces.shape[0] > 1:
                    # Get areas of all faces
                    areas = [w * h for x, y, w, h in faces]
                    # Find index of largest face
                    largest_idx = np.argmax(areas)
                    face_rect = tuple(faces[largest_idx])
                else:
                    # Just one face, convert first element to tuple
                    face_rect = tuple(faces[0])
                    
                # Extract face coordinates
                x, y, w, h = face_rect
                
                # Calculate angle (simpler version)
                center_x = x + w//2
                offset_from_center = (center_x - frame.shape[1]//2) / (frame.shape[1]//4)
                angle = offset_from_center * 15  # Scale to reasonable angle range
                
                # If we only need detection data, return early
                if detection_only:
                    return None, face_rect, angle
                    
                # Extract the face image
                try:
                    face_img = frame[y:y+h, x:x+w].copy()
                    return face_img, face_rect, angle
                except Exception as e:
                    logger.error(f"Error extracting face image: {e}")
                    return None, face_rect, angle
            else:
                # No faces detected, try with more lenient parameters
                faces = self.face_detector.detect_faces(
                    frame, 
                    scale_factor=1.1,
                    min_neighbors=2,
                    min_size=(20, 20)
                )
                
                if isinstance(faces, np.ndarray) and faces.shape[0] > 0:
                    # Process as above
                    if faces.shape[0] > 1:
                        areas = [w * h for x, y, w, h in faces]
                        largest_idx = np.argmax(areas)
                        face_rect = tuple(faces[largest_idx])
                    else:
                        face_rect = tuple(faces[0])
                        
                    x, y, w, h = face_rect
                    angle = 0.0  # Simplified angle calculation
                    
                    if detection_only:
                        return None, face_rect, angle
                        
                    try:
                        face_img = frame[y:y+h, x:x+w].copy()
                        return face_img, face_rect, angle
                    except Exception as e:
                        logger.error(f"Error extracting face image: {e}")
                        return None, face_rect, angle
                else:
                    # Still no faces, return None
                    return None, None, None
                
        except Exception as e:
            logger.error(f"Error getting detected face: {e}")
            return None, None, None

    def _get_current_frame(self):
        """Get the current frame from the camera with error handling"""
        try:
            # Get the latest frame from the camera manager
            frame = self.camera.get_frame()
            
            # Check if frame is valid
            if frame is None or frame.size == 0:
                # If camera isn't running, try to restart it
                if not self.camera.is_running:
                    logger.debug("Camera not running, attempting to restart")
                    self.camera.start()
                return None
                
            return frame
            
        except Exception as e:
            logger.error(f"Error getting frame: {e}")
            return None
            
    def is_camera_active(self):
        """Check if the camera is active and providing frames - fixed to be more reliable"""
        try:
            # Check directly if the camera is running, without requiring a frame
            return self.camera.is_running
        except:
            return False

    def log_message(self, message):
        """Display a message in the status area and log it"""
        if hasattr(self, 'status_label'):
            self.status_label.config(text=message)
        logger.info(message)
        
    def log_audit(self, message):
        """Log an audit message for security tracking"""
        logger.info(f"AUDIT: {message}")

    def login(self):
        """Authenticate the user using face recognition - simplified for reliability"""
        # Ensure camera is initialized first
        if not self.camera_initialized or not self.camera:
            self.log_message("Camera initializing, please wait...")
            messagebox.showinfo("System Starting", "The camera is still initializing. Please try again in a moment.")
            return
            
        # Check if models are initialized
        if not (self.detector_ready and self.recognizer_ready and self.spoof_ready):
            self.log_message("Models still loading, please wait...")
            messagebox.showinfo("System Initializing", "Please wait while the system finishes initializing.")
            return
            
        # Reset any previous detection overlays
        self.show_detection = True
        
        # Set anti-spoofing to normal mode for better reliability
        self.anti_spoofing.set_mode("normal")
        
        # Check if the camera is active - just use the running flag without requiring a frame
        if not self.camera.is_running:
            self.camera.start()  # Try to start it if not running
            time.sleep(0.5)      # Give it a moment to start
            if not self.camera.is_running:
                messagebox.showerror("Error", "Camera not active. Please restart the application.")
                self.show_detection = False
                return
        
        # Log the authentication attempt
        self.log_audit("Authentication attempt initiated")
        self.log_message("Looking for your face...")
        
        # Update UI to show we're processing
        self.root.update()
        
        # Try to detect and recognize face with multiple attempts
        face_found = False
        max_attempts = 5
        
        for attempt in range(max_attempts):
            try:
                # Get a current frame
                frame = self.camera.get_frame()
                if frame is None:
                    time.sleep(0.2)
                    continue
                
                # Get face from frame
                faces = self.face_detector.detect_faces(frame, min_neighbors=3)
                
                # If we found faces
                if isinstance(faces, np.ndarray) and faces.shape[0] > 0:
                    # Choose the largest face
                    if faces.shape[0] > 1:
                        areas = [w * h for x, y, w, h in faces]
                        largest_idx = np.argmax(areas)
                        face_rect = tuple(faces[largest_idx])
                    else:
                        face_rect = tuple(faces[0])
                    
                    # Extract the face coordinates
                    x, y, w, h = face_rect
                    
                    # Security: make sure coordinates are valid
                    if (x < 0 or y < 0 or w <= 0 or h <= 0 or
                        x + w > frame.shape[1] or y + h > frame.shape[0]):
                        logger.warning("Invalid face coordinates detected")
                        continue
                    
                    # Extract face image
                    face_img = frame[y:y+h, x:x+w]
                    face_found = True
                    
                    # Display success message
                    self.log_message("Face detected, verifying identity...")
                    
                    # Check if it's a real face (anti-spoofing)
                    try:
                        is_real, spoof_type = self.anti_spoofing.check(face_img)
                        
                        if not is_real:
                            self.log_message(f"Spoof detected: {spoof_type}")
                            messagebox.showerror("Security Alert", f"Spoof detected: {spoof_type}\nAccess denied.")
                            self.show_detection = False
                            return
                    except Exception as e:
                        logger.error(f"Anti-spoofing error: {e}")
                        # Continue with recognition even if anti-spoofing fails
                    
                    # Recognize the face
                    try:
                        user_id, confidence = self.face_recognizer.recognize(face_img)
                        
                        if user_id is not None and confidence >= 0.6:  # Lower threshold for better UX
                            # Success!
                            self.log_message(f"Welcome, {user_id}! ({confidence:.2f})")
                            self.log_audit(f"Successful authentication - {user_id}")
                            self.current_user = user_id
                            self.on_successful_login(user_id)
                            self.show_detection = False
                            return
                        else:
                            self.log_message("Face not recognized")
                            messagebox.showinfo("Access Denied", "Face not recognized")
                            self.show_detection = False
                            return
                    except Exception as e:
                        logger.error(f"Recognition error: {e}")
                        messagebox.showerror("Error", "Recognition failed. Please try again.")
                        self.show_detection = False
                        return
                    
                    # We're done if we got here with a face
                    break
                        
                # If we're on the first attempt, show searching message
                if attempt == 0:
                    self.log_message("Searching for your face...")
                    
                # Small delay between attempts
                time.sleep(0.2)
                    
            except Exception as e:
                logger.error(f"Login attempt {attempt} error: {e}")
        
        # If we didn't find a face after all attempts
        if not face_found:
            self.log_message("No face detected. Please center your face and ensure good lighting.")
            messagebox.showinfo("Face Not Detected", 
                              "Your face could not be detected. Please make sure:\n"
                              "1. You are directly facing the camera\n"
                              "2. Your face is well lit\n"
                              "3. You are not too far from the camera")
        
        # Always reset the UI state
        self.show_detection = False

    def register_new_user(self):
        """Register a new user in the face recognition system"""
        # Check if the camera is active
        if not self.is_camera_active():
            messagebox.showerror("Error", "Camera is not active. Please restart the application.")
            return
            
        # Ask for the user's name
        name = simpledialog.askstring("Register New User", "Enter your name:")
        if not name or name.strip() == "":
            return  # User canceled
            
        # Log the registration attempt
        self.log_audit(f"Registration attempt for {name}")
        
        # Set anti-spoofing to registration mode (less strict)
        self.anti_spoofing.set_mode("registration")
        
        # Enable face detection overlay
        self.show_detection = True
        
        try:
            # Create progress window
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Registration")
            progress_window.geometry("400x200")
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            # Status label
            status_label = tk.Label(progress_window, text="Capturing face images...", font=("Arial", 12))
            status_label.pack(pady=10)
            
            # Tip label
            tip_label = tk.Label(progress_window, 
                                text="Position your face in the center and look directly at the camera.\n"
                                     "Try to maintain a neutral expression.")
            tip_label.pack(pady=10)
            
            # Status counter label
            self.counter_label = tk.Label(progress_window, text="0/5 images captured")
            self.counter_label.pack(pady=5)
            
            # Update UI
            progress_window.update()
            
            # Collect face images
            face_images = []
            max_images = 5
            capture_delay = 0.5  # seconds between captures
            last_capture_time = time.time()
            
            # Set timeout for capturing (30 seconds max)
            start_time = time.time()
            timeout = 30  # seconds
            
            while len(face_images) < max_images:
                # Check for timeout
                if time.time() - start_time > timeout:
                    status_label.config(text="Timeout. Could not capture enough clear images.")
                    self.root.after(2000, progress_window.destroy)
                    return
                    
                # Process events to keep UI responsive
                self.root.update()
                
                # Wait for capture delay
                if time.time() - last_capture_time < capture_delay:
                    time.sleep(0.1)
                    continue
                    
                # Get current frame
                frame = self._get_current_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                    
                # Get detected face
                face_img, face_rect, angle = self._get_detected_face_with_angle(frame)
                
                # Skip if no face detected or face is not properly aligned
                if face_img is None or face_rect is None:
                    status_label.config(text="No face detected. Please center your face.")
                    time.sleep(0.1)
                    continue
                    
                # Check face angle - needs to be relatively straight for good registration
                if angle is not None and abs(angle) > 15:  # 15 degrees threshold
                    status_label.config(text=f"Head tilted ({angle:.1f}°). Please straighten your head.")
                    time.sleep(0.1)
                    continue
                
                # Check for spoofing
                is_real, _, _ = self.anti_spoofing.check_with_extended_info(face_img)
                if not is_real:
                    status_label.config(text="Security check failed. Please ensure you are not using a photo.")
                    time.sleep(0.5)
                    continue
                    
                # Add face image to collection
                face_images.append(face_img)
                last_capture_time = time.time()
                
                # Update counter
                self.counter_label.config(text=f"{len(face_images)}/{max_images} images captured")
                
                # Short delay to show the progress update
                self.root.after(300)
                self.root.update()
            
            # Update status
            status_label.config(text="Processing images...")
            progress_window.update()
            
            # Register the user with captured face images
            success = self.face_recognizer.register_new_user(name, face_images)
            
            if success:
                status_label.config(text="Registration successful!")
                self.log_audit(f"Registration successful for {name}")
                
                # Close window after delay
                self.root.after(2000, progress_window.destroy)
                
                # Show success message
                messagebox.showinfo("Success", f"User {name} has been registered successfully.")
            else:
                status_label.config(text="Registration failed. Please try again.")
                self.log_audit(f"Registration failed for {name}")
                
                # Close window after delay
                self.root.after(2000, progress_window.destroy)
                
                # Show error message
                messagebox.showerror("Error", "Failed to register user. Please try again.")
                
        except Exception as e:
            logger.error(f"Registration error: {e}")
            self.log_message("Error during registration")
            messagebox.showerror("Error", f"Registration error: {str(e)}")
            
        finally:
            # Reset anti-spoofing mode
            self.anti_spoofing.set_mode("normal")
            
            # Disable detection overlay
            self.show_detection = False

    def open_settings(self):
        """Open the settings window to configure system parameters"""
        try:
            # Create settings window
            settings_window = tk.Toplevel(self.root)
            settings_window.title("Settings")
            settings_window.geometry("500x400")
            settings_window.transient(self.root)
            settings_window.grab_set()
            
            # Settings notebook with tabs
            notebook = ttk.Notebook(settings_window)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # General settings tab
            general_tab = ttk.Frame(notebook)
            notebook.add(general_tab, text="General")
            
            # Security settings tab
            security_tab = ttk.Frame(notebook)
            notebook.add(security_tab, text="Security")
            
            # Database settings tab
            database_tab = ttk.Frame(notebook)
            notebook.add(database_tab, text="Database")
            
            #
            # General Settings
            #
            tk.Label(general_tab, text="General Settings", font=("Arial", 12, "bold")).pack(pady=10)
            
            # Detection display toggle
            detection_frame = tk.Frame(general_tab)
            detection_frame.pack(fill=tk.X, pady=5)
            
            self.detection_var = tk.BooleanVar(value=self.show_detection)
            tk.Label(detection_frame, text="Show face detection:").pack(side=tk.LEFT, padx=5)
            tk.Checkbutton(detection_frame, variable=self.detection_var, 
                           command=lambda: setattr(self, 'show_detection', self.detection_var.get())).pack(side=tk.LEFT)
            
            # Frame rate setting
            fps_frame = tk.Frame(general_tab)
            fps_frame.pack(fill=tk.X, pady=5)
            
            tk.Label(fps_frame, text="Frame update rate (ms):").pack(side=tk.LEFT, padx=5)
            fps_values = [15, 30, 50, 100]
            fps_combo = ttk.Combobox(fps_frame, values=fps_values, width=5)
            fps_combo.pack(side=tk.LEFT)
            fps_combo.current(1)  # Default 30ms
            
            #
            # Security Settings
            #
            tk.Label(security_tab, text="Security Settings", font=("Arial", 12, "bold")).pack(pady=10)
            
            # Security threshold
            threshold_frame = tk.Frame(security_tab)
            threshold_frame.pack(fill=tk.X, pady=5)
            
            tk.Label(threshold_frame, text="Recognition confidence threshold:").pack(side=tk.LEFT, padx=5)
            threshold_scale = tk.Scale(threshold_frame, from_=0.5, to=0.95, resolution=0.05, orient=tk.HORIZONTAL, length=200)
            threshold_scale.set(0.7)  # Default threshold
            threshold_scale.pack(side=tk.LEFT)
            
            # Anti-spoofing settings
            spoof_frame = tk.Frame(security_tab)
            spoof_frame.pack(fill=tk.X, pady=5)
            
            tk.Label(spoof_frame, text="Anti-spoofing mode:").pack(side=tk.LEFT, padx=5)
            spoof_modes = ["normal", "strict", "lenient"]
            spoof_combo = ttk.Combobox(spoof_frame, values=spoof_modes, width=10)
            spoof_combo.pack(side=tk.LEFT)
            spoof_combo.current(0)  # Default normal
            
            # Capture log toggle
            log_frame = tk.Frame(security_tab)
            log_frame.pack(fill=tk.X, pady=5)
            
            self.log_var = tk.BooleanVar(value=False)
            tk.Label(log_frame, text="Enable security audit logging:").pack(side=tk.LEFT, padx=5)
            tk.Checkbutton(log_frame, variable=self.log_var).pack(side=tk.LEFT)
            
            #
            # Database Settings
            #
            tk.Label(database_tab, text="Database Management", font=("Arial", 12, "bold")).pack(pady=10)
            
            # Database info
            db_info_frame = tk.Frame(database_tab)
            db_info_frame.pack(fill=tk.X, pady=5)
            
            # Get user count
            try:
                user_count = len(self.face_recognizer.get_user_ids())
                db_status = "OK"
            except:
                user_count = "Unknown"
                db_status = "Error"
                
            tk.Label(db_info_frame, text=f"Users in database: {user_count}").pack(anchor=tk.W, padx=10)
            tk.Label(db_info_frame, text=f"Database status: {db_status}").pack(anchor=tk.W, padx=10)
            
            # Database actions
            btn_frame = tk.Frame(database_tab)
            btn_frame.pack(fill=tk.X, pady=10)
            
            # Repair button
            repair_btn = tk.Button(btn_frame, text="Repair Database", width=15,
                                 command=self._repair_database_from_settings)
            repair_btn.pack(pady=5)
            
            # Backup button
            backup_btn = tk.Button(btn_frame, text="Backup Database", width=15)
            backup_btn.pack(pady=5)
            
            # Reset button
            reset_btn = tk.Button(btn_frame, text="Reset Database", width=15, fg="red",
                                command=lambda: messagebox.showwarning("Warning", 
                                                                     "This functionality is disabled for security reasons."))
            reset_btn.pack(pady=5)
            
            # Apply & Close buttons
            btn_frame = tk.Frame(settings_window)
            btn_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=10)
            
            # Apply button action
            def apply_settings():
                try:
                    # Apply security threshold
                    threshold_value = threshold_scale.get()
                    
                    # Apply anti-spoofing mode
                    spoof_mode = spoof_combo.get()
                    self.anti_spoofing.set_mode(spoof_mode)
                    
                    # Apply frame rate
                    frame_rate = int(fps_combo.get())
                    
                    # Log settings changes
                    self.log_audit(f"Settings updated: threshold={threshold_value}, spoof_mode={spoof_mode}")
                    messagebox.showinfo("Settings", "Settings applied successfully.")
                except Exception as e:
                    logger.error(f"Error applying settings: {e}")
                    messagebox.showerror("Error", f"Could not apply settings: {str(e)}")
            
            # Apply button
            apply_btn = tk.Button(btn_frame, text="Apply", width=10, command=apply_settings)
            apply_btn.pack(side=tk.RIGHT, padx=5)
            
            # Close button
            close_btn = tk.Button(btn_frame, text="Close", width=10, command=settings_window.destroy)
            close_btn.pack(side=tk.RIGHT, padx=5)
            
        except Exception as e:
            logger.error(f"Error opening settings: {e}")
            messagebox.showerror("Error", f"Could not open settings: {str(e)}")
            
    def _repair_database_from_settings(self):
        """Repair user database from settings panel"""
        try:
            # Show confirmation dialog
            confirm = messagebox.askyesno("Confirm Repair", 
                                        "This will scan the database for corrupted entries. Continue?")
            if not confirm:
                return
                
            # Perform the repair
            users_fixed, users_removed = self.face_recognizer.repair_user_database()
            
            # Show results
            if users_fixed > 0 or users_removed > 0:
                messagebox.showinfo("Database Repaired", 
                                  f"The face database has been repaired.\n\n"
                                  f"Users fixed: {users_fixed}\n"
                                  f"Users removed: {users_removed}")
            else:
                messagebox.showinfo("Database Healthy", 
                                  "No issues found with the database.")
                
            # Log the repair action
            self.log_audit(f"Database repaired: {users_fixed} fixed, {users_removed} removed")
            
        except Exception as e:
            logger.error(f"Error repairing database from settings: {e}")
            messagebox.showerror("Error", f"Could not repair database: {str(e)}")

    def start_test_mode(self):
        """Start a test mode that continuously checks for faces and shows detection info"""
        # Prevent multiple test windows
        if hasattr(self, 'test_window') and self.test_window and self.test_window.winfo_exists():
            self.test_window.lift()
            return

        self.test_mode_active = True # Flag to control the update loop
        
        # Create the Toplevel window
        self.test_window = tk.Toplevel(self.root)
        self.test_window.title("Face Detection Test Mode")
        self.test_window.geometry("550x450") # Adjusted size for clarity
        self.test_window.transient(self.root)
        
        # Ensure closing the window stops the test mode update loop
        self.test_window.protocol("WM_DELETE_WINDOW", lambda: self._close_test_mode(self.test_window))

        # Main frame for layout within the test window
        main_test_frame = tk.Frame(self.test_window)
        main_test_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Status Labels Frame --- 
        status_labels_frame = tk.Frame(main_test_frame)
        status_labels_frame.pack(fill=tk.X, pady=(0,10))
        
        # Define labels as instance attributes for the update function
        self.test_face_status_label = tk.Label(status_labels_frame, text="Face detected: No", font=("Arial", 10))
        self.test_face_status_label.pack(anchor=tk.W)
        
        self.test_recognition_label = tk.Label(status_labels_frame, text="Recognition: None", font=("Arial", 10))
        self.test_recognition_label.pack(anchor=tk.W)
        
        self.test_spoof_label = tk.Label(status_labels_frame, text="Anti-spoof: Not checked", font=("Arial", 10))
        self.test_spoof_label.pack(anchor=tk.W)
        
        self.test_confidence_label = tk.Label(status_labels_frame, text="Confidence: N/A", font=("Arial", 10))
        self.test_confidence_label.pack(anchor=tk.W)

        # --- Scrolled Text Area Frame --- 
        text_area_frame = tk.Frame(main_test_frame)
        # Configure the text area frame to take up remaining space
        text_area_frame.pack(fill=tk.BOTH, expand=True, pady=(0,5))

        # Create Scrolled Text widget and assign to self.test_results_text
        self.test_results_text = tk.Text(text_area_frame, height=15, wrap=tk.WORD, font=("Arial", 9))
        scrollbar = tk.Scrollbar(text_area_frame, command=self.test_results_text.yview)
        self.test_results_text.config(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar first, then text widget to fill remaining space
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.test_results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # --- Close Button --- 
        close_btn = tk.Button(main_test_frame, text="Close Test Mode", command=lambda: self._close_test_mode(self.test_window))
        close_btn.pack(pady=5, side=tk.BOTTOM)
        
        # --- Internal Update Function --- 
        def update_test_results_display(): # Keep name distinct if needed
            # Check if the test window still exists and mode is active
            if not self.test_mode_active or not hasattr(self, 'test_window') or not self.test_window or not self.test_window.winfo_exists():
                return # Stop the loop if window is closed or mode deactivated
            
            frame = None
            # Safely check camera and its readiness
            if hasattr(self, 'camera') and self.camera and self.camera.is_ready():
                frame = self.camera.get_frame()
            
            # Safely check detector and its readiness
            if frame is not None and hasattr(self, 'face_detector') and self.face_detector and self.detector_ready:
                try:
                    # Pass a copy of the frame
                    processed_display_frame, detected_faces_list = self.face_detector.process_single_frame(frame.copy())
                    
                    # Update status labels based on the first detected face
                    if detected_faces_list:
                        first_face = detected_faces_list[0] 
                        name = first_face.get('recognized_name', 'Unknown')
                        confidence = first_face.get('recognition_confidence', 0.0)
                        is_real = first_face.get('is_real', False)
                        spoof_type = first_face.get('spoof_type', 'N/A')
                        detection_score = first_face.get('detection_score', 0.0)

                        self.test_face_status_label.config(text=f"Face detected: Yes ({len(detected_faces_list)} found)", fg="green")
                        self.test_recognition_label.config(text=f"Recognition: {name}", fg="green" if name != "Unknown" else "blue")
                        self.test_spoof_label.config(text=f"Anti-spoof: {spoof_type} ('Real' if is_real else 'Spoof')", fg="green" if is_real else "red")
                        self.test_confidence_label.config(text=f"Det. Score: {detection_score:.2f} | Rec. Conf: {confidence:.2f}")
                        
                        # Log details for *all* detected faces to the text area
                        self.test_results_text.insert(tk.END, f"--- Frame @ {time.strftime('%H:%M:%S')} ---\n")
                        for idx, face_info in enumerate(detected_faces_list):
                            log_name = face_info.get('recognized_name', 'Unknown')
                            log_conf = face_info.get('recognition_confidence', 0.0)
                            log_real = face_info.get('is_real', False)
                            log_spoof = face_info.get('spoof_type', 'N/A')
                            log_det_score = face_info.get('detection_score', 0.0)
                            self.test_results_text.insert(tk.END, f"  Face {idx+1}: ID={log_name}({log_conf:.2f}) | Real={log_real}({log_spoof}) | Det={log_det_score:.2f}\n")
                        self.test_results_text.see(tk.END) # Scroll to the latest entry
                    else:
                        # No faces detected in this frame
                        self.test_face_status_label.config(text="Face detected: No", fg="red")
                        self.test_recognition_label.config(text="Recognition: None")
                        self.test_spoof_label.config(text="Anti-spoof: Not checked")
                        self.test_confidence_label.config(text="Confidence: N/A")
                except Exception as e:
                    logger.error(f"Error during test mode frame processing: {e}", exc_info=True)
                    self.test_face_status_label.config(text="Face detected: PROCESSING ERROR", fg="red")
            else:
                # Conditions for processing not met (e.g., no frame, detector not ready)
                status_msg = "Waiting for "
                wait_list = []
                if frame is None: wait_list.append("frame")
                if not (hasattr(self, 'face_detector') and self.face_detector and self.detector_ready): wait_list.append("detector")
                status_msg += " & ".join(wait_list)
                self.test_face_status_label.config(text=f"Face detected: ({status_msg})", fg="orange")
                # Reset other labels if waiting
                self.test_recognition_label.config(text="Recognition: Waiting")
                self.test_spoof_label.config(text="Anti-spoof: Waiting")
                self.test_confidence_label.config(text="Confidence: N/A")

            # Schedule the next update only if the mode is still active
            if self.test_mode_active and self.test_window and self.test_window.winfo_exists():
                self.test_window.after(100, update_test_results_display) # Update ~10 times/sec
        
        # Start the update loop
        update_test_results_display()
        
    def _close_test_mode(self, test_window_ref):
        """Closes the test mode window and stops the update loop."""
        self.test_mode_active = False # Signal the update loop to stop
        if hasattr(self, 'test_window') and self.test_window:
            if self.test_window.winfo_exists():
                self.test_window.destroy()
            self.test_window = None # Clear the reference
        logger.info("Test mode closed.")

    @property
    def is_running(self):
        """Check if the face authentication system is running"""
        return self.running

if __name__ == "__main__":
    # Set process priority higher (platform-specific)
    try:
        import psutil
        process = psutil.Process(os.getpid())
        if os.name == 'nt':  # Windows
            process.nice(psutil.HIGH_PRIORITY_CLASS)
        else:  # Unix
            process.nice(-10)  # Lower value = higher priority
    except:
        pass  # Ignore if psutil not available
        
    root = tk.Tk()
    app = FaceAuthSystem(root)
    root.mainloop()
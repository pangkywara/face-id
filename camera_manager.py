import cv2
import numpy as np
import os
import time
import threading
import queue
from utils import logger

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
        max_retries = 2  # Reduced from 3 for faster startup
        for attempt in range(max_retries):
            try:
                # Always use DSHOW backend on Windows for faster startup
                backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
                
                with self.lock:
                    # Ultra-minimal initialization - open with default settings first
                    self.cap = cv2.VideoCapture(self.camera_index, backend)
                    
                    if not self.cap.isOpened():
                        # Try again with a different backend if first fails
                        self.cap = cv2.VideoCapture(self.camera_index)
                        if not self.cap.isOpened():
                            logger.warning(f"Failed to open camera on attempt {attempt+1}")
                            time.sleep(0.05)  # Very short wait before retry
                            continue
                    
                    # Optimize USB polling - minimal settings only
                    self.cap = self._optimize_usb_polling(self.cap)
                    
                    # Disable auto settings that slow down startup
                    self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                    
                    # Skip warming up - just set the resolution immediately
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                    self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                    
                    # Just grab one frame to make sure we can
                    self.cap.grab()
                    self.prewarm_complete = True
                    
                    # Don't log until later to avoid slowing down startup
                    
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
                time.sleep(0.05)  # Shorter wait between retries
        
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
                    
                    # Skip frames if system is under load (every 3rd frame)
                    skip_frame_count += 1
                    if skip_frame_count % 3 == 0:
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
                    if error_count > 5:  # Reduced from 10 to recover faster
                        logger.warning("Attempting to recover camera connection")
                        error_count = 0
                        with self.lock:
                            if self.cap:
                                # Try to reset the camera without fully releasing
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
import cv2
import numpy as np
import os
import onnxruntime as ort
import time
from utils import logger
import threading

# Global cache for model instances with a lock for thread safety
MODEL_CACHE = {}
CACHE_LOCK = threading.Lock()

# Prewarm flag to avoid redundant warming
PREWARM_COMPLETE = False

SPOOF_TYPE = [
    'Live',                               # 0 - live
    'Photo', 'Poster', 'A4',              # 1,2,3 - PRINT
    'Face Mask', 'Upper Body Mask', 'Region Mask',  # 4,5,6 - PAPER CUT
    'PC', 'Pad', 'Phone',                 # 7,8,9 - REPLAY
    '3D Mask'                             # 10 - 3D MASK
]

def prewarm_module():
    """Pre-warm the anti-spoofing module at import time"""
    global PREWARM_COMPLETE
    if not PREWARM_COMPLETE:
        try:
            # Create dummy input for pre-warming
            dummy_input = np.zeros((128, 128, 3), dtype=np.uint8)
            
            # Initialize model with minimal options for fast startup
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'AntiSpoofing_bin_128.onnx')
            
            if os.path.exists(model_path):
                # Use simpler options for pre-warming
                simple_options = ort.SessionOptions()
                session = ort.InferenceSession(
                    model_path,
                    providers=['CPUExecutionProvider'],
                    sess_options=simple_options
                )
                
                # Do a minimal inference to warm up the system
                input_name = session.get_inputs()[0].name
                preprocessed = np.expand_dims(np.transpose(dummy_input.astype(np.float32) / 255.0, (2, 0, 1)), axis=0)
                session.run(None, {input_name: preprocessed})
                
                # Don't cache this initial model, it's just for warming up
                PREWARM_COMPLETE = True
        except Exception as e:
            # Silently handle pre-warming errors - they're not critical
            pass

# Call pre-warming when module is imported
threading.Thread(target=prewarm_module, daemon=True).start()

class AntiSpoofing:
    def __init__(self, use_cached=True):
        # Path to the ONNX model
        self.model_path = os.path.join(os.path.dirname(__file__), 'models', 'AntiSpoofing_bin_128.onnx')
        
        # Model parameters
        self.model_img_size = 128
        self.debug = False  # Set to True to enable verbose logging
        
        # Decision parameters - adjusted for better accuracy
        self.live_confidence_threshold = 0.45  # Balanced threshold for normal operation
        self.decision_frames = 5  # Increased number of frames to consider for more stable results
        self.recent_predictions = []  # Store recent predictions
        self.recent_live_probs = []   # Store raw probabilities for smoother decisions
        
        # Special thresholds for different modes
        self.registration_threshold = 0.25  # More lenient for registration
        self.strict_threshold = 0.65     # More strict for high security scenarios
        
        # Current operation mode
        self.current_mode = "normal"  # "normal", "registration", "strict"
        
        # Safe mode enabled by default
        self.safe_mode = True
        
        # Bypass mode - if True, always authenticates as real face (for debugging when model has issues)
        self.bypass_mode = False
        
        # Preprocessing cache for faster repeated operations
        self._preprocess_cache = {}
        self._cache_size_limit = 10  # Limit cache size
        
        # Pre-allocate a fixed buffer for preprocessing
        self._preprocess_buffer = np.zeros((1, 3, self.model_img_size, self.model_img_size), dtype=np.float32)
        
        # Phone spoofing detection parameters
        self.phone_pattern_threshold = 0.65  # Adjusted threshold for phone detection
        self.consecutive_phone_detections = 0  # Counter for consecutive phone detections
        
        # Check if model is in cache and use it if requested
        if use_cached:
            with CACHE_LOCK:
                if 'anti_spoofing' in MODEL_CACHE and MODEL_CACHE['anti_spoofing']:
                    logger.info("Using cached anti-spoofing model")
                    self.session = MODEL_CACHE['anti_spoofing']
                    if self.session:
                        self.input_name = self.session.get_inputs()[0].name
                    else:
                        self.input_name = None
                    return
        
        # Load the model if it exists
        self._load_model()
    
    def _log(self, message):
        """Log message if debug is enabled"""
        if self.debug:
            logger.debug(f"[AntiSpoofing] {message}")
    
    def _load_model(self):
        """Load the ONNX model with optimizations"""
        if os.path.exists(self.model_path):
            try:
                start_time = time.time()
                
                # First just check that the file exists - don't load yet
                if os.path.getsize(self.model_path) < 100:
                    logger.error("Anti-spoofing model file appears corrupt")
                    self.session = None
                    return
                
                # Defer actual model loading until first use
                # This makes startup much faster
                logger.info("Anti-spoofing model path verified (lazy loading enabled)")
                self.session = None
                self.model_loaded = False
                self.model_loading = False
                
                # Start loading in background with very low priority
                # This will only complete loading if system is idle
                threading.Timer(1.0, self._load_model_in_background).start()
                                
            except Exception as e:
                logger.error(f"Error loading anti-spoofing model: {e}")
                self.session = None
                with CACHE_LOCK:
                    MODEL_CACHE['anti_spoofing'] = None
        else:
            logger.error("Anti-spoofing model not found")
            self.session = None
    
    def _load_model_in_background(self):
        """Load the model in background thread with lowest priority"""
        if self.model_loading or self.model_loaded:
            return
            
        self.model_loading = True
        
        try:
            # Use minimal settings for fastest loading
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session_options.intra_op_num_threads = 1
            session_options.inter_op_num_threads = 1
            session_options.enable_mem_pattern = False
            session_options.enable_mem_reuse = True
            
            # Just CPU provider for initial load
            providers = ['CPUExecutionProvider']
            
            # Create the session with minimal compute
            self.session = ort.InferenceSession(
                self.model_path, 
                providers=providers,
                sess_options=session_options
            )
            
            # Cache the model
            with CACHE_LOCK:
                MODEL_CACHE['anti_spoofing'] = self.session
            
            # Get input name
            self.input_name = self.session.get_inputs()[0].name
            
            # Mark as loaded
            self.model_loaded = True
            
            # No need to warm up here - will do that on first use
            logger.info(f"Anti-spoofing model loaded in background")
            
            # Optimize model in another background thread
            threading.Thread(target=self._optimize_model_post_load, daemon=True).start()
            
        except Exception as e:
            logger.error(f"Error in background model loading: {e}")
            self.model_loading = False
    
    def _ensure_model_loaded(self):
        """Make sure model is loaded before use - blocks if needed"""
        # Return if already loaded
        if self.model_loaded and self.session is not None:
            return True
            
        # If background loading hasn't started yet, start it synchronously
        if not self.model_loading:
            logger.info("Loading anti-spoofing model on demand")
            try:
                # Use minimal settings for fastest loading
                session_options = ort.SessionOptions()
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
                session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                session_options.intra_op_num_threads = 1
                session_options.inter_op_num_threads = 1
                
                # Just CPU provider for initial load
                providers = ['CPUExecutionProvider']
                
                # Create the session with minimal options
                self.session = ort.InferenceSession(
                    self.model_path, 
                    providers=providers,
                    sess_options=session_options
                )
                
                # Cache the model
                with CACHE_LOCK:
                    MODEL_CACHE['anti_spoofing'] = self.session
                
                # Get input name
                self.input_name = self.session.get_inputs()[0].name
                
                # Mark as loaded
                self.model_loaded = True
                
                # No optimization yet - will do that later
                return True
                
            except Exception as e:
                logger.error(f"Error loading anti-spoofing model: {e}")
                return False
        else:
            # Model is loading in background - wait with timeout
            wait_time = 0
            while not self.model_loaded and wait_time < 2.0:
                time.sleep(0.1)
                wait_time += 0.1
                
            return self.model_loaded and self.session is not None
    
    def _optimize_model_post_load(self):
        """Optimize the model after it's already loaded and working (in background)"""
        if self.session is None:
            return
            
        try:
            # Check for better providers now that model is working
            better_providers = ['CPUExecutionProvider']
            try:
                # Check if CUDA is available (for NVIDIA GPUs)
                if 'CUDAExecutionProvider' in ort.get_available_providers():
                    better_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                # Check if DirectML is available (for other GPUs, especially on Windows)
                elif 'DmlExecutionProvider' in ort.get_available_providers():
                    better_providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
            except:
                # Fallback to CPU if error checking providers
                pass
                
            # If we found better providers, recreate the session
            if better_providers != ['CPUExecutionProvider']:
                # Create optimized session options
                session_options = ort.SessionOptions()
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                session_options.intra_op_num_threads = min(4, os.cpu_count() or 1)
                session_options.enable_mem_pattern = True
                session_options.enable_mem_reuse = True
                
                # Create new session with better providers
                new_session = ort.InferenceSession(
                    self.model_path, 
                    providers=better_providers,
                    sess_options=session_options
                )
                
                # Replace session
                with CACHE_LOCK:
                    self.session = new_session
                    MODEL_CACHE['anti_spoofing'] = new_session
                    logger.info(f"Anti-spoofing model optimized with providers: {better_providers}")
        except Exception as e:
            logger.warning(f"Could not optimize anti-spoofing model: {e}")
    
    def preprocess_face(self, face_img):
        """
        Preprocess the face image for the network with enhanced processing for better anti-spoofing
        """
        if face_img is None or face_img.size == 0:
            raise ValueError("Invalid face image provided")
        
        # Check cache using hash of image data
        img_hash = hash(face_img.tobytes())
        if img_hash in self._preprocess_cache:
            return self._preprocess_cache[img_hash]
        
        # Convert BGR to RGB if needed
        if len(face_img.shape) == 3 and face_img.shape[2] == 3:
            img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        else:
            img = face_img.copy()
        
        # ENHANCEMENT: Apply texture enhancement to better detect screen patterns
        # This helps with detecting phones/screens by enhancing their high-frequency patterns
        if self.current_mode == "strict":
            # Apply sharpening to reveal screen patterns
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            img = cv2.filter2D(img, -1, kernel)
            
            # Enhance color saturation which helps detect screens
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.2, 0, 255)  # Increase saturation
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Resize with aspect ratio preservation
        new_size = self.model_img_size
        h, w = img.shape[:2]
        
        # Calculate new dimensions
        ratio = new_size / max(h, w)
        new_h, new_w = int(h * ratio), int(w * ratio)
        
        # Better resize method for anti-spoofing - uses INTER_AREA for downsampling
        # which better preserves the texture patterns used to detect spoofs
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Pre-calculate padding
        pad_h = new_size - new_h
        pad_w = new_size - new_w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        # Add padding
        if pad_h > 0 or pad_w > 0:
            img = cv2.copyMakeBorder(img, top, bottom, left, right, 
                                    cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        # Normalize image
        img = img.astype(np.float32) / 255.0
        
        # Optimize for ONNX input (NCHW format)
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        
        # Use pre-allocated buffer
        self._preprocess_buffer[0] = img
        preprocessed = self._preprocess_buffer
        
        # Cache the result for future use
        if len(self._preprocess_cache) >= self._cache_size_limit:
            self._preprocess_cache.clear()
        self._preprocess_cache[img_hash] = preprocessed
        
        return preprocessed
    
    def check(self, face_img):
        """
        Check if the face is real or a spoof
        Returns: (is_real, spoof_type)
        """
        if self.bypass_mode:
            return True, 'Live'
            
        if self.session is None:
            # Return as real if model is not available
            return True, 'Live'
        
        try:
            # Get extended info which includes live probability
            is_real, spoof_type, live_prob = self.check_with_extended_info(face_img)
            return is_real, spoof_type
                
        except Exception as e:
            logger.error(f"Error in anti-spoofing check: {e}")
            # Return as real in case of error (fail open)
            return True, 'Error'
    
    def _detect_phone_spoof_patterns(self, face_img):
        """
        Enhanced detection of patterns typical of phone screens with multi-algorithm approach
        """
        try:
            # Convert to grayscale
            if len(face_img.shape) == 3:
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_img.copy()
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # 1. Check for high frequency patterns (moiré effects) with improved detection
            # Apply Laplacian filter to detect edges/high frequency content
            laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
            laplacian_var = np.var(laplacian)
            
            # 2. Check for uniform illumination (screens tend to have uniform lighting)
            mean_val = np.mean(gray)
            std_val = np.std(gray)
            uniformity = 1 - (std_val / (mean_val + 1e-5))  # Normalized uniformity
            
            # 3. Enhanced screen reflection detection
            # Most phone photos have characteristic reflection patterns
            # Use Sobel filter which better detects screen edge patterns
            sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            
            # Compute gradient magnitude
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            gradient_mean = np.mean(gradient_magnitude)
            
            # 4. IMPROVED: Better FFT analysis for screen pattern detection
            # Calculate FFT and analyze spectral properties more thoroughly
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.log(np.abs(f_shift) + 1)
            
            # Calculate energy in high frequency regions (phone screens have more energy there)
            h, w = magnitude.shape
            center_h, center_w = h//2, w//2
            
            # Create multiple bands to analyze frequency patterns more precisely
            # Phone screens often show specific frequency patterns
            inner_mask_size = min(h, w) // 6
            middle_mask_size = min(h, w) // 4
            outer_mask_size = min(h, w) // 3
            
            # Create masks for different frequency bands
            inner_mask = np.zeros_like(magnitude)
            middle_mask = np.zeros_like(magnitude)
            outer_mask = np.zeros_like(magnitude)
            
            # Set up masks for different frequency regions
            inner_mask[center_h-inner_mask_size:center_h+inner_mask_size, 
                      center_w-inner_mask_size:center_w+inner_mask_size] = 1
            
            middle_mask[center_h-middle_mask_size:center_h+middle_mask_size, 
                       center_w-middle_mask_size:center_w+middle_mask_size] = 1
            middle_mask -= inner_mask
            
            outer_mask[center_h-outer_mask_size:center_h+outer_mask_size, 
                      center_w-outer_mask_size:center_w+outer_mask_size] = 1
            outer_mask -= (inner_mask + middle_mask)
            
            # Calculate energy in each frequency band
            inner_energy = np.sum(magnitude * inner_mask) / np.sum(inner_mask) if np.sum(inner_mask) > 0 else 0
            middle_energy = np.sum(magnitude * middle_mask) / np.sum(middle_mask) if np.sum(middle_mask) > 0 else 0
            outer_energy = np.sum(magnitude * outer_mask) / np.sum(outer_mask) if np.sum(outer_mask) > 0 else 0
            
            # Ratio between bands is important for phone screen detection
            # Phone screens typically have high middle-to-inner ratio
            freq_ratio = middle_energy / (inner_energy + 1e-5)
            
            # 5. Pixel grid pattern detection
            small_rect_count = 0
            pixel_grid_factor = 0
            
            if face_img.shape[0] >= 100 and face_img.shape[1] >= 100:
                # Apply adaptive threshold to emphasize pixel grid
                thresh = cv2.adaptiveThreshold(
                    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
                
                # Find contours which can identify pixel grids
                contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                
                # Screens usually have many small rectangular contours
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if 2 <= w <= 8 and 2 <= h <= 8:  # Typical size for visible pixels
                        small_rect_count += 1
                
                # Normalize by image size
                pixel_grid_factor = small_rect_count / (face_img.shape[0] * face_img.shape[1]) * 10000
            
            # 6. NEW: Check for reflections with HSV analysis (for color images)
            reflection_score = 0
            if len(face_img.shape) == 3 and face_img.shape[2] == 3:
                # Convert to HSV to better detect screen reflections
                hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
                
                # Screens often have blue-ish reflections or highlights
                # Check for high value (brightness) with low saturation
                high_value_mask = (hsv[:,:,2] > 200)
                low_sat_mask = (hsv[:,:,1] < 50)
                
                # Combined mask for typical screen reflection pattern
                reflection_mask = high_value_mask & low_sat_mask
                
                # Calculate percentage of potential reflection pixels
                reflection_score = np.sum(reflection_mask) / (face_img.shape[0] * face_img.shape[1])
                reflection_score = min(reflection_score * 10, 1.0)  # Scale and cap
            
            # 7. NEW: Check for unnatural color distribution in RGB channels
            color_anomaly_score = 0
            if len(face_img.shape) == 3 and face_img.shape[2] == 3:
                # Calculate RGB channel means
                r_mean = np.mean(face_img[:,:,2])  # OpenCV uses BGR
                g_mean = np.mean(face_img[:,:,1])
                b_mean = np.mean(face_img[:,:,0])
                
                # Calculate ratios between channels
                # Phone screens often have unnatural RGB distribution
                rg_ratio = r_mean / (g_mean + 1e-5)
                rb_ratio = r_mean / (b_mean + 1e-5)
                gb_ratio = g_mean / (b_mean + 1e-5)
                
                # Check if ratios are in unusual ranges (outside typical face values)
                if rg_ratio > 1.3 or rg_ratio < 0.7 or rb_ratio > 1.4 or rb_ratio < 0.6:
                    color_anomaly_score = 0.5
                    
                # Screens often have more pronounced blue tint
                if b_mean > r_mean * 1.2:
                    color_anomaly_score += 0.3
            
            # Final combined scoring with updated weights and thresholds
            phone_score = (
                (laplacian_var > 250) * 0.2 +              # High edge content
                (uniformity > 0.65) * 0.15 +               # Uniform illumination
                (gradient_mean > 15) * 0.1 +               # Edge detection
                (freq_ratio > 1.8) * 0.2 +                 # Frequency band ratio
                (pixel_grid_factor > 1.5) * 0.15 +         # Pixel grid detection
                (reflection_score > 0.3) * 0.1 +           # Reflection detection
                color_anomaly_score * 0.1                  # Color anomalies
            )
            
            # Debug logging for suspicious scores
            if phone_score > 0.4 and self.debug:
                self._log(f"Phone score: {phone_score:.2f}, Components: laplacian_var={laplacian_var:.1f}, "
                         f"uniformity={uniformity:.2f}, grad_mean={gradient_mean:.1f}, "
                         f"freq_ratio={freq_ratio:.2f}, grid={pixel_grid_factor:.2f}, "
                         f"refl={reflection_score:.2f}, color={color_anomaly_score:.2f}")
            
            # Final decision with adjustable threshold
            phone_spoof_confidence = phone_score
            is_phone_spoof = phone_spoof_confidence > self.phone_pattern_threshold
            
            # Store information for temporal consistency
            if is_phone_spoof:
                self.consecutive_phone_detections += 1
            else:
                self.consecutive_phone_detections = max(0, self.consecutive_phone_detections - 1)
                
            # Require at least 2 consecutive phone detections to reduce false positives
            is_confident_phone_spoof = self.consecutive_phone_detections >= 2
            
            return is_phone_spoof
            
        except Exception as e:
            logger.error(f"Error in phone spoof detection: {e}")
            return False
    
    def _get_current_threshold(self):
        """Get the threshold based on current operation mode"""
        if self.current_mode == "registration":
            return self.registration_threshold
        elif self.current_mode == "strict":
            return self.strict_threshold
        else:
            return self.live_confidence_threshold
    
    def set_mode(self, mode):
        """Set the operation mode: 'normal', 'registration', or 'strict'"""
        if mode in ["normal", "registration", "strict"]:
            old_mode = self.current_mode
            self.current_mode = mode
            
            # Reset history when changing modes to avoid invalid decisions
            if old_mode != mode:
                self.recent_predictions = []
                self.recent_live_probs = []
                
            logger.info(f"Anti-spoofing mode set to: {mode}")
            
            # Adjust phone detection sensitivity based on mode
            if mode == "registration":
                self.phone_pattern_threshold = 0.85  # More lenient in registration mode
            elif mode == "strict":
                self.phone_pattern_threshold = 0.55  # More strict in strict mode
            else:
                self.phone_pattern_threshold = 0.65  # Default value
                
            return True
        return False
    
    def check_with_extended_info(self, face_img):
        """
        Enhanced version that returns additional information
        Returns: (is_real, spoof_type, live_prob)
        """
        if self.bypass_mode:
            return True, 'Live', 1.0
            
        # Ensure model is loaded
        if not self._ensure_model_loaded():
            # Fail open if model can't be loaded
            return True, 'Live', 1.0
            
        try:
            # Preprocess face image
            preprocessed_face = self.preprocess_face(face_img)
            
            # Run inference
            outputs = self.session.run(None, {self.input_name: preprocessed_face})
            raw_output = outputs[0]
            
            # Apply softmax for probabilities
            exp_out = np.exp(raw_output - np.max(raw_output))
            probabilities = exp_out / exp_out.sum()
            
            # Get live probability
            live_prob = float(probabilities[0][0])
            
            # Store for temporal consistency
            self.recent_live_probs.append(live_prob)
            if len(self.recent_live_probs) > self.decision_frames:
                self.recent_live_probs.pop(0)
            
            # Use average of recent probabilities for more stable results
            if len(self.recent_live_probs) > 1:
                # Weight recent frames more heavily
                weights = np.linspace(0.5, 1.0, len(self.recent_live_probs))
                weights = weights / np.sum(weights)
                avg_live_prob = np.average(self.recent_live_probs, weights=weights)
            else:
                avg_live_prob = live_prob
            
            # Get threshold based on current mode
            threshold = self._get_current_threshold()
            
            # Make decision based on threshold and smoothed probability
            is_real = avg_live_prob >= threshold
            
            # IMPORTANT FIX: Always check for phone spoofing regardless of mode
            # This ensures consistent detection between login and test modes
            is_phone_spoof = self._detect_phone_spoof_patterns(face_img)
            
            # If phone spoofing is detected with high confidence, override the decision
            if is_phone_spoof:
                # Only override if we're fairly confident about the phone detection
                if self.consecutive_phone_detections >= 2:
                    is_real = False
                    return False, 'Phone Screen', live_prob
            
            # Store decision for temporal consistency
            self.recent_predictions.append(is_real)
            if len(self.recent_predictions) > self.decision_frames:
                self.recent_predictions.pop(0)
            
            # Special handling for registration mode to reduce false positives
            if self.current_mode == "registration":
                # If most frames are real, consider it real to reduce false positives
                if sum(self.recent_predictions) > len(self.recent_predictions) // 2:
                    is_real = True
                    
                # In registration mode, give benefit of the doubt when close to threshold
                elif 0.8 * threshold <= avg_live_prob < threshold:
                    is_real = True
                    logger.info(f"Registration leniency applied: {avg_live_prob:.3f} considered real")
            
            # Get spoof prediction index (highest probability after live)
            if not is_real and len(probabilities[0]) > 1:
                # Skip the 'live' class (index 0)
                spoof_idx = np.argmax(probabilities[0][1:]) + 1
                if spoof_idx < len(SPOOF_TYPE):
                    spoof_type = SPOOF_TYPE[spoof_idx]
                else:
                    spoof_type = 'Fake'
            else:
                spoof_type = 'Live' if is_real else 'Fake'
            
            return is_real, spoof_type, live_prob
        
        except Exception as e:
            logger.error(f"Error in anti-spoofing check: {e}")
            return True, 'Error', 1.0
    
    def check_with_quality(self, face_img):
        """
        Check with face quality assessment
        Returns: (is_real, confidence, quality)
        """
        is_real, spoof_type, live_prob = self.check_with_extended_info(face_img)
        
        # Calculate simple quality metric based on image properties
        quality = 0.0
        try:
            # Simple brightness and contrast check - vectorized operations
            if len(face_img.shape) == 3 and face_img.shape[2] == 3:
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_img
                
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Add focus measure using Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            focus_score = min(np.var(laplacian) / 500.0, 1.0)
            
            # Normalize quality to [0,1]
            # Higher for balanced brightness (not too dark, not too bright)
            # and good contrast
            brightness_score = 1.0 - 2.0 * abs(brightness - 127.5) / 255.0
            contrast_score = min(contrast / 80.0, 1.0)  # Normalize contrast
            
            quality = (brightness_score * 0.4 + contrast_score * 0.3 + focus_score * 0.3)
        except Exception as e:
            logger.error(f"Error calculating face quality: {e}")
            quality = 0.5  # Default value on error
            
        return is_real, live_prob, quality
    
    def set_sensitivity(self, threshold):
        """Set the sensitivity threshold for live detection"""
        self.live_confidence_threshold = max(0.1, min(0.9, float(threshold)))
        
        # Also update other thresholds proportionally
        self.registration_threshold = max(0.1, min(0.7, self.live_confidence_threshold - 0.2))
        self.strict_threshold = max(0.3, min(0.9, self.live_confidence_threshold + 0.2))
        
        logger.info(f"Anti-spoofing thresholds updated: normal={self.live_confidence_threshold:.2f}, "
                  f"registration={self.registration_threshold:.2f}, "
                  f"strict={self.strict_threshold:.2f}")
                  
        return self.live_confidence_threshold
    
    def set_safe_mode(self, enabled):
        """Set safe mode on/off"""
        self.safe_mode = enabled
        self._log(f"Anti-spoofing safe mode: {'ENABLED' if enabled else 'DISABLED'}")
        return True
    
    def set_bypass_mode(self, enabled):
        """
        Set bypass mode on/off. When enabled, always authenticates as real face.
        THIS IS FOR DEBUGGING ONLY - INSECURE!
        """
        self.bypass_mode = enabled
        self._log(f"ANTI-SPOOFING BYPASS: {'ENABLED' if enabled else 'DISABLED'}")
        if enabled:
            logger.warning("WARNING: Anti-spoofing bypass is active - all faces will be authenticated!")
        return True
    
    def reset_history(self):
        """Reset all temporal history - useful when changing users or after long pauses"""
        self.recent_predictions = []
        self.recent_live_probs = []
        self.consecutive_phone_detections = 0
        return True
        
    def __del__(self):
        """Clean up resources when object is destroyed"""
        # Release session (though ONNX runtime should handle this automatically)
        self.session = None
        # Clear preprocessing cache
        self._preprocess_cache.clear()
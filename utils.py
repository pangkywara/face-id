# p:\Python\PROJECT PKM_KC\face_new\utils.py
import cv2
import numpy as np
import os
import time
import platform
import logging
from threading import Lock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("face_auth.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("face_auth")

# Mutex for thread-safe operations
io_lock = Lock()

def preprocess_face(face_img, target_size=(112, 112)):  # Updated size for MobileFaceNet
    """Preprocess face image for recognition"""
    # Resize
    face = cv2.resize(face_img, target_size)
    
    # Convert to RGB if needed
    if len(face.shape) == 3 and face.shape[2] == 3:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    
    return face

def adjust_brightness_contrast(img, brightness=0, contrast=0):
    """Adjust brightness and contrast of an image"""
    # Apply brightness
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        img = cv2.addWeighted(img, alpha_b, img, 0, gamma_b)
    
    # Apply contrast
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        img = cv2.addWeighted(img, alpha_c, img, 0, gamma_c)
    
    return img

def get_available_cameras():
    """
    Check for available camera devices with optimized performance
    Returns a list of available camera indices
    """
    available_cameras = []
    max_cameras_to_check = 5  # Limit how many cameras to check
    
    for i in range(max_cameras_to_check):
        try:
            # Use DirectShow on Windows for faster initialization
            backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
            
            # Open with minimal settings for faster detection
            cap = cv2.VideoCapture(i, backend)
            
            # Skip property setting to speed up detection
            if cap.isOpened():
                # Fast check - just grab a frame without decoding
                ret = cap.grab()
                if ret:
                    available_cameras.append(i)
                cap.release()
                
                # Don't wait between checks
        except Exception as e:
            logger.warning(f"Error checking camera {i}: {e}")
    
    # If no cameras detected, try fallback method on Windows
    if not available_cameras and os.name == 'nt':
        try:
            # Try alternative methods on Windows
            for i in range(max_cameras_to_check):
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    available_cameras.append(i)
                cap.release()
        except:
            pass
    
    return available_cameras

def enhance_face_image(face_img):
    """
    Apply image enhancement techniques to improve face image quality
    Useful for low-light conditions or low-quality cameras
    """
    if face_img is None or face_img.size == 0:
        return None
        
    try:
        # Convert to grayscale for processing if it's color
        is_color = len(face_img.shape) == 3 and face_img.shape[2] == 3
        if is_color:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img.copy()
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        
        # If original was color, convert back
        if is_color:
            # Use the enhanced grayscale as luminance and keep original color information
            hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
            hsv[:,:,2] = enhanced_gray  # Replace V channel with enhanced luminance
            enhanced_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            return enhanced_img
        else:
            return enhanced_gray
    except Exception as e:
        logger.error(f"Error enhancing face image: {e}")
        return face_img

def estimate_system_performance():
    """
    Estimate the system performance to adapt processing parameters
    Returns a performance score (0-100) with higher being better
    """
    score = 50  # Default middle score
    try:
        # Check CPU count
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        score += min(25, cpu_count * 5)  # Up to 25 points for CPUs
        
        # Simple performance test
        start_time = time.time()
        # Create a test image and perform common operations
        test_img = np.zeros((640, 480, 3), dtype=np.uint8)
        for _ in range(20):
            cv2.GaussianBlur(test_img, (5, 5), 0)
            cv2.Canny(test_img, 100, 200)
        
        elapsed = time.time() - start_time
        # Adjust score based on speed (lower time = higher score)
        if elapsed < 0.1:
            score += 25
        elif elapsed < 0.5:
            score += 15
        elif elapsed < 1.0:
            score += 5
            
        # Check available memory
        import psutil
        memory_info = psutil.virtual_memory()
        if memory_info.available > 4 * 1024 * 1024 * 1024:  # 4GB
            score += 25
        elif memory_info.available > 2 * 1024 * 1024 * 1024:  # 2GB
            score += 15
        elif memory_info.available > 1 * 1024 * 1024 * 1024:  # 1GB
            score += 5
            
    except Exception as e:
        logger.warning(f"Error in performance estimation: {e}")
        
    # Ensure score is in valid range
    return max(0, min(100, score))

def safe_imread(file_path):
    """Thread-safe image reading with proper error handling"""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
        
    try:
        with io_lock:
            img = cv2.imread(file_path)
        return img
    except Exception as e:
        logger.error(f"Error reading image {file_path}: {e}")
        return None
        
def safe_imwrite(file_path, img):
    """Thread-safe image writing with proper error handling"""
    try:
        # Ensure directory exists
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            
        with io_lock:
            success = cv2.imwrite(file_path, img)
        return success
    except Exception as e:
        logger.error(f"Error writing image {file_path}: {e}")
        return False

def get_optimal_settings():
    """Get optimized settings based on system performance"""
    performance_score = estimate_system_performance()
    logger.info(f"System performance score: {performance_score}/100")
    
    # Default settings for medium performance
    settings = {
        'face_detection_interval': 2,  # Process every 2nd frame
        'use_enhanced_detection': False,
        'camera_resolution': (640, 480),
        'camera_fps': 30,
        'recognition_batch_size': 1,
        'enable_multiprocessing': False,
        'detection_scale_factor': 1.1,
        'detection_min_neighbors': 5,
        'detection_min_size': (30, 30)
    }
    
    # Adjust based on performance score
    if performance_score >= 75:  # High-performance system
        settings.update({
            'face_detection_interval': 1,  # Process every frame
            'use_enhanced_detection': True,
            'camera_resolution': (1280, 720),
            'camera_fps': 30,
            'recognition_batch_size': 4,
            'enable_multiprocessing': True,
            'detection_min_neighbors': 3
        })
    elif performance_score <= 25:  # Low-performance system
        settings.update({
            'face_detection_interval': 3,  # Process every 3rd frame
            'use_enhanced_detection': False,
            'camera_resolution': (320, 240),
            'camera_fps': 15,
            'recognition_batch_size': 1,
            'enable_multiprocessing': False,
            'detection_scale_factor': 1.2,
            'detection_min_neighbors': 7,
            'detection_min_size': (40, 40)
        })
        
    return settings
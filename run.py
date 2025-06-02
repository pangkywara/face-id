import os
import sys

# Set environment variables to suppress TensorFlow warnings and optimize startup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=debug, 1=info, 2=warning, 3=error
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage for faster startup

from main import FaceAuthSystem
import tkinter as tk
import cv2
import numpy as np
import argparse
import time
import logging
import traceback
import platform

from anti_spoofing import AntiSpoofing
from face_detector import FaceDetector
from utils import get_optimal_settings, logger, get_available_cameras

def check_models_exist():
    """Check if required model files exist"""
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    
    # Check if the model directory exists
    if not os.path.exists(model_dir):
        logger.error("Models directory not found")
        return False
        
    # Check for anti-spoofing model
    antispoofing_path = os.path.join(model_dir, 'AntiSpoofing_bin_128.onnx')
    if not os.path.exists(antispoofing_path):
        logger.error("Anti-spoofing model not found")
        return False
        
    # Check for face recognition model
    facenet_path = os.path.join(model_dir, 'MobileFaceNet_9925_9680.tflite')
    if not os.path.exists(facenet_path):
        logger.error("Face recognition model not found")
        return False
        
    return True

def download_models():
    """Run the model download script"""
    logger.info("Running model download script...")
    try:
        import download_models
        download_models.main()
        return True
    except Exception as e:
        logger.error(f"Error running download script: {e}")
        logger.error(traceback.format_exc())
        return False

def run_main_app():
    """Run the main face authentication app with error handling"""
    # First check if models exist
    if not check_models_exist():
        logger.info("Required models not found, attempting to download...")
        if not download_models():
            logger.error("Failed to download required models")
            print("\nFailed to download required models.")
            print("Please run 'python download_models.py' manually to download required models.")
            return False
    
    # Get optimal settings based on system performance
    optimal_settings = get_optimal_settings()
    logger.info(f"Using settings: {optimal_settings}")
    
    # Check if any cameras are available
    available_cameras = get_available_cameras()
    if not available_cameras:
        logger.error("No cameras detected. Please connect a camera and try again.")
        print("No cameras detected. Please connect a camera and try again.")
        return False
        
    logger.info(f"Available cameras: {available_cameras}")
    
    try:
        # Create and run the application
        root = tk.Tk()
        app = FaceAuthSystem(root)
        root.protocol("WM_DELETE_WINDOW", lambda: on_closing(root))
        root.mainloop()
        return True
    except Exception as e:
        logger.error(f"Application error: {e}")
        logger.error(traceback.format_exc())
        print(f"Error running application: {e}")
        return False
        
def on_closing(root):
    """Handle application closing"""
    logger.info("Application closing")
    root.destroy()

def test_anti_spoofing():
    """Test the anti-spoofing module with webcam"""
    logger.info("Starting anti-spoofing test mode")
    
    if not check_models_exist():
        logger.error("Required models not found")
        print("Required models not found. Please run download_models.py first.")
        return False
        
    # Get optimal settings
    optimal_settings = get_optimal_settings()
    
    # Initialize components
    try:
        anti_spoofing = AntiSpoofing()
        face_detector = FaceDetector()
        
        # Anti-spoofing confidence threshold
        threshold = 0.5
        
        # Open webcam with improved initialization
        available_cameras = get_available_cameras()
        if not available_cameras:
            logger.error("No cameras detected")
            print("No cameras detected. Please connect a camera and try again.")
            return False
            
        camera_index = available_cameras[0]  # Use first available camera
        resolution = optimal_settings['camera_resolution']
        fps = optimal_settings['camera_fps']
        
        logger.info(f"Opening camera {camera_index} with resolution {resolution} @ {fps}fps")
        
        # Improved camera initialization with prewarm
        backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
        cap = cv2.VideoCapture(camera_index, backend)
        
        # Disable auto adjustments for faster startup
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        
        # Start with low resolution for faster initialization
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        cap.set(cv2.CAP_PROP_FPS, 15)
        
        # Read a few frames to initialize camera (prewarm)
        for _ in range(5):
            cap.read()
        
        # Now set the desired resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        cap.set(cv2.CAP_PROP_FPS, fps)
        
        # Verify camera opened successfully
        if not cap.isOpened():
            logger.error("Failed to open camera")
            print("Error: Could not open camera.")
            return False
            
        print(f"Using camera {camera_index} with resolution {resolution}")
        print("Press 'q' to quit, '+'/'-' to adjust threshold")
        logger.info("Anti-spoofing test started")
        
        frame_count = 0
        process_interval = optimal_settings['face_detection_interval']
        
        # Display a window immediately to show responsiveness
        cv2.namedWindow('Anti-Spoofing Test', cv2.WINDOW_NORMAL)
        cv2.imshow('Anti-Spoofing Test', np.zeros((240, 320, 3), dtype=np.uint8))
        cv2.waitKey(1)  # Update display
        
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                # Try to recover quickly without waiting
                cap.release()
                cap = cv2.VideoCapture(camera_index, backend)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
                
                # Show blank frame while recovering
                cv2.imshow('Anti-Spoofing Test', np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8))
                cv2.waitKey(1)
                continue
            
            # Process only every few frames for better performance
            frame_count += 1
            if frame_count % process_interval != 0:
                # Still display the frame but skip processing
                cv2.imshow('Anti-Spoofing Test', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue
                
            # Create a copy for display
            display = frame.copy()
            
            # Detect faces with optimized parameters
            try:
                faces = face_detector.detect_faces(
                    frame, 
                    scale_factor=optimal_settings['detection_scale_factor'],
                    min_neighbors=optimal_settings['detection_min_neighbors'],
                    min_size=optimal_settings['detection_min_size']
                )
                
                # Process each face
                for (x, y, w, h) in faces:
                    # Extract face
                    face_img = frame[y:y+h, x:x+w]
                    
                    # Set colors
                    color = (0, 0, 255)  # Red = fake by default
                    text = "FAKE"
                    
                    try:
                        # Check for spoofing
                        is_real, spoof_type, live_prob = anti_spoofing.check_with_extended_info(face_img)
                        _, _, quality = anti_spoofing.check_with_quality(face_img)
                        
                        # Determine color and text based on result
                        if is_real:
                            color = (0, 255, 0)  # Green = real
                            text = "REAL"
                        
                        # Draw rectangle around face
                        cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
                        
                        # Add text with probability
                        cv2.putText(display, f"{text} {live_prob:.2f}", (x, y-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Add quality information
                        cv2.putText(display, f"Quality: {quality:.2f}", (x, y+h+20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # Add threshold information
                        cv2.putText(display, f"Threshold: {threshold:.2f}", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                    except Exception as e:
                        logger.error(f"Spoofing check error: {e}")
                        cv2.putText(display, f"Error: {str(e)[:20]}", (x, y-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            except Exception as e:
                logger.error(f"Face detection error: {e}")
                cv2.putText(display, f"Detection error: {str(e)[:30]}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add performance info
            fps_text = f"Processing every {process_interval} frame(s)"
            cv2.putText(display, fps_text, (10, display.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display the frame
            cv2.imshow('Anti-Spoofing Test', display)
            
            # Wait for key press
            key = cv2.waitKey(1) & 0xFF
            
            # Adjust threshold if + or - is pressed
            if key == ord('+') or key == ord('='):
                threshold = min(0.9, threshold + 0.05)
                anti_spoofing.set_sensitivity(threshold)
                logger.info(f"Threshold increased to {threshold:.2f}")
            elif key == ord('-') or key == ord('_'):
                threshold = max(0.1, threshold - 0.05)
                anti_spoofing.set_sensitivity(threshold)
                logger.info(f"Threshold decreased to {threshold:.2f}")
            
            # Exit if 'q' is pressed
            if key == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Anti-spoofing test ended")
        return True
    
    except Exception as e:
        logger.error(f"Test error: {e}")
        logger.error(traceback.format_exc())
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    # Configure command line arguments
    parser = argparse.ArgumentParser(description="Face Authentication System")
    parser.add_argument('--test-spoof', action='store_true', 
                        help='Run anti-spoofing test instead of main app')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--camera', type=int, default=None,
                        help='Specify camera index to use')
    args = parser.parse_args()
    
    # Set logger level if debug mode
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Log system info
    logger.info(f"Starting application - Python {sys.version}")
    logger.info(f"OpenCV version: {cv2.__version__}")
    logger.info(f"System: {platform.system()} {platform.release()}")
    
    try:
        if args.test_spoof:
            success = test_anti_spoofing()
        else:
            success = run_main_app()
            
        if not success:
            sys.exit(1)
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}")
        logger.critical(traceback.format_exc())
        print(f"Critical error: {e}")
        sys.exit(1)
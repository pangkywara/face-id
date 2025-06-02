# p:\Python\PROJECT PKM_KC\face_new\face_recognizer.py
import cv2
import numpy as np
import os
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import shutil
import threading
import time
from utils import logger, safe_imread, safe_imwrite

# Global model cache
MODEL_CACHE = {}

# Thread safety lock for model access
FACE_MODEL_LOCK = threading.Lock()

class FaceRecognizer:
    def __init__(self, model_path=None, use_cached=True, debug=False):
        # Flag to track initialization status
        self.initialized = False
        
        # Path to the model file
        if model_path is None:
            self.tflite_model_path = os.path.join(os.path.dirname(__file__), 'models', 'MobileFaceNet_9925_9680.tflite')
        else:
            self.tflite_model_path = model_path
            
        # Embedding size of the model
        self.embedding_size = 128
        
        # User database
        self.users = {}
        self.db_path = os.path.join(os.path.dirname(__file__), 'face_database', 'users.pkl')
        
        # Preprocessing cache
        self._preprocess_cache = {}
        self._cache_size_limit = 10
        
        # Enable contrast enhancement
        self.enhance_contrast = True
        
        # Debug flag for detailed logging
        self.debug = debug
        
        # Set up fallback values in case model loading fails
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.input_shape = (112, 112)
        
        # Initialize the model
        self._load_model()
        
        # Load the user database
        self._load_user_db()
    
    def _load_model(self):
        """Load the facial recognition model, caching to avoid repeats"""
        if not self.initialized:
            try:
                start_time = time.time()
                
                # Check for cached model to speed up load time
                with FACE_MODEL_LOCK:
                    if 'recognition_model' in MODEL_CACHE and MODEL_CACHE['recognition_model']:
                        self.interpreter = MODEL_CACHE['recognition_model']
                        load_time = time.time() - start_time
                        logger.info(f"Using cached face recognition model, loaded in {load_time:.2f}s")
                        
                        # Set up model details
                        self.input_details = self.interpreter.get_input_details()
                        self.output_details = self.interpreter.get_output_details()
                        self.input_shape = self.input_details[0]['shape'][1:3]
                        
                        self.initialized = True
                        return True
                
                # Model is not cached, initialize it
                model_path = os.path.join(os.path.dirname(__file__), 'models', 'MobileFaceNet_9925_9680.tflite')
                if not os.path.exists(model_path):
                    logger.error(f"Face recognition model not found at: {model_path}")
                    return False
                    
                # Initialize the interpreter with optimizations
                try:
                    # Suppress TensorFlow warnings during initialization
                    import warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UserWarning)
                        
                        # Use minimal initialization for speed - no threading yet
                        # This loads the model but doesn't allocate tensors yet (faster)
                        interpreter = tf.lite.Interpreter(
                            model_path=model_path,
                            num_threads=1
                        )
                        
                        # Just get the tensor details but don't allocate yet
                        input_details = interpreter.get_input_details()
                        output_details = interpreter.get_output_details()
                        
                        # Cache model even before allocation
                        with FACE_MODEL_LOCK:
                            MODEL_CACHE['recognition_model'] = interpreter
                        
                        # Set instance variables
                        self.interpreter = interpreter
                        self.input_details = input_details
                        self.output_details = output_details
                        self.input_shape = input_details[0]['shape'][1:3]
                        
                        # Mark as initialized but not fully prewarm
                        self.initialized = True
                        self.fully_warmed = False
                        
                        # Start tensor allocation and warming in background thread
                        threading.Thread(target=self._finish_model_initialization, daemon=True).start()
                        
                        return True
                        
                except Exception as e:
                    logger.error(f"Error initializing TFLite interpreter: {e}")
                    # Last fallback attempt with minimal options
                    interpreter = tf.lite.Interpreter(model_path=model_path)
                    
                    # Store in cache
                    with FACE_MODEL_LOCK:
                        MODEL_CACHE['recognition_model'] = interpreter
                    
                    # Set instance variables
                    self.interpreter = interpreter
                    self.input_details = interpreter.get_input_details()
                    self.output_details = interpreter.get_output_details()
                    self.input_shape = self.input_details[0]['shape'][1:3]
                    
                    # Allocate tensors (unavoidable in this fallback case)
                    self.interpreter.allocate_tensors()
                    self.fully_warmed = True
                    self.initialized = True
                    return True
                
            except Exception as e:
                logger.error(f"Error loading face recognition model: {e}")
                return False
        return True
        
    def _finish_model_initialization(self):
        """Complete model initialization in background thread"""
        try:
            # Allocate tensors (delayed from _load_model)
            self.interpreter.allocate_tensors()
            
            # Run a small warmup inference
            dummy_input = np.zeros((1, self.input_shape[0], self.input_shape[1], 3), dtype=np.float32)
            self.interpreter.set_tensor(self.input_details[0]['index'], dummy_input)
            self.interpreter.invoke()
            
            # Set optimal thread count
            try:
                self.interpreter.set_num_threads(min(4, os.cpu_count() or 1))
            except:
                pass
                
            # Mark as fully warmed
            self.fully_warmed = True
            logger.info("Face recognition model fully initialized")
        except Exception as e:
            logger.error(f"Error completing model initialization: {e}")
    
    def _preprocess_face(self, face_img, use_cache=True):
        """Preprocess face for the network with enhanced preprocessing for robustness"""
        # Check if we have valid input_shape, if not use default
        if not hasattr(self, 'input_shape') or self.input_shape is None:
            self.input_shape = (112, 112)  # Use standard face model size
        
        # Check cache
        if use_cache:
            img_hash = hash(face_img.tobytes())
            if img_hash in self._preprocess_cache:
                return self._preprocess_cache[img_hash]
        
        try:
            # Convert BGR to RGB if needed
            if len(face_img.shape) == 3 and face_img.shape[2] == 3:
                img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            else:
                img = face_img.copy()
            
            # ENHANCEMENT: Improved preprocessing pipeline
            # 1. Apply histogram equalization to improve contrast and handle different lighting
            if len(img.shape) == 3:
                # Convert to YUV and equalize Y channel (luminance)
                yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
                yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
                img = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
            else:
                img = cv2.equalizeHist(img)
            
            # 2. Apply adaptive contrast enhancement (for better feature extraction)
            if self.enhance_contrast:
                # CLAHE provides better results than simple histogram equalization
                if len(img.shape) == 3:
                    # Apply CLAHE to L channel in LAB color space
                    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    lab[:,:,0] = clahe.apply(lab[:,:,0])
                    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                else:
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    img = clahe.apply(img)
            
            # 3. Special handling for glasses detection and reflection reduction
            # Detect potential eyeglasses using gradient information
            if len(img.shape) == 3 and self.input_shape[0] >= 96:  # Only for higher res images
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                magnitude = np.sqrt(sobelx**2 + sobely**2)
                
                # The eye region typically falls in the middle-upper part of the face
                h, w = gray.shape
                eye_region = magnitude[int(h*0.2):int(h*0.5), int(w*0.1):int(w*0.9)]
                
                # Higher gradient in eye region often indicates glasses
                avg_gradient = np.mean(eye_region)
                has_glasses = avg_gradient > 30
                
                if has_glasses:
                    # If glasses detected, apply subtle deglare to reduce reflections
                    # This helps especially with tinted glasses
                    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                    # Reduce value (brightness) of high saturation areas in the eye region
                    eye_section = hsv[int(h*0.2):int(h*0.5), int(w*0.1):int(w*0.9)]
                    mask = (eye_section[:,:,1] > 150) & (eye_section[:,:,2] > 200)
                    if np.any(mask):
                        eye_section[:,:,2][mask] = eye_section[:,:,2][mask] * 0.8
                    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                    
                    # Log this information for debugging
                    if self.debug:
                        logger.debug(f"Glasses detected and processed (gradient: {avg_gradient:.1f})")
            
            # Resize to model input size
            if img.shape[0] != self.input_shape[0] or img.shape[1] != self.input_shape[1]:
                img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
            
            # Normalize pixel values to [-1, 1]
            preprocessed = img.astype(np.float32) / 127.5 - 1
            
            # Expand dimensions for model input [batch, height, width, channels]
            preprocessed = np.expand_dims(preprocessed, axis=0)
            
            # Store in cache if enabled
            if use_cache and len(self._preprocess_cache) < self._cache_size_limit:
                self._preprocess_cache[img_hash] = preprocessed
                
            return preprocessed
            
        except Exception as e:
            logger.error(f"Error preprocessing face: {e}")
            # Return a safe fallback (original image resized and normalized)
            try:
                img = cv2.resize(face_img, (self.input_shape[1], self.input_shape[0]))
                preprocessed = np.expand_dims(img.astype(np.float32) / 127.5 - 1, axis=0)
                return preprocessed
            except:
                # Last resort - create a blank image
                return np.zeros((1, self.input_shape[0], self.input_shape[1], 3), dtype=np.float32)
    
    def _load_user_db(self):
        """Load user database with error handling"""
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, 'rb') as f:
                    self.users = pickle.load(f)
                logger.info(f"Loaded {len(self.users)} users from database")
            else:
                logger.info("User database not found. Creating a new one.")
                self.users = {}  # Dictionary to store user names and their embedding information
        except Exception as e:
            logger.error(f"Error loading user database: {e}")
            self.users = {}
            
    def extract_embedding(self, face_img):
        """Extract embedding vector from face image using TFLite model with optimizations"""
        try:
            # Check if model is loaded
            if not hasattr(self, 'interpreter') or self.interpreter is None:
                logger.warning("Face recognition model not loaded or not found")
                return np.zeros(self.embedding_size)
            
            # Wait for full model initialization if needed (blocks only when necessary)
            if not getattr(self, 'fully_warmed', False):
                wait_start = time.time()
                wait_count = 0
                while not getattr(self, 'fully_warmed', False) and wait_count < 10:
                    time.sleep(0.1)
                    wait_count += 1
                    
                if not getattr(self, 'fully_warmed', False):
                    logger.warning("Timeout waiting for face recognition model initialization")
            
            # Process the face image
            preprocessed = self._preprocess_face(face_img)
            
            # Set the input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], preprocessed)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output tensor
            embedding = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            
            # Normalize the embedding
            embedding_norm = np.linalg.norm(embedding)
            if embedding_norm > 0:
                embedding = embedding / embedding_norm
            
            return embedding
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            # Return a dummy embedding instead of raising an error
            return np.zeros(self.embedding_size)
        
    def register_new_user(self, name, face_images):
        """Register a new user with multiple face images"""
        # Create a directory for this user if it doesn't exist
        user_dir = os.path.join(os.path.dirname(self.db_path), name)
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)
        
        # Store reference embeddings for this user in background
        embeddings = []
        
        # Process each face
        for i, face_img in enumerate(face_images):
            # Extract and store embedding
            embedding = self.extract_embedding(face_img)
            embeddings.append(embedding)
            
            # Save the face image in background thread
            img_path = os.path.join(user_dir, f"face_{i+1}.jpg")
            threading.Thread(target=safe_imwrite, args=(img_path, face_img)).start()
        
        # Store user data
        self.users[name] = {
            'embeddings': embeddings,
            'face_count': len(face_images)
        }
        
        # Save the updated user database in background
        threading.Thread(target=self._save_user_db).start()
        
        logger.info(f"User {name} registered with {len(face_images)} face images")
        return True
    
    def _save_user_db(self):
        """Save user database with error handling"""
        try:
            with open(self.db_path, 'wb') as f:
                pickle.dump(self.users, f)
        except Exception as e:
            logger.error(f"Error saving user database: {e}")
            
    def recognize(self, face_img, threshold=0.7):
        """
        Recognize a face against the database - improved for reliability
        Returns: (name, confidence) or (None, 0) if no match
        """
        if face_img is None or face_img.size == 0:
            logger.warning("Empty face image provided to recognizer")
            return None, 0
            
        if not self.users or len(self.users) == 0:
            logger.warning("Face recognition database is empty")
            return None, 0
            
        try:
            # Extract embedding for the input face
            embedding = self.extract_embedding(face_img)
            if embedding is None or np.all(embedding == 0):
                logger.error("Failed to extract embedding for recognition")
                return None, 0
                
            # Track best match
            best_match_name = None
            best_match_score = 0.0
            
            # Check each user in the database
            for user_id, user_data in self.users.items():
                # Skip if user data is invalid
                if not isinstance(user_data, dict):
                    continue
                    
                # Get embeddings safely
                user_embeddings = user_data.get('embeddings', [])
                if not user_embeddings or len(user_embeddings) == 0:
                    continue
                
                # Calculate similarities for each embedding
                user_scores = []
                for ref_embedding in user_embeddings:
                    # Skip invalid embeddings
                    if ref_embedding is None or len(ref_embedding) != len(embedding):
                        continue
                    
                    # Calculate cosine similarity
                    similarity = np.dot(embedding, ref_embedding)
                    if similarity > 0:  # Avoid negative similarities
                        user_scores.append(similarity)
                
                # Get highest score for this user
                if user_scores:
                    user_max_score = max(user_scores)
                    
                    # Update best match if better than current best
                    if user_max_score > best_match_score:
                        best_match_score = user_max_score
                        best_match_name = user_id
            
            # Check if best match exceeds threshold
            if best_match_score >= threshold:
                return best_match_name, float(best_match_score)
            else:
                return None, float(best_match_score)
                
        except Exception as e:
            logger.error(f"Error in face recognition: {e}")
            return None, 0
    
    def get_user_faces(self, name):
        """Get all stored face images for a user with optimized loading"""
        user_dir = os.path.join(os.path.dirname(self.db_path), name)
        if not os.path.exists(user_dir):
            return []
            
        face_paths = [os.path.join(user_dir, f) for f in os.listdir(user_dir) if f.endswith('.jpg')]
        
        # Using optimized image loading with error handling
        faces = []
        for path in face_paths:
            img = safe_imread(path)
            if img is not None:
                faces.append(img)
        
        return faces
    
    def delete_user(self, name):
        """Delete a user and their face images from the database"""
        if name in self.users:
            # Remove user from dictionary
            del self.users[name]
            
            # Save the updated user database in background
            threading.Thread(target=self._save_user_db).start()
            
            # Delete user's face directory in background
            user_dir = os.path.join(os.path.dirname(self.db_path), name)
            if os.path.exists(user_dir):
                threading.Thread(target=lambda: shutil.rmtree(user_dir, ignore_errors=True)).start()
                
            return True
        return False
        
    def __del__(self):
        """Clean up resources when object is destroyed"""
        # No need to explicitly clean up TFLite interpreter
        pass

    def repair_user_database(self):
        """
        Repair potentially corrupted user database by:
        1. Validating user data structure
        2. Checking for empty/corrupt embeddings
        3. Removing invalid users
        
        Returns (fixed_count, removed_count, total_users)
        """
        if not self._load_user_db():
            logger.error("Cannot repair user database: failed to load database file")
            return 0, 0, 0
        
        if not self.users:
            logger.warning("User database is empty, nothing to repair")
            return 0, 0, 0
            
        fixed_count = 0
        removed_count = 0
        initial_user_count = len(self.users)
        
        # Create a copy to iterate while modifying the original
        users_to_check = list(self.users.items())
        
        for user_id, user_data in users_to_check:
            # Check if user_data is a dictionary
            if not isinstance(user_data, dict):
                logger.warning(f"User {user_id} has invalid data format, removing")
                del self.users[user_id]
                removed_count += 1
                continue
                
            # Check if embeddings exist
            if 'embeddings' not in user_data or not isinstance(user_data['embeddings'], list):
                logger.warning(f"User {user_id} has no embeddings array, removing")
                del self.users[user_id]
                removed_count += 1
                continue
                
            # Filter out None or invalid embeddings
            valid_embeddings = []
            embedding_count = len(user_data['embeddings'])
            
            for embedding in user_data['embeddings']:
                # Check if embedding is valid
                if embedding is not None and isinstance(embedding, (list, np.ndarray)) and len(embedding) == self.embedding_size:
                    # Convert to numpy array if it's a list
                    if isinstance(embedding, list):
                        embedding = np.array(embedding, dtype=np.float32)
                    valid_embeddings.append(embedding)
            
            # If we lost any embeddings, update the user data
            if len(valid_embeddings) != embedding_count:
                # Check if we have any valid embeddings left
                if valid_embeddings:
                    logger.info(f"Fixed user {user_id}: kept {len(valid_embeddings)} of {embedding_count} embeddings")
                    self.users[user_id]['embeddings'] = valid_embeddings
                    fixed_count += 1
                else:
                    logger.warning(f"User {user_id} has no valid embeddings, removing")
                    del self.users[user_id]
                    removed_count += 1
        
        # Save the database if any changes were made
        if fixed_count > 0 or removed_count > 0:
            self._save_user_db()
            logger.info(f"User database repaired: {fixed_count} users fixed, {removed_count} users removed")
        else:
            logger.info("User database verified: no issues found")
            
        return fixed_count, removed_count, len(self.users)

    def get_user_ids(self):
        """Get a list of all user IDs in the database"""
        try:
            # Load user database if not already loaded
            if self.users is None:
                self._load_user_db()
                
            # Return user IDs (names)
            return list(self.users.keys())
        except Exception as e:
            logger.error(f"Error getting user IDs: {e}")
            return []
import cv2
import mediapipe as mp
import numpy as np
import os
import joblib
import time
import argparse
import glob
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='ASL Sign Language Detection using Trained Models')
    parser.add_argument('--models_dir', type=str, default='./webcam_models',
                        help='Directory containing trained models')
    parser.add_argument('--camera_index', type=int, default=1,
                        help='Camera index to use (default: 1)')
    parser.add_argument('--confidence_threshold', type=float, default=0.7,
                        help='Confidence threshold for detection (default: 0.7)')
    parser.add_argument('--use_preprocessing', action='store_true',
                        help='Apply preprocessing to images')
    parser.add_argument('--hand', choices=['right', 'left', 'both'], default='both',
                        help='Which hand to use for detection (default: both)')
    parser.add_argument('--target_letter', type=str, default='a', choices=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'],
                        help='Target letter to practice forming (default: a)')
    return parser.parse_args()

def preprocess_image(image):
    """Apply preprocessing to improve hand detection."""
    # Resize to a standard size
    image = cv2.resize(image, (640, 480))
    
    # Apply contrast enhancement
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Apply bilateral filter to smooth while preserving edges
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    return filtered

def extract_features(landmarks, is_right_hand):
    """Extract features from hand landmarks for the model."""
    # Flatten coordinates into a 1D array
    features = []
    
    # Add a feature to indicate hand type (0 for left, 1 for right)
    features.append(1.0 if is_right_hand else 0.0)
    
    # Add all raw landmarks (x, y, z for each of the 21 landmarks)
    for landmark in landmarks:
        features.extend([landmark.x, landmark.y, landmark.z])
    
    # Add calculated features (distances between key points)
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    wrist = landmarks[0]
    
    # Distance between thumb and index finger
    thumb_index_dist = np.sqrt(
        (thumb_tip.x - index_tip.x)**2 + 
        (thumb_tip.y - index_tip.y)**2 +
        (thumb_tip.z - index_tip.z)**2
    )
    features.append(thumb_index_dist)
    
    # Distance between thumb and side of hand
    thumb_side_dist = np.sqrt(
        (thumb_tip.x - landmarks[17].x)**2 + 
        (thumb_tip.y - landmarks[17].y)**2 +
        (thumb_tip.z - landmarks[17].z)**2
    )
    features.append(thumb_side_dist)
    
    # Check if fingers are closed (making a fist)
    index_closed = 1 if index_tip.y > landmarks[5].y else 0
    middle_closed = 1 if middle_tip.y > landmarks[9].y else 0
    ring_closed = 1 if ring_tip.y > landmarks[13].y else 0
    pinky_closed = 1 if pinky_tip.y > landmarks[17].y else 0
    
    features.extend([index_closed, middle_closed, ring_closed, pinky_closed])
    
    # Add relative positions of fingers to wrist
    for i in [4, 8, 12, 16, 20]:  # Fingertips
        features.append(landmarks[i].x - wrist.x)
        features.append(landmarks[i].y - wrist.y)
        features.append(landmarks[i].z - wrist.z)
    
    return features

def get_asl_position_feedback(letter, landmarks):
    """Analyze landmarks to provide correction feedback for specific ASL signs."""
    feedback = {"correct": False, "message": "", "finger_issues": []}
    
    if letter == 'a':
        # Check if fingers are closed (making a fist)
        if landmarks[8].y < landmarks[5].y:  # Index finger not closed
            feedback["finger_issues"].append("Close your index finger into a fist")
        if landmarks[12].y < landmarks[9].y:  # Middle finger not closed
            feedback["finger_issues"].append("Close your middle finger into a fist")
        if landmarks[16].y < landmarks[13].y:  # Ring finger not closed
            feedback["finger_issues"].append("Close your ring finger into a fist")
        if landmarks[20].y < landmarks[17].y:  # Pinky not closed
            feedback["finger_issues"].append("Close your pinky finger into a fist")
            
        # Check thumb position
        thumb_tip = landmarks[4]
        index_mcp = landmarks[5]  # Base of index finger
        thumb_distance = np.sqrt(
            (thumb_tip.x - index_mcp.x)**2 + 
            (thumb_tip.y - index_mcp.y)**2 +
            (thumb_tip.z - index_mcp.z)**2
        )
        
        if thumb_tip.x < index_mcp.x:
            feedback["finger_issues"].append("Keep your thumb alongside your fingers, not across them")
        elif thumb_distance > 0.08:
            feedback["finger_issues"].append("Thumb should touch the side of your index finger")
        
        # Set overall message
        if not feedback["finger_issues"]:
            feedback["correct"] = True
            feedback["message"] = "Perfect 'A' sign!"
        else:
            feedback["message"] = "Adjust your 'A' sign:"
            
    elif letter == 'b':
        # Check if fingers are straight and extended
        if landmarks[8].y > landmarks[5].y:  # Index finger not extended
            feedback["finger_issues"].append("Extend your index finger upward")
        if landmarks[12].y > landmarks[9].y:  # Middle finger not extended
            feedback["finger_issues"].append("Extend your middle finger upward")
        if landmarks[16].y > landmarks[13].y:  # Ring finger not extended
            feedback["finger_issues"].append("Extend your ring finger upward")
        if landmarks[20].y > landmarks[17].y:  # Pinky not extended
            feedback["finger_issues"].append("Extend your pinky finger upward")
            
        # Check finger separation
        if abs(landmarks[8].x - landmarks[12].x) > 0.1:
            feedback["finger_issues"].append("Keep your fingers together")
            
        # Check thumb position
        thumb_tip = landmarks[4]
        palm_center_x = (landmarks[0].x + landmarks[9].x) / 2
        if thumb_tip.x > palm_center_x:
            feedback["finger_issues"].append("Tuck your thumb against your palm")
        
        # Set overall message
        if not feedback["finger_issues"]:
            feedback["correct"] = True
            feedback["message"] = "Perfect 'B' sign!"
        else:
            feedback["message"] = "Adjust your 'B' sign:"
    
    elif letter == 'c':
        # Check if fingers are curved correctly
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        wrist = landmarks[0]
        
        # Check curvature - all fingers should be slightly curved
        fingers_too_straight = (
            abs(landmarks[8].x - landmarks[5].x) < 0.03 or  # Index finger
            abs(landmarks[12].x - landmarks[9].x) < 0.03 or  # Middle finger
            abs(landmarks[16].x - landmarks[13].x) < 0.03 or  # Ring finger
            abs(landmarks[20].x - landmarks[17].x) < 0.03    # Pinky
        )
        
        # Check finger separation - fingers should be together
        fingers_separated = (
            abs(landmarks[8].x - landmarks[12].x) > 0.05 or
            abs(landmarks[12].x - landmarks[16].x) > 0.05 or
            abs(landmarks[16].x - landmarks[20].x) > 0.05
        )
        
        # Check if thumb and index form a C-shape
        thumb_index_dist = np.sqrt(
            (thumb_tip.x - index_tip.x)**2 + 
            (thumb_tip.y - index_tip.y)**2
        )
        
        # Analyze positioning
        if fingers_too_straight:
            feedback["finger_issues"].append("Curve your fingers more to form a 'C' shape")
        if fingers_separated:
            feedback["finger_issues"].append("Keep your fingers closer together")
        if thumb_index_dist > 0.2 or thumb_index_dist < 0.1:
            feedback["finger_issues"].append("Adjust your thumb and index finger to form a clear 'C' shape")
            
        # Set overall message
        if not feedback["finger_issues"]:
            feedback["correct"] = True
            feedback["message"] = "Perfect 'C' sign!"
        else:
            feedback["message"] = "Adjust your 'C' sign:"
            
    elif letter == 'd':
        # Check if thumb and index finger form a circle
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        # Distance between thumb and index tips - should be small for the circle
        thumb_index_dist = np.sqrt(
            (thumb_tip.x - index_tip.x)**2 + 
            (thumb_tip.y - index_tip.y)**2 +
            (thumb_tip.z - index_tip.z)**2
        )
        
        # Check if other fingers are extended
        middle_extended = landmarks[12].y < landmarks[9].y
        ring_extended = landmarks[16].y < landmarks[13].y
        pinky_extended = landmarks[20].y < landmarks[17].y
        
        # Check if extended fingers are straight and together
        fingers_straight = (
            abs(landmarks[12].x - landmarks[9].x) < 0.05 and
            abs(landmarks[16].x - landmarks[13].x) < 0.05 and
            abs(landmarks[20].x - landmarks[17].x) < 0.05
        )
        
        fingers_together = (
            abs(landmarks[12].x - landmarks[16].x) < 0.05 and
            abs(landmarks[16].x - landmarks[20].x) < 0.05
        )
        
        # Analyze positioning
        if thumb_index_dist > 0.05:
            feedback["finger_issues"].append("Bring your thumb and index fingertips closer to form a circle")
        if not (middle_extended and ring_extended and pinky_extended):
            feedback["finger_issues"].append("Extend your middle, ring, and pinky fingers upward")
        if not fingers_straight:
            feedback["finger_issues"].append("Keep your extended fingers straight")
        if not fingers_together:
            feedback["finger_issues"].append("Keep your extended fingers together")
            
        # Set overall message
        if not feedback["finger_issues"]:
            feedback["correct"] = True
            feedback["message"] = "Perfect 'D' sign!"
        else:
            feedback["message"] = "Adjust your 'D' sign:"
            
    elif letter == 'e':
        # Check if all fingers are curled
        index_curled = landmarks[8].y > landmarks[6].y
        middle_curled = landmarks[12].y > landmarks[10].y
        ring_curled = landmarks[16].y > landmarks[14].y
        pinky_curled = landmarks[20].y > landmarks[18].y
        
        # Check thumb position - should be against index finger
        thumb_tip = landmarks[4]
        index_base = landmarks[5]
        thumb_index_dist = np.sqrt(
            (thumb_tip.x - index_base.x)**2 + 
            (thumb_tip.y - index_base.y)**2 +
            (thumb_tip.z - index_base.z)**2
        )
        
        # Analyze positioning
        if not index_curled:
            feedback["finger_issues"].append("Curl your index finger into your palm")
        if not middle_curled:
            feedback["finger_issues"].append("Curl your middle finger into your palm")
        if not ring_curled:
            feedback["finger_issues"].append("Curl your ring finger into your palm")
        if not pinky_curled:
            feedback["finger_issues"].append("Curl your pinky finger into your palm")
        if thumb_index_dist > 0.08:
            feedback["finger_issues"].append("Tuck your thumb against your index finger")
            
        # Set overall message
        if not feedback["finger_issues"]:
            feedback["correct"] = True
            feedback["message"] = "Perfect 'E' sign!"
        else:
            feedback["message"] = "Adjust your 'E' sign:"
            
    elif letter == 'f':
        # Check if thumb and index form a circle
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        thumb_index_dist = np.sqrt(
            (thumb_tip.x - index_tip.x)**2 + 
            (thumb_tip.y - index_tip.y)**2 +
            (thumb_tip.z - index_tip.z)**2
        )
        
        # Check if other fingers are extended
        middle_extended = landmarks[12].y < landmarks[9].y
        ring_extended = landmarks[16].y < landmarks[13].y
        pinky_extended = landmarks[20].y < landmarks[17].y
        
        # Check if extended fingers are together
        fingers_together = (
            abs(landmarks[12].x - landmarks[16].x) < 0.05 and
            abs(landmarks[16].x - landmarks[20].x) < 0.05
        )
        
        # Analyze positioning
        if thumb_index_dist > 0.05:
            feedback["finger_issues"].append("Touch your thumb and index fingertips to form a circle")
        if not (middle_extended and ring_extended and pinky_extended):
            feedback["finger_issues"].append("Extend your middle, ring, and pinky fingers upward")
        if not fingers_together:
            feedback["finger_issues"].append("Keep your extended fingers together")
            
        # Set overall message
        if not feedback["finger_issues"]:
            feedback["correct"] = True
            feedback["message"] = "Perfect 'F' sign!"
        else:
            feedback["message"] = "Adjust your 'F' sign:"
            
    elif letter == 'g':
        # Check if index finger is extended forward and other fingers are closed
        index_extended = landmarks[8].x > landmarks[5].x
        middle_closed = landmarks[12].y > landmarks[9].y
        ring_closed = landmarks[16].y > landmarks[13].y
        pinky_closed = landmarks[20].y > landmarks[17].y
        
        # Check palm orientation - should be sideways
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        palm_orientation = abs(wrist.z - middle_mcp.z)
        
        # Check thumb position - should be alongside fist, not tucked
        thumb_tip = landmarks[4]
        thumb_position_good = thumb_tip.x > wrist.x
        
        # Analyze positioning
        if not index_extended:
            feedback["finger_issues"].append("Extend your index finger forward (not upward)")
        if not (middle_closed and ring_closed and pinky_closed):
            feedback["finger_issues"].append("Close your middle, ring, and pinky fingers into a fist")
        if palm_orientation < 0.05:
            feedback["finger_issues"].append("Turn your palm to face sideways")
        if not thumb_position_good:
            feedback["finger_issues"].append("Position your thumb alongside your fist, not tucked in")
            
        # Set overall message
        if not feedback["finger_issues"]:
            feedback["correct"] = True
            feedback["message"] = "Perfect 'G' sign!"
        else:
            feedback["message"] = "Adjust your 'G' sign:"
            
    elif letter == 'h':
        # Check if index and middle fingers are extended sideways and others closed
        index_extended = landmarks[8].x > landmarks[5].x
        middle_extended = landmarks[12].x > landmarks[9].x
        ring_closed = landmarks[16].y > landmarks[13].y
        pinky_closed = landmarks[20].y > landmarks[17].y
        
        # Check if index and middle are parallel and together
        fingers_parallel = abs((landmarks[8].y - landmarks[5].y) - (landmarks[12].y - landmarks[9].y)) < 0.05
        fingers_together = abs(landmarks[8].y - landmarks[12].y) < 0.05
        
        # Check palm orientation - should be sideways
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        palm_orientation = abs(wrist.z - middle_mcp.z)
        
        # Analyze positioning
        if not (index_extended and middle_extended):
            feedback["finger_issues"].append("Extend your index and middle fingers forward, not upward")
        if not (ring_closed and pinky_closed):
            feedback["finger_issues"].append("Close your ring and pinky fingers into a fist")
        if not fingers_parallel:
            feedback["finger_issues"].append("Keep your index and middle fingers parallel")
        if not fingers_together:
            feedback["finger_issues"].append("Keep your index and middle fingers together")
        if palm_orientation < 0.05:
            feedback["finger_issues"].append("Turn your palm to face sideways")
            
        # Set overall message
        if not feedback["finger_issues"]:
            feedback["correct"] = True
            feedback["message"] = "Perfect 'H' sign!"
        else:
            feedback["message"] = "Adjust your 'H' sign:"
            
    elif letter == 'i':
        # Check if only pinky is extended and others closed
        index_closed = landmarks[8].y > landmarks[5].y
        middle_closed = landmarks[12].y > landmarks[9].y
        ring_closed = landmarks[16].y > landmarks[13].y
        pinky_extended = landmarks[20].y < landmarks[17].y
        
        # Check palm orientation - should be sideways
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        palm_orientation = abs(wrist.z - middle_mcp.z)
        
        # Check thumb position - should rest across curled fingers
        thumb_tip = landmarks[4]
        thumb_position_good = thumb_tip.x > landmarks[9].x  # Check if thumb crosses middle finger base
        
        # Analyze positioning
        if not (index_closed and middle_closed and ring_closed):
            feedback["finger_issues"].append("Close your index, middle, and ring fingers into a fist")
        if not pinky_extended:
            feedback["finger_issues"].append("Extend your pinky finger upward")
        if palm_orientation < 0.05:
            feedback["finger_issues"].append("Turn your palm to face sideways")
        if not thumb_position_good:
            feedback["finger_issues"].append("Position your thumb across your curled fingers")
            
        # Set overall message
        if not feedback["finger_issues"]:
            feedback["correct"] = True
            feedback["message"] = "Perfect 'I' sign!"
        else:
            feedback["message"] = "Adjust your 'I' sign:"
    
    return feedback

def get_asl_instructions(letter):
    """Return instructions for how to form the given ASL letter."""
    instructions = {
        'a': [
            "Make a fist with your hand",
            "Keep your thumb against the side of your hand",
            "Your thumb should rest alongside your fingers, not across them",
            "Keep your palm facing forward"
        ],
        'b': [
            "Hold your hand up with palm facing forward",
            "Keep your fingers straight and together",
            "Tuck your thumb against your palm",
            "Your fingers should be pointing upward"
        ],
        'c': [
            "Curve your fingers and thumb to form a 'C' shape",
            "Keep all fingers together in the curved position",
            "Palm should face to the side",
            "Thumb and fingers should be aligned in the same curved plane"
        ],
        'd': [
            "Make a circle with your thumb and index finger",
            "Keep your middle, ring, and pinky fingers pointing up",
            "Palm should face forward",
            "The middle, ring, and pinky fingers should be straight and together"
        ],
        'e': [
            "Curl all fingers into the palm",
            "Tuck your thumb against the side of your index finger",
            "Keep your palm facing forward",
            "Your fingernails should be visible as you curl them"
        ],
        'f': [
            "Connect your thumb and index finger to form a circle",
            "Extend your other three fingers upward",
            "Keep your index finger and thumb touching at the tips",
            "Your remaining three fingers should be straight and together"
        ],
        'g': [
            "Make a fist with your hand, palm facing sideways",
            "Extend your index finger pointing forward",
            "Thumb should rest alongside your fist, not tucked in",
            "The index finger and thumb should form a 'G' shape"
        ],
        'h': [
            "Make a fist with your hand, palm facing sideways",
            "Extend your index and middle fingers forward together",
            "Keep your fingers parallel to the ground",
            "Thumb should rest alongside your fist, not tucked in"
        ],
        'i': [
            "Make a fist with your hand, palm facing sideways",
            "Extend only your pinky finger upward",
            "Keep the rest of your fingers curled into a fist",
            "Thumb should rest across the curled fingers"
        ]
    }
    
    return instructions.get(letter.lower(), ["No instructions available for this letter"])

def draw_semitransparent_rect(image, start_point, end_point, color, alpha=0.7):
    """Draw a semi-transparent rectangle on the image."""
    overlay = image.copy()
    cv2.rectangle(overlay, start_point, end_point, color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    return image

def setup_camera(camera_index):
    """Set up and return a camera capture object, trying multiple indices if needed."""
    print("Trying to initialize camera...")
    
    # Direct approach first
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        ret, test_frame = cap.read()
        if ret and test_frame is not None and test_frame.size > 0:
            print(f"Successfully opened camera at index {camera_index}")
            return cap
        cap.release()
    
    # Try common indices if direct approach failed
    for idx in [1, 2, 3]:
        if idx == camera_index:
            continue  # Already tried
        
        print(f"Trying camera index {idx}...")
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None and test_frame.size > 0:
                print(f"Successfully opened camera at index {idx}")
                return cap
            cap.release()
    
    print("Error: Could not open any camera")
    return None

def load_all_models(models_dir):
    """Load all trained models from models directory."""
    models = {}
    combined_model_path = None
    
    # First check for a combined model
    combined_model_paths = [
        os.path.join(models_dir, "asl_combined_model.joblib"),
        os.path.join(models_dir, "combined_model.joblib")
    ]
    
    for path in combined_model_paths:
        if os.path.exists(path):
            combined_model_path = path
            break
    
    # If we found a combined model, load it
    if combined_model_path:
        print(f"Found combined model at {combined_model_path}")
        try:
            model = joblib.load(combined_model_path)
            print("Successfully loaded combined model")
            print(f"Model type: {type(model).__name__}")
            
            # Check model feature count
            if hasattr(model, 'n_features_in_'):
                print(f"Combined model expects {model.n_features_in_} features")
            
            # Print model attributes for debugging
            print("Model attributes:")
            for attr in dir(model):
                if not attr.startswith('_'):
                    try:
                        value = getattr(model, attr)
                        if not callable(value):
                            print(f"  {attr}: {type(value)}")
                    except Exception as e:
                        print(f"  {attr}: Error getting value")
            
            # For combined models, use a special key
            models['combined'] = model
            
            # If the model has classes_ attribute, extract characters from it
            if hasattr(model, 'classes_'):
                print(f"Model can predict these characters: {', '.join(model.classes_)}")
            
            return models
        except Exception as e:
            print(f"Error loading combined model: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # If no combined model or loading failed, look for individual models
    model_files = glob.glob(os.path.join(models_dir, "asl_char_*_model.joblib"))
    
    if not model_files:
        # If no models found, try alternative pattern
        model_files = glob.glob(os.path.join(models_dir, "*.joblib"))
        # Filter out the combined model if we couldn't load it
        model_files = [f for f in model_files if "combined" not in os.path.basename(f).lower()]
    
    print(f"Loading {len(model_files)} individual models from {models_dir}...")
    
    for model_file in model_files:
        try:
            # Extract character from filename - simpler, more reliable approach
            filename = os.path.basename(model_file)
            if "_char_" in filename:
                # For files like asl_char_a_model.joblib
                char = filename.split("_char_")[1][0]  # Take just the first character
            else:
                # For other formats, extract character directly from filename
                for c in filename:
                    if c in "abcdefghijklmnopqrstuvwxyz0123456789":
                        char = c
                        break
                else:
                    # If no character found, skip this file
                    print(f"Could not extract character from filename: {filename}")
                    continue
            
            # Load model
            model = joblib.load(model_file)
            models[char] = model
            
            # Check model feature count
            if hasattr(model, 'n_features_in_'):
                print(f"Model '{char}' expects {model.n_features_in_} features")
            print(f"Loaded model for character: '{char}' from {os.path.basename(model_file)}")
        except Exception as e:
            print(f"Error loading model {model_file}: {str(e)}")
    
    return models

class ASLDetector:
    def __init__(self):
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Higher confidence for detection
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Model paths and settings
        self.models_dir = "./webcam_models"
        self.model = None
        self.model_classes = []
        
        # Target letter and hand settings
        self.target_letter = 'a'  # Default to letter 'a'
        self.target_hand = "right"  # Default to right hand
        self.confidence_threshold = 0.85  # Increase threshold to 0.95
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the combined ASL model."""
        # Check for a combined model
        combined_model_paths = [
            os.path.join(self.models_dir, "asl_combined_model.joblib"),
            os.path.join(self.models_dir, "combined_model.joblib")
        ]
        
        combined_model_path = None
        for path in combined_model_paths:
            if os.path.exists(path):
                combined_model_path = path
                break
        
        if combined_model_path:
            try:
                self.model = joblib.load(combined_model_path)
                print(f"Successfully loaded model from {combined_model_path}")
                if hasattr(self.model, 'classes_'):
                    self.model_classes = self.model.classes_
                    print(f"Model can predict: {', '.join(self.model_classes)}")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print(f"No model found in {self.models_dir}")
        
        return False
    
    def extract_features(self, landmarks, is_right_hand=True):
        """Extract features from hand landmarks for the model."""
        # Flatten coordinates into a 1D array
        features = []
        
        # Add a feature to indicate hand type (0 for left, 1 for right)
        features.append(1.0 if is_right_hand else 0.0)
        
        # Add all raw landmarks (x, y, z for each of the 21 landmarks)
        for landmark in landmarks:
            features.extend([landmark.x, landmark.y, landmark.z])
        
        # Add calculated features (distances between key points)
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        wrist = landmarks[0]
        
        # Distance between thumb and index finger
        thumb_index_dist = np.sqrt(
            (thumb_tip.x - index_tip.x)**2 + 
            (thumb_tip.y - index_tip.y)**2 +
            (thumb_tip.z - index_tip.z)**2
        )
        features.append(thumb_index_dist)
        
        # Distance between thumb and side of hand
        thumb_side_dist = np.sqrt(
            (thumb_tip.x - landmarks[17].x)**2 + 
            (thumb_tip.y - landmarks[17].y)**2 +
            (thumb_tip.z - landmarks[17].z)**2
        )
        features.append(thumb_side_dist)
        
        # Check if fingers are closed (making a fist)
        index_closed = 1 if index_tip.y > landmarks[5].y else 0
        middle_closed = 1 if middle_tip.y > landmarks[9].y else 0
        ring_closed = 1 if ring_tip.y > landmarks[13].y else 0
        pinky_closed = 1 if pinky_tip.y > landmarks[17].y else 0
        
        features.extend([index_closed, middle_closed, ring_closed, pinky_closed])
        
        # Add relative positions of fingers to wrist
        for i in [4, 8, 12, 16, 20]:  # Fingertips
            features.append(landmarks[i].x - wrist.x)
            features.append(landmarks[i].y - wrist.y)
            features.append(landmarks[i].z - wrist.z)
        
        return features
    
    def get_hand_type(self, hand_landmarks, results):
        """Determine if the hand is left or right."""
        handedness = results.multi_handedness
        for idx, classification in enumerate(handedness):
            if idx == results.multi_hand_landmarks.index(hand_landmarks):
                return classification.classification[0].label == "Right"
        return True  # Default to right if can't determine
    
    def toggle_target_hand(self):
        """Toggle between left and right hand."""
        self.target_hand = "left" if self.target_hand == "right" else "right"
        return self.target_hand
    
    def toggle_target_letter(self):
        """Toggle between letters A through I."""
        letter_cycle = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
        current_index = letter_cycle.index(self.target_letter)
        next_index = (current_index + 1) % len(letter_cycle)
        self.target_letter = letter_cycle[next_index]
        return self.target_letter
    
    def setup_camera(self):
        """Set up and return a camera capture object."""
        print("Setting up camera...")
        cap = cv2.VideoCapture(0)
        
        # Check if camera opened successfully
        if cap.isOpened():
            ret, test_frame = cap.read()
            if not ret or test_frame is None or test_frame.size == 0:
                print("Camera opened but couldn't read frames, trying alternatives...")
                cap.release()
                cap = None
        else:
            print("Failed to open camera, trying alternatives...")
            cap = None
        
        # Try other camera indices if the default failed
        if cap is None:
            for i in range(1, 5):
                print(f"Trying camera index {i}...")
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        print(f"Successfully opened camera {i}")
                        time.sleep(1)  # Give the camera time to initialize
                        return cap
                    cap.release()
            
            print("Failed to find a working camera")
            return None
        
        time.sleep(1)  # Give the camera time to initialize
        return cap
    
    def get_position_feedback_for_b(self, landmarks):
        """Special strict position checking for letter B to match confidence values."""
        error_text = "No feedback available"
        correct_position = False
        
        # Check if fingers are extended (straight up)
        fingers_extended = (
            landmarks[8].y < landmarks[5].y and  # Index finger
            landmarks[12].y < landmarks[9].y and  # Middle finger
            landmarks[16].y < landmarks[13].y and  # Ring finger
            landmarks[20].y < landmarks[17].y  # Pinky
        )
        
        # Check finger separation (should be close together)
        finger_separation_index_middle = abs(landmarks[8].x - landmarks[12].x)
        finger_separation_middle_ring = abs(landmarks[12].x - landmarks[16].x)
        finger_separation_ring_pinky = abs(landmarks[16].x - landmarks[20].x)
        
        # Check finger straightness
        index_straightness = abs(landmarks[8].x - landmarks[5].x)
        middle_straightness = abs(landmarks[12].x - landmarks[9].x)
        ring_straightness = abs(landmarks[16].x - landmarks[13].x)
        pinky_straightness = abs(landmarks[20].x - landmarks[17].x)
        
        # Check thumb position (should be tucked in)
        thumb_tucked = landmarks[4].x < landmarks[9].x
        
        # Check hand orientation (should be palm forward)
        palm_depth = landmarks[0].z - landmarks[9].z
        
        # Very strict checking for B
        if not fingers_extended:
            error_text = "Error: Extend your fingers straight up"
        elif finger_separation_index_middle > 0.05 or finger_separation_middle_ring > 0.05 or finger_separation_ring_pinky > 0.05:
            error_text = "Error: Keep your fingers closer together"
        elif index_straightness > 0.05 or middle_straightness > 0.05 or ring_straightness > 0.05 or pinky_straightness > 0.05:
            error_text = "Error: Keep your fingers straight"
        elif not thumb_tucked:
            error_text = "Error: Tuck your thumb against your palm"
        elif abs(palm_depth) > 0.1:
            error_text = "Error: Keep your palm facing forward"
        else:
            error_text = "Good form! Keep it up"
            correct_position = True
        
        return error_text, correct_position
    
    def get_position_feedback_for_c(self, landmarks):
        """Check if the hand forms the ASL 'C' shape."""
        error_text = "No feedback available"
        correct_position = False

        # Use thumb tip and index tip to check the circular shape
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        pinky_tip = landmarks[20]
        
        # Measure distance between thumb and index (should be moderate)
        thumb_index_dist = np.sqrt(
            (thumb_tip.x - index_tip.x) ** 2 +
            (thumb_tip.y - index_tip.y) ** 2 +
            (thumb_tip.z - index_tip.z) ** 2
        )

        # Check if all fingers are curved (tip is below base)
        fingers_curved = (
            landmarks[8].y > landmarks[5].y and
            landmarks[12].y > landmarks[9].y and
            landmarks[16].y > landmarks[13].y and
            landmarks[20].y > landmarks[17].y
        )

        if not fingers_curved:
            error_text = "Error: Curve all your fingers downward"
        elif thumb_index_dist < 0.03:
            error_text = "Error: Keep some space between thumb and index finger"
        elif pinky_tip.y < landmarks[17].y:
            error_text = "Error: Curve your pinky finger downward"
        else:
            error_text = "Good form! Keep it up"
            correct_position = True

        return error_text, correct_position
    
    def get_position_feedback_for_d(self, landmarks):
        """Check if the hand forms the ASL 'D' shape."""
        error_text = "No feedback available"
        correct_position = False

        # Index should be straight up
        index_straight = landmarks[8].y < landmarks[6].y

        # Other fingers should be curled
        middle_curled = landmarks[12].y > landmarks[10].y
        ring_curled = landmarks[16].y > landmarks[14].y
        pinky_curled = landmarks[20].y > landmarks[18].y

        # Thumb should touch middle finger or near palm center
        thumb_tip = landmarks[4]
        middle_mcp = landmarks[9]
        thumb_middle_dist = np.sqrt(
            (thumb_tip.x - middle_mcp.x) ** 2 +
            (thumb_tip.y - middle_mcp.y) ** 2 +
            (thumb_tip.z - middle_mcp.z) ** 2
        )

        if not index_straight:
            error_text = "Error: Extend your index finger upward"
        elif not (middle_curled and ring_curled and pinky_curled):
            error_text = "Error: Curl your middle, ring, and pinky fingers"
        elif thumb_middle_dist > 0.07:
            error_text = "Error: Touch your thumb to your middle finger"
        else:
            error_text = "Good form! Keep it up"
            correct_position = True

        return error_text, correct_position

    def get_position_feedback_for_e(self, landmarks):
        """Check if the hand forms the ASL 'E' shape."""
        error_text = "No feedback available"
        correct_position = False

        # All fingers should be bent toward the palm, not closed tightly (like 'A')
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        palm_base = landmarks[0]

        # Thumb should be across fingers
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]

        # Check if fingertips are near the palm
        finger_to_palm_dist = lambda tip: np.sqrt(
            (tip.x - palm_base.x) ** 2 +
            (tip.y - palm_base.y) ** 2 +
            (tip.z - palm_base.z) ** 2
        )

        distances = [finger_to_palm_dist(tip) for tip in [index_tip, middle_tip, ring_tip, pinky_tip]]

        # Conditions
        fingers_bent = all(d < 0.1 for d in distances)
        thumb_across = thumb_tip.x > landmarks[8].x  # Thumb to the left of index (mirror image for webcam)

        if not fingers_bent:
            error_text = "Error: Curl all fingers toward your palm"
        elif not thumb_across:
            error_text = "Error: Move your thumb across your curled fingers"
        else:
            error_text = "Good form! Keep it up"
            correct_position = True

        return error_text, correct_position
    
    def get_position_feedback_for_f(self, landmarks):
        # F: OK sign (thumb touches index tip, other fingers extended)
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        other_extended = (
            landmarks[12].y < landmarks[10].y and
            landmarks[16].y < landmarks[14].y and
            landmarks[20].y < landmarks[18].y
        )
        thumb_index_dist = np.linalg.norm(np.array([
            thumb_tip.x - index_tip.x,
            thumb_tip.y - index_tip.y,
            thumb_tip.z - index_tip.z
        ]))

        if thumb_index_dist > 0.04:
            return "Error: Touch your thumb and index finger to form a circle", False
        if not other_extended:
            return "Error: Extend your middle, ring, and pinky fingers", False
        return "Good form! Keep it up", True

    def get_position_feedback_for_g(self, landmarks):
        # G: Thumb and index extended sideways, rest closed
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]

        if index_tip.y > landmarks[6].y:
            return "Error: Extend your index finger straight", False
        if thumb_tip.x < landmarks[3].x:
            return "Error: Extend your thumb sideways", False
        if middle_tip.y < landmarks[10].y:
            return "Error: Close your middle finger", False
        return "Good form! Keep it up", True

    def get_position_feedback_for_h(self, landmarks):
        # H: Index and middle extended, others closed
        if not (landmarks[8].y < landmarks[6].y and landmarks[12].y < landmarks[10].y):
            return "Error: Extend your index and middle fingers", False
        if landmarks[16].y < landmarks[14].y or landmarks[20].y < landmarks[18].y:
            return "Error: Close your ring and pinky fingers", False
        return "Good form! Keep it up", True

    def get_position_feedback_for_i(self, landmarks):
        # I: Pinky extended, others closed
        if landmarks[20].y < landmarks[18].y:
            if all(landmarks[i].y > landmarks[i-2].y for i in [8,12,16]):
                return "Good form! Keep it up", True
            else:
                return "Error: Close all fingers except pinky", False
        return "Error: Extend your pinky finger", False

    def get_position_feedback_for_j(self, landmarks):
        # J: Draw 'J' shape with pinky (requires time-series, this is static fallback)
        return "Draw a 'J' shape with your pinky (motion not checked in static mode)", False

    def get_position_feedback_for_k(self, landmarks):
        # K: Index and middle extended in a 'V', thumb in between
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]

        if not (index_tip.y < landmarks[6].y and middle_tip.y < landmarks[10].y):
            return "Error: Extend index and middle fingers", False
        if not (thumb_tip.x > landmarks[8].x and thumb_tip.x < landmarks[12].x):
            return "Error: Place thumb between index and middle fingers", False
        return "Good form! Keep it up", True

    def get_position_feedback_for_l(self, landmarks):
        # L: Index up, thumb out to form an 'L'
        if landmarks[8].y > landmarks[6].y:
            return "Error: Extend your index finger upward", False
        if landmarks[4].x < landmarks[3].x:
            return "Error: Extend your thumb outward", False
        return "Good form! Keep it up", True

    def get_position_feedback_for_m(self, landmarks):
        # M: Thumb crosses under 3 fingers (index to ring)
        thumb_tip = landmarks[4]
        over_fingers = [landmarks[8], landmarks[12], landmarks[16]]
        if not all(thumb_tip.y > finger.y for finger in over_fingers):
            return "Error: Tuck thumb under your index, middle, and ring fingers", False
        return "Good form! Keep it up", True

    def get_position_feedback_for_n(self, landmarks):
        # N: Thumb under two fingers (index & middle)
        thumb_tip = landmarks[4]
        over_fingers = [landmarks[8], landmarks[12]]
        if not all(thumb_tip.y > finger.y for finger in over_fingers):
            return "Error: Tuck thumb under index and middle fingers", False
        return "Good form! Keep it up", True

    def get_position_feedback_for_o(self, landmarks):
        # O: Form an 'O' shape with all fingers and thumb
        tip_coords = np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in [4,8,12,16,20]])
        center = np.mean(tip_coords, axis=0)
        radii = [np.linalg.norm(tip - center) for tip in tip_coords]
        if np.std(radii) > 0.05:
            return "Error: Fingers should form a round 'O' shape", False
        return "Good form! Keep it up", True
    
    def get_position_feedback_for_p(self, landmarks):
        # P: Like K but tilted downward (hard to capture without orientation)
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        if not (index_tip.y < landmarks[6].y and middle_tip.y < landmarks[10].y):
            return "Error: Extend index and middle fingers", False
        if not (thumb_tip.x > landmarks[8].x and thumb_tip.x < landmarks[12].x):
            return "Error: Place thumb between index and middle fingers", False
        return "Good form! (Tilt detection not included)", True

    def get_position_feedback_for_q(self, landmarks):
        # Q: Like G but pointing downward (also hard statically)
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        if index_tip.y < landmarks[6].y:
            return "Error: Point your index finger downward", False
        if thumb_tip.x < landmarks[3].x:
            return "Error: Extend your thumb sideways", False
        return "Good form! (Tilt detection not included)", True

    def get_position_feedback_for_r(self, landmarks):
        # R: Index and middle crossed
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        if abs(index_tip.x - middle_tip.x) > 0.02:
            return "Error: Cross your index and middle fingers", False
        return "Good form! Keep it up", True

    def get_position_feedback_for_s(self, landmarks):
        # S: Fist with thumb across fingers
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        if thumb_tip.y < index_tip.y:
            return "Error: Place your thumb across your fingers", False
        return "Good form! Keep it up", True

    def get_position_feedback_for_t(self, landmarks):
        # T: Thumb between index and middle finger
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        if not (thumb_tip.y > index_tip.y and thumb_tip.y < middle_tip.y):
            return "Error: Place your thumb between index and middle fingers", False
        return "Good form! Keep it up", True

    def get_position_feedback_for_u(self, landmarks):
        # U: Index and middle together, pointing up
        if not (landmarks[8].y < landmarks[6].y and landmarks[12].y < landmarks[10].y):
            return "Error: Extend your index and middle fingers", False
        if abs(landmarks[8].x - landmarks[12].x) > 0.03:
            return "Error: Keep index and middle fingers close together", False
        return "Good form! Keep it up", True

    def get_position_feedback_for_v(self, landmarks):
        # V: Index and middle apart forming a V
        if not (landmarks[8].y < landmarks[6].y and landmarks[12].y < landmarks[10].y):
            return "Error: Extend your index and middle fingers", False
        if abs(landmarks[8].x - landmarks[12].x) < 0.05:
            return "Error: Separate your index and middle fingers to form a V", False
        return "Good form! Keep it up", True

    def get_position_feedback_for_w(self, landmarks):
        # W: Index, middle, and ring fingers up and spread
        if not (landmarks[8].y < landmarks[6].y and landmarks[12].y < landmarks[10].y and landmarks[16].y < landmarks[14].y):
            return "Error: Extend index, middle, and ring fingers", False
        if abs(landmarks[8].x - landmarks[12].x) < 0.03 or abs(landmarks[12].x - landmarks[16].x) < 0.03:
            return "Error: Spread your fingers to form a 'W' shape", False
        return "Good form! Keep it up", True

    def get_position_feedback_for_x(self, landmarks):
        # X: Index finger bent (hook shape), others closed
        if landmarks[8].y > landmarks[6].y:
            return "Error: Bend your index finger downward to form a hook", False
        if any(landmarks[i].y < landmarks[i-2].y for i in [12,16,20]):
            return "Error: Close other fingers", False
        return "Good form! Keep it up", True

    def get_position_feedback_for_y(self, landmarks):
        # Y: Thumb and pinky extended, others closed
        if not (landmarks[4].x > landmarks[3].x and landmarks[20].y < landmarks[18].y):
            return "Error: Extend your thumb and pinky", False
        if any(landmarks[i].y < landmarks[i-2].y for i in [8,12,16]):
            return "Error: Close middle three fingers", False
        return "Good form! Keep it up", True

    def get_position_feedback_for_z(self, landmarks):
        # Z: Motion-based, here we provide fallback message
        return "Draw a 'Z' shape with your index finger (motion not detected in static mode)", False
    
    def get_position_feedback(self, letter, landmarks):
        """Get feedback on hand position for specific letter."""
        error_text = "No feedback available"
        correct_position = False
        
        if letter == 'a':
            # Check if fingers are closed (making a fist)
            fingers_closed = (
                landmarks[8].y > landmarks[5].y and  # Index finger
                landmarks[12].y > landmarks[9].y and  # Middle finger
                landmarks[16].y > landmarks[13].y and  # Ring finger
                landmarks[20].y > landmarks[17].y  # Pinky
            )
                
            # Calculate distance for thumb position
            thumb_tip = landmarks[4]
            index_base = landmarks[5]
            index_pip = landmarks[6]  # Middle knuckle of index finger
            
            # Calculate distances for various thumb position checks
            thumb_to_index_base = np.sqrt(
                    (thumb_tip.x - index_base.x)**2 + 
                    (thumb_tip.y - index_base.y)**2 +
                    (thumb_tip.z - index_base.z)**2
                )
                
            thumb_to_index_pip = np.sqrt(
                (thumb_tip.x - index_pip.x)**2 + 
                (thumb_tip.y - index_pip.y)**2 +
                (thumb_tip.z - index_pip.z)**2
            )
            
            # Check if thumb is positioned alongside the index finger
            thumb_alongside_index = thumb_tip.x >= index_base.x
            
            # Check if thumb is pointing upward
            thumb_pointing_up = thumb_tip.y <= landmarks[3].y  # compared to thumb IP
            
            # Analyze positioning for 'A' sign
            if not fingers_closed:
                error_text = "Error: Close all your fingers into a fist"
            elif not thumb_alongside_index:
                error_text = "Error: Keep your thumb alongside your fingers, not across palm"
            elif thumb_to_index_base > 0.06:  # Stricter check for thumb-to-index distance
                error_text = "Error: Thumb should touch the side of your index finger"
            elif not thumb_pointing_up:
                error_text = "Error: Thumb should point upward, not downward"
            else:
                error_text = "Good form! Keep it up"
                correct_position = True
                
        elif letter == 'b':
            # Use the more strict B detection that matches the model's confidence
            return self.get_position_feedback_for_b(landmarks)
        elif letter == 'c':
            return self.get_position_feedback_for_c(landmarks)
        elif letter == 'd':
            return self.get_position_feedback_for_d(landmarks)
        elif letter == 'e':
            return self.get_position_feedback_for_e(landmarks)
        elif letter == 'f':
            return self.get_position_feedback_for_f(landmarks)
        elif letter == 'g':
            return self.get_position_feedback_for_g(landmarks)
        elif letter == 'h':
            return self.get_position_feedback_for_h(landmarks)
        elif letter == 'i':
            return self.get_position_feedback_for_i(landmarks)
        elif letter == 'j':
            return self.get_position_feedback_for_j(landmarks)
        elif letter == 'k':
            return self.get_position_feedback_for_k(landmarks)
        elif letter == 'l':
            return self.get_position_feedback_for_l(landmarks)
        elif letter == 'm':
            return self.get_position_feedback_for_m(landmarks)
        elif letter == 'n':
            return self.get_position_feedback_for_n(landmarks)
        elif letter == 'o':
            return self.get_position_feedback_for_o(landmarks)
        elif letter == 'p':
            return self.get_position_feedback_for_p(landmarks)
        elif letter == 'q':
            return self.get_position_feedback_for_q(landmarks)
        elif letter == 'r':
            return self.get_position_feedback_for_r(landmarks)
        elif letter == 's':
            return self.get_position_feedback_for_s(landmarks)
        elif letter == 't':
            return self.get_position_feedback_for_t(landmarks)
        elif letter == 'u':
            return self.get_position_feedback_for_u(landmarks)
        elif letter == 'v':
            return self.get_position_feedback_for_v(landmarks)
        elif letter == 'w':
            return self.get_position_feedback_for_w(landmarks)
        elif letter == 'x':
            return self.get_position_feedback_for_x(landmarks)
        elif letter == 'y':
            return self.get_position_feedback_for_y(landmarks)
        elif letter == 'z':
            return self.get_position_feedback_for_z(landmarks)
        
        return error_text, correct_position
    
    def get_letter_instructions(self, letter):
        """Return instructions for the current target letter."""
        if letter == 'a':
            return [
                "Make a fist with all fingers closed",
                "Place your thumb against the side of your index finger",
                "Keep thumb pointing upward, not across palm"
            ]
        elif letter == 'b':
            return [
                "Hold your hand up with palm facing forward",
                "Keep your fingers straight and together",
                "Tuck your thumb against your palm",
                "Your fingers should be pointing upward"
            ]
        elif letter == 'c':
            return [
                "Curve your fingers and thumb to form a 'C' shape",
                "Keep all fingers together in the curved position",
                "Palm should face to the side",
                "Thumb and fingers should be aligned in the same curved plane"
            ]
        elif letter == 'd':
            return [
                "Make a circle with your thumb and index finger",
                "Keep your middle, ring, and pinky fingers pointing up",
                "Palm should face forward",
                "The middle, ring, and pinky fingers should be straight and together"
            ]
        elif letter == 'e':
            return [
                "Curl all fingers into the palm",
                "Tuck your thumb against the side of your index finger",
                "Keep your palm facing forward",
                "Your fingernails should be visible as you curl them"
            ]
        elif letter == 'f':
            return [
                "Connect your thumb and index finger to form a circle",
                "Extend your other three fingers upward",
                "Keep your index finger and thumb touching at the tips",
                "Your remaining three fingers should be straight and together"
            ]
        elif letter == 'g':
            return [
                "Make a fist with your hand, palm facing sideways",
                "Extend your index finger pointing forward",
                "Thumb should rest alongside your fist, not tucked in",
                "The index finger and thumb should form a 'G' shape"
            ]
        elif letter == 'h':
            return [
                "Make a fist with your hand, palm facing sideways",
                "Extend your index and middle fingers forward together",
                "Keep your fingers parallel to the ground",
                "Thumb should rest alongside your fist, not tucked in"
            ]
        elif letter == 'i':
            return [
                "Make a fist with your hand, palm facing sideways",
                "Extend only your pinky finger upward",
                "Keep the rest of your fingers curled into a fist",
                "Thumb should rest across the curled fingers"
            ]
        else:
            return ["No instructions available for this letter"]
    
    def draw_semitransparent_rect(self, image, start_point, end_point, color, alpha=0.3):
        """Draw a semi-transparent rectangle on the image."""
        overlay = image.copy()
        cv2.rectangle(overlay, start_point, end_point, color, -1)
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        return image
    
    def detect_sign(self):
        """Run real-time detection to identify ASL signs."""
        if self.model is None:
            print("No model loaded. Please load a model first.")
            return
        
        print(f"Starting sign detection mode...")
        print(f"Current target letter: {self.target_letter.upper()}")
        print(f"Current target hand: {self.target_hand.upper()}")
        print("Press 'l' to toggle between letters A through I")
        print("Press 'h' to toggle between left/right hand")
        print("Press 'q' or ESC to quit")
        
        # Setup camera
        cap = self.setup_camera()
        if cap is None:
            print("Failed to setup camera. Exiting detection.")
            return
        
        # Guidance title
        guide_title = f"How to form the '{self.target_letter.upper()}' sign correctly:"
        
        # Main detection loop
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Failed to grab frame")
                continue
            
            # Flip image for selfie view
            image = cv2.flip(image, 1)
            
            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)
            
            # Create semi-transparent overlay for the guide area
            self.draw_semitransparent_rect(image, (0, 0), (image.shape[1], 150), (0, 0, 0))
            
            # Draw guide title and instructions
            cv2.putText(image, f"How to form the '{self.target_letter.upper()}' sign correctly:", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw guide instructions
            instructions = self.get_letter_instructions(self.target_letter)
            for i, line in enumerate(instructions):
                cv2.putText(image, f"{i+1}. {line}", (20, 50 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Draw hand information text
            cv2.putText(image, f"Target Letter: {self.target_letter.upper()} | Target Hand: {self.target_hand.upper()}", 
                        (image.shape[1] - 450, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Initialize a flag to track if target hand was found
            found_target_hand = False
            error_text = "No hand detected"
            
            # Process hand landmarks if detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Check hand type
                    is_right_hand = self.get_hand_type(hand_landmarks, results)
                    hand_type = "right" if is_right_hand else "left"
                    
                    # Draw hand landmarks
                    self.mp_drawing.draw_landmarks(
                        image, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Label the hand
                    hand_pos_x = int(hand_landmarks.landmark[0].x * image.shape[1])
                    hand_pos_y = int(hand_landmarks.landmark[0].y * image.shape[0])
                    cv2.putText(image, hand_type.upper(), (hand_pos_x - 20, hand_pos_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    
                    # Only process the target hand
                    if hand_type == self.target_hand or self.target_hand == "both":
                        found_target_hand = True
                        
                        # Extract features
                        features = self.extract_features(hand_landmarks.landmark, is_right_hand)
                        
                        # Get position feedback
                        error_text, correct_position = self.get_position_feedback(
                            self.target_letter, hand_landmarks.landmark
                        )
                        
                        # Make prediction
                        prediction = self.model.predict([features])[0]
                        probabilities = self.model.predict_proba([features])[0]
                        
                        # Match probability to the correct letter
                        letter_index = np.where(self.model_classes == self.target_letter)[0][0]
                        target_probability = probabilities[letter_index]
                        
                        # Special check for letter A and thumb position
                        if self.target_letter == 'a' and 'thumb' in error_text.lower():
                            # If there's a thumb position error for A, reduce the effective confidence
                            # to ensure it doesn't override the position feedback
                            effective_confidence = target_probability * 0.5
                        else:
                            effective_confidence = target_probability
                            
                        # If confidence is high enough, override position feedback
                        # Except for thumb position errors with letter A
                        if effective_confidence >= self.confidence_threshold:
                            # Don't override if it's a thumb position error for letter A
                            if not (self.target_letter == 'a' and 'thumb' in error_text.lower()):
                                correct_position = True
                                error_text = ""
                        # Otherwise synchronize the position feedback with the confidence level
                        elif target_probability < self.confidence_threshold:
                            correct_position = False
                            # If the previous feedback was "good form" but confidence is low,
                            # update the feedback to reflect the discrepancy
                            if error_text == "Good form! Keep it up":
                                error_text = f"Adjust your hand position for better {self.target_letter.upper()} recognition"
                        
                        # Display result based on both position and confidence
                        if (prediction == self.target_letter and target_probability >= self.confidence_threshold and correct_position) or target_probability >= 0.99:
                            result_text = f"{self.target_letter.upper()} sign detected! (Confidence: {target_probability:.2f})"
                            color = (0, 255, 0)
                            # Clear any error message when detection is successful
                            error_text = ""
                        else:
                            # Only show "not an X sign" message when confidence is truly low
                            # Otherwise show "almost an X sign" for medium confidence
                            if target_probability < 0.5:
                                result_text = f"Not a{' n' if self.target_letter == 'a' else ' '}{self.target_letter.upper()} sign (Confidence: {target_probability:.2f})"
                                color = (0, 0, 255)
                            else:
                                result_text = f"Almost a{' n' if self.target_letter == 'a' else ' '}{self.target_letter.upper()} sign (Confidence: {target_probability:.2f})"
                                color = (0, 165, 255)
                        
                        # Add confidence bar visualization
                        confidence_bar_width = int(300 * target_probability)
                        
                        # Determine bar color based on the same logic as the text
                        if (prediction == self.target_letter and target_probability >= self.confidence_threshold and correct_position) or target_probability >= 0.99:
                            confidence_color = (0, 255, 0)  # Green for success
                        elif target_probability < 0.5:
                            confidence_color = (0, 0, 255)  # Red for low confidence
                        else:
                            confidence_color = (0, 165, 255)  # Orange for medium confidence
                        
                        # Draw confidence bar background
                        cv2.rectangle(image, (10, 185), (310, 200), (100, 100, 100), -1)
                        # Draw filled portion
                        cv2.rectangle(image, (10, 185), (10 + confidence_bar_width, 200), confidence_color, -1)
                        # Draw threshold marker
                        threshold_x = int(10 + 300 * self.confidence_threshold)
                        cv2.line(image, (threshold_x, 180), (threshold_x, 205), (255, 255, 255), 2)
                        
                        cv2.putText(image, result_text, (10, 175), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if not found_target_hand:
                # Target hand not detected
                cv2.putText(image, f"No {self.target_hand} hand detected", (10, 175), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Display error/guidance text
            if error_text:  # Only display error text if there's an actual message
                # Set color based on whether it's an error or just guidance
                error_color = (0, 0, 255) if error_text.startswith("Error") else (0, 165, 255)
                cv2.putText(image, error_text, (10, 140), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, error_color, 2)
            
            # Add instructions to screen
            cv2.putText(image, "Press 'l': Toggle letter | 'h': Toggle hand | 'q' or ESC: Quit", 
                        (10, image.shape[0] - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display the image
            cv2.imshow('ASL Sign Detector', image)
            
            # Process key presses
            key = cv2.waitKey(5) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or q key
                break
            elif key == ord('h'):  # Toggle hand
                self.toggle_target_hand()
                print(f"Switched to {self.target_hand} hand")
            elif key == ord('l'):  # Toggle letter
                self.toggle_target_letter()
                print(f"Switched to letter {self.target_letter.upper()}")
                # Update guide title
                guide_title = f"How to form the '{self.target_letter.upper()}' sign correctly:"
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

def main():
    detector = ASLDetector()
    
    print("\nASL Sign Detector for A through I")
    print("================================")
    
    if not os.path.exists("./webcam_models"):
        os.makedirs("./webcam_models")
        print("Created webcam_models directory. Please place your model file there.")
    
    while True:
        print("\nOptions:")
        print("1. Load model from a different directory")
        print("2. Detect signs (requires model)")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == '1':
            model_dir = input("Enter the directory path containing the model: ")
            detector.models_dir = model_dir
            detector.load_model()
        elif choice == '2':
            if detector.model is not None:
                detector.detect_sign()
            else:
                print("No model loaded. Please load a model first.")
        elif choice == '3':
            print("Exiting program")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
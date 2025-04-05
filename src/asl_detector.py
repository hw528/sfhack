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
        self.target_hand = "both"  # Default to both hands
        self.confidence_threshold = 0.7
        
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
        """Toggle between letters A through Y (excluding J and Z)."""
        letter_cycle = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']
        current_index = letter_cycle.index(self.target_letter) if self.target_letter in letter_cycle else 0
        next_index = (current_index + 1) % len(letter_cycle)
        self.target_letter = letter_cycle[next_index]
        return self.target_letter
    
    def setup_camera(self):
        """Set up and return a camera capture object."""
        print("Setting up camera...")
        cap = cv2.VideoCapture(1)
        
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
        """Special strict position checking for letter C."""
        error_text = "No feedback available"
        correct_position = False
        
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
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        thumb_index_dist = np.sqrt(
            (thumb_tip.x - index_tip.x)**2 + 
            (thumb_tip.y - index_tip.y)**2
        )
        
        # Check palm orientation - should be facing sideways
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        palm_orientation = abs(wrist.z - middle_mcp.z)
        
        # Very strict checking for C
        if fingers_too_straight:
            error_text = "Error: Curve your fingers more to form a 'C' shape"
        elif fingers_separated:
            error_text = "Error: Keep your fingers closer together"
        elif thumb_index_dist > 0.2 or thumb_index_dist < 0.1:
            error_text = "Error: Adjust your thumb and index finger to form a clear 'C' shape"
        elif palm_orientation < 0.05:
            error_text = "Error: Turn your palm to face sideways"
        else:
            error_text = "Good form! Keep it up"
            correct_position = True
        
        return error_text, correct_position
    
    def get_position_feedback_for_d(self, landmarks):
        """Special strict position checking for letter D."""
        error_text = "No feedback available"
        correct_position = False
        
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
        
        # Check palm orientation - should be facing forward
        palm_depth = landmarks[0].z - landmarks[9].z
        
        # Very strict checking for D
        if thumb_index_dist > 0.05:
            error_text = "Error: Bring your thumb and index fingertips closer to form a circle"
        elif not (middle_extended and ring_extended and pinky_extended):
            error_text = "Error: Extend your middle, ring, and pinky fingers upward"
        elif not fingers_straight:
            error_text = "Error: Keep your extended fingers straight"
        elif not fingers_together:
            error_text = "Error: Keep your extended fingers together"
        elif abs(palm_depth) > 0.1:
            error_text = "Error: Keep your palm facing forward"
        else:
            error_text = "Good form! Keep it up"
            correct_position = True
        
        return error_text, correct_position
    
    def get_position_feedback_for_e(self, landmarks):
        """Special strict position checking for letter E."""
        error_text = "No feedback available"
        correct_position = False
        
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
            error_text = "Error: Curl your index finger into your palm"
        elif not middle_curled:
            error_text = "Error: Curl your middle finger into your palm"
        elif not ring_curled:
            error_text = "Error: Curl your ring finger into your palm"
        elif not pinky_curled:
            error_text = "Error: Curl your pinky finger into your palm"
        elif thumb_index_dist > 0.08:
            error_text = "Error: Tuck your thumb against your index finger"
            
        # Set overall message
        if not error_text:
            error_text = "Good form! Keep it up"
            correct_position = True
        
        return error_text, correct_position
    
    def get_position_feedback_for_f(self, landmarks):
        """Special strict position checking for letter F."""
        error_text = "No feedback available"
        correct_position = False
        
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
            error_text = "Error: Touch your thumb and index fingertips to form a circle"
        elif not (middle_extended and ring_extended and pinky_extended):
            error_text = "Error: Extend your middle, ring, and pinky fingers upward"
        elif not fingers_together:
            error_text = "Error: Keep your extended fingers together"
            
        # Set overall message
        if not error_text:
            error_text = "Good form! Keep it up"
            correct_position = True
        
        return error_text, correct_position
    
    def get_position_feedback_for_g(self, landmarks):
        """Special strict position checking for letter G."""
        error_text = "No feedback available"
        correct_position = False
        
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
            error_text = "Error: Extend your index finger forward (not upward)"
        elif not (middle_closed and ring_closed and pinky_closed):
            error_text = "Error: Close your middle, ring, and pinky fingers into a fist"
        elif palm_orientation < 0.05:
            error_text = "Error: Turn your palm to face sideways"
        elif not thumb_position_good:
            error_text = "Error: Position your thumb alongside your fist, not tucked in"
            
        # Set overall message
        if not error_text:
            error_text = "Good form! Keep it up"
            correct_position = True
        
        return error_text, correct_position
    
    def get_position_feedback_for_h(self, landmarks):
        """Special strict position checking for letter H."""
        error_text = "No feedback available"
        correct_position = False
        
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
            error_text = "Error: Extend your index and middle fingers forward, not upward"
        elif not (ring_closed and pinky_closed):
            error_text = "Error: Close your ring and pinky fingers into a fist"
        elif not fingers_parallel:
            error_text = "Error: Keep your index and middle fingers parallel"
        elif not fingers_together:
            error_text = "Error: Keep your index and middle fingers together"
        elif palm_orientation < 0.05:
            error_text = "Error: Turn your palm to face sideways"
            
        # Set overall message
        if not error_text:
            error_text = "Good form! Keep it up"
            correct_position = True
        
        return error_text, correct_position
    
    def get_position_feedback_for_i(self, landmarks):
        """Special strict position checking for letter I."""
        error_text = "No feedback available"
        correct_position = False
        
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
            error_text = "Error: Close your index, middle, and ring fingers into a fist"
        elif not pinky_extended:
            error_text = "Error: Extend your pinky finger upward"
        elif palm_orientation < 0.05:
            error_text = "Error: Turn your palm to face sideways"
        elif not thumb_position_good:
            error_text = "Error: Position your thumb across your curled fingers"
            
        # Set overall message
        if not error_text:
            error_text = "Good form! Keep it up"
            correct_position = True
        
        return error_text, correct_position
    
    def get_position_feedback_for_k(self, landmarks):
        """Special strict position checking for letter K."""
        error_text = "No feedback available"
        correct_position = False
        
        # Check if index and middle fingers are extended upward and others closed
        index_extended = landmarks[8].y < landmarks[5].y
        middle_extended = landmarks[12].y < landmarks[9].y
        ring_closed = landmarks[16].y > landmarks[13].y
        pinky_closed = landmarks[20].y > landmarks[17].y
        
        # Check if index and middle fingers form a 'K' shape
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        index_mcp = landmarks[5]
        index_angle = np.arctan2(index_tip.y - index_mcp.y, index_tip.x - index_mcp.x)
        middle_angle = np.arctan2(middle_tip.y - index_mcp.y, middle_tip.x - index_mcp.x)
        angle_diff = abs(index_angle - middle_angle)
        k_shape = angle_diff > 0.5  # Fingers should be at an angle to each other
        
        # Check thumb position - should be extended
        thumb_tip = landmarks[4]
        thumb_extended = thumb_tip.y < landmarks[2].y
        
        # Check palm orientation - should be facing forward
        palm_depth = landmarks[0].z - landmarks[9].z
        
        # Analyze positioning for 'K' sign
        if not (index_extended and middle_extended):
            error_text = "Error: Extend your index and middle fingers upward"
        elif not (ring_closed and pinky_closed):
            error_text = "Error: Close your ring and pinky fingers"
        elif not k_shape:
            error_text = "Error: Position your index finger straight up and middle finger at an angle"
        elif not thumb_extended:
            error_text = "Error: Extend your thumb"
        elif abs(palm_depth) > 0.1:
            error_text = "Error: Keep your palm facing forward"
        else:
            error_text = "Good form! Keep it up"
            correct_position = True
        
        return error_text, correct_position
    
    def get_position_feedback_for_l(self, landmarks):
        """Special strict position checking for letter L."""
        error_text = "No feedback available"
        correct_position = False
        
        # Check if thumb is extended to the side and index is pointing up
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        wrist = landmarks[0]
        
        thumb_extended_sideways = thumb_tip.x > (wrist.x + 0.1)
        index_extended_up = index_tip.y < landmarks[5].y
        
        # Check if other fingers are closed
        middle_closed = landmarks[12].y > landmarks[9].y
        ring_closed = landmarks[16].y > landmarks[13].y
        pinky_closed = landmarks[20].y > landmarks[17].y
        
        # Check L shape (90 degree angle)
        thumb_index_horizontal = abs(thumb_tip.y - index_tip.y) < 0.1
        index_vertical = abs(index_tip.x - landmarks[5].x) < 0.1
        
        # Analyze positioning for 'L' sign
        if not thumb_extended_sideways:
            error_text = "Error: Extend your thumb to the side"
        elif not index_extended_up:
            error_text = "Error: Extend your index finger upward"
        elif not (middle_closed and ring_closed and pinky_closed):
            error_text = "Error: Close your middle, ring, and pinky fingers"
        elif not (thumb_index_horizontal and index_vertical):
            error_text = "Error: Form a clear 'L' shape with your thumb and index finger"
        else:
            error_text = "Good form! Keep it up"
            correct_position = True
        
        return error_text, correct_position
    
    def get_position_feedback_for_m(self, landmarks):
        """Special strict position checking for letter M."""
        error_text = "No feedback available"
        correct_position = False
        
        # For 'M', all fingers are folded with thumb tucked under
        thumb_tip = landmarks[4]
        wrist = landmarks[0]
        
        # Check if fingers are tucked
        fingers_tucked = (
            landmarks[8].y > landmarks[6].y and  # Index tucked
            landmarks[12].y > landmarks[10].y and  # Middle tucked
            landmarks[16].y > landmarks[14].y and  # Ring tucked
            landmarks[20].y > landmarks[18].y  # Pinky tucked
        )
        
        # Check thumb position - should be tucked between fingers
        thumb_tucked = thumb_tip.x < wrist.x
        
        # Check palm orientation - should be facing down
        palm_down = landmarks[9].y > wrist.y
        
        # Analyze positioning for 'M' sign
        if not fingers_tucked:
            error_text = "Error: Tuck all fingers into your palm"
        elif not thumb_tucked:
            error_text = "Error: Tuck your thumb between your fingers"
        elif not palm_down:
            error_text = "Error: Turn your palm to face downward"
        else:
            error_text = "Good form! Keep it up"
            correct_position = True
        
        return error_text, correct_position
    
    def get_position_feedback_for_n(self, landmarks):
        """Special strict position checking for letter N."""
        error_text = "No feedback available"
        correct_position = False
        
        # 'N' is similar to 'M' but with different finger positions
        thumb_tip = landmarks[4]
        wrist = landmarks[0]
        
        # Check if fingers are positioned correctly
        index_ring_pinky_tucked = (
            landmarks[8].y > landmarks[6].y and  # Index tucked
            landmarks[16].y > landmarks[14].y and  # Ring tucked
            landmarks[20].y > landmarks[18].y  # Pinky tucked
        )
        
        middle_tucked = landmarks[12].y > landmarks[10].y
        
        # Check thumb position
        thumb_tucked = thumb_tip.x < wrist.x
        
        # Check palm orientation
        palm_down = landmarks[9].y > wrist.y
        
        # Analyze positioning for 'N' sign
        if not index_ring_pinky_tucked:
            error_text = "Error: Tuck your index, ring, and pinky fingers"
        elif not middle_tucked:
            error_text = "Error: Tuck your middle finger"
        elif not thumb_tucked:
            error_text = "Error: Tuck your thumb into your palm"
        elif not palm_down:
            error_text = "Error: Turn your palm to face downward"
        else:
            error_text = "Good form! Keep it up"
            correct_position = True
        
        return error_text, correct_position
    
    def get_position_feedback_for_o(self, landmarks):
        """Special strict position checking for letter O."""
        error_text = "No feedback available"
        correct_position = False
        
        # All fingers form a circular shape with the thumb
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        # Check if fingers are curved to form an 'O'
        fingers_curved = (
            landmarks[8].y > landmarks[7].y and  # Index curved
            landmarks[12].y > landmarks[11].y and  # Middle curved
            landmarks[16].y > landmarks[15].y and  # Ring curved
            landmarks[20].y > landmarks[19].y  # Pinky curved
        )
        
        # Check if the fingertips are close to the thumb (forming a circle)
        thumb_to_index_dist = np.sqrt(
            (thumb_tip.x - index_tip.x)**2 + 
            (thumb_tip.y - index_tip.y)**2
        )
        
        # Analyze positioning for 'O' sign
        if not fingers_curved:
            error_text = "Error: Curve all your fingers to form an 'O' shape"
        elif thumb_to_index_dist > 0.05:
            error_text = "Error: Bring your thumb and index finger together to form a circle"
        else:
            error_text = "Good form! Keep it up"
            correct_position = True
        
        return error_text, correct_position
    
    def get_position_feedback_for_p(self, landmarks):
        """Special strict position checking for letter P."""
        error_text = "No feedback available"
        correct_position = False
        
        # For 'P', index finger points down, thumb to side, other fingers extended
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        # Check index finger pointing down
        index_pointing_down = index_tip.y > landmarks[5].y
        
        # Check thumb extended to side
        thumb_extended = thumb_tip.x > landmarks[1].x
        
        # Check other fingers extended
        other_fingers_extended = (
            landmarks[12].y < landmarks[9].y and  # Middle extended
            landmarks[16].y < landmarks[13].y and  # Ring extended
            landmarks[20].y < landmarks[17].y  # Pinky extended
        )
        
        # Check palm orientation - should be facing to the side
        palm_sideways = abs(landmarks[0].z - landmarks[9].z) > 0.05
        
        # Analyze positioning for 'P' sign
        if not index_pointing_down:
            error_text = "Error: Point your index finger downward"
        elif not thumb_extended:
            error_text = "Error: Extend your thumb to the side"
        elif not other_fingers_extended:
            error_text = "Error: Extend your middle, ring, and pinky fingers"
        elif not palm_sideways:
            error_text = "Error: Turn your palm to face sideways"
        else:
            error_text = "Good form! Keep it up"
            correct_position = True
        
        return error_text, correct_position

    def get_position_feedback_for_q(self, landmarks):
        """Special strict position checking for letter Q."""
        error_text = "No feedback available"
        correct_position = False
        
        # For 'Q', similar to 'P' but with different finger positioning
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        # Check index finger pointing down
        index_pointing_down = index_tip.y > landmarks[5].y
        
        # Check thumb position
        thumb_position_good = thumb_tip.x < landmarks[1].x
        
        # Check other fingers curled
        other_fingers_curled = (
            landmarks[12].y > landmarks[9].y and  # Middle curled
            landmarks[16].y > landmarks[13].y and  # Ring curled
            landmarks[20].y > landmarks[17].y  # Pinky curled
        )
        
        # Check palm orientation - should be facing to the side
        palm_sideways = abs(landmarks[0].z - landmarks[9].z) > 0.05
        
        # Analyze positioning for 'Q' sign
        if not index_pointing_down:
            error_text = "Error: Point your index finger downward"
        elif not thumb_position_good:
            error_text = "Error: Position your thumb correctly along your fingers"
        elif not other_fingers_curled:
            error_text = "Error: Curl your middle, ring, and pinky fingers"
        elif not palm_sideways:
            error_text = "Error: Turn your palm to face sideways"
        else:
            error_text = "Good form! Keep it up"
            correct_position = True
        
        return error_text, correct_position
        
    def get_position_feedback_for_r(self, landmarks):
        """Special strict position checking for letter R."""
        error_text = "No feedback available"
        correct_position = False
        
        # For 'R', index and middle fingers cross
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        
        # Check if index and middle are extended
        index_extended = index_tip.y < landmarks[5].y
        middle_extended = middle_tip.y < landmarks[9].y
        
        # Check if fingers are crossed (x positions are reversed)
        fingers_crossed = (middle_tip.x < index_tip.x) 
        
        # Check if other fingers are closed
        ring_pinky_closed = (
            landmarks[16].y > landmarks[13].y and  # Ring closed
            landmarks[20].y > landmarks[17].y  # Pinky closed
        )
        
        # Check thumb position
        thumb_tip = landmarks[4]
        thumb_tucked = thumb_tip.x < landmarks[1].x
        
        # Analyze positioning for 'R' sign
        if not (index_extended and middle_extended):
            error_text = "Error: Extend your index and middle fingers upward"
        elif not fingers_crossed:
            error_text = "Error: Cross your index and middle fingers"
        elif not ring_pinky_closed:
            error_text = "Error: Close your ring and pinky fingers"
        elif not thumb_tucked:
            error_text = "Error: Tuck your thumb against your palm"
        else:
            error_text = "Good form! Keep it up"
            correct_position = True
        
        return error_text, correct_position
    
    def get_position_feedback_for_s(self, landmarks):
        """Special strict position checking for letter S."""
        error_text = "No feedback available"
        correct_position = False
        
        # For 'S', the hand forms a fist with the thumb wrapped across the front
        thumb_tip = landmarks[4]
        
        # Check if fingers are closed (making a fist)
        fingers_closed = (
            landmarks[8].y > landmarks[5].y and  # Index closed
            landmarks[12].y > landmarks[9].y and  # Middle closed
            landmarks[16].y > landmarks[13].y and  # Ring closed
            landmarks[20].y > landmarks[17].y  # Pinky closed
        )
        
        # Check thumb position - should be in front of fingers
        thumb_in_front = thumb_tip.z < landmarks[5].z
        
        # Analyze positioning for 'S' sign
        if not fingers_closed:
            error_text = "Error: Close all your fingers into a fist"
        elif not thumb_in_front:
            error_text = "Error: Wrap your thumb across the front of your fist"
        else:
            error_text = "Good form! Keep it up"
            correct_position = True
        
        return error_text, correct_position
    
    def get_position_feedback_for_t(self, landmarks):
        """Special strict position checking for letter T."""
        error_text = "No feedback available"
        correct_position = False
        
        # For 'T', the thumb is between index and middle fingers
        thumb_tip = landmarks[4]
        index_mcp = landmarks[5]
        middle_mcp = landmarks[9]
        
        # Check thumb position - should be between index and middle fingers
        thumb_between_fingers = (
            thumb_tip.x > index_mcp.x and
            thumb_tip.x < middle_mcp.x and
            thumb_tip.y < index_mcp.y  # Thumb should be above the knuckles
        )
        
        # Check if fingers are closed
        fingers_closed = (
            landmarks[8].y > landmarks[5].y and  # Index closed
            landmarks[12].y > landmarks[9].y and  # Middle closed
            landmarks[16].y > landmarks[13].y and  # Ring closed
            landmarks[20].y > landmarks[17].y  # Pinky closed
        )
        
        # Analyze positioning for 'T' sign
        if not thumb_between_fingers:
            error_text = "Error: Position your thumb between index and middle fingers"
        elif not fingers_closed:
            error_text = "Error: Close all your fingers"
        else:
            error_text = "Good form! Keep it up"
            correct_position = True
        
        return error_text, correct_position
    
    def get_position_feedback_for_u(self, landmarks):
        """Special strict position checking for letter U."""
        error_text = "No feedback available"
        correct_position = False
        
        # For 'U', index and middle fingers are extended together
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        
        # Check if index and middle are extended
        index_extended = index_tip.y < landmarks[5].y
        middle_extended = middle_tip.y < landmarks[9].y
        
        # Check if index and middle are close together
        fingers_together = abs(index_tip.x - middle_tip.x) < 0.05
        
        # Check if other fingers are closed
        ring_pinky_closed = (
            landmarks[16].y > landmarks[13].y and  # Ring closed
            landmarks[20].y > landmarks[17].y  # Pinky closed
        )
        
        # Check thumb position
        thumb_tip = landmarks[4]
        thumb_tucked = thumb_tip.x < landmarks[1].x
        
        # Analyze positioning for 'U' sign
        if not (index_extended and middle_extended):
            error_text = "Error: Extend your index and middle fingers upward"
        elif not fingers_together:
            error_text = "Error: Keep your index and middle fingers close together"
        elif not ring_pinky_closed:
            error_text = "Error: Close your ring and pinky fingers"
        elif not thumb_tucked:
            error_text = "Error: Tuck your thumb against your palm"
        else:
            error_text = "Good form! Keep it up"
            correct_position = True
        
        return error_text, correct_position
    
    def get_position_feedback_for_v(self, landmarks):
        """Special strict position checking for letter V."""
        error_text = "No feedback available"
        correct_position = False
        
        # For 'V', index and middle fingers are extended in a V shape
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        
        # Check if index and middle are extended
        index_extended = index_tip.y < landmarks[5].y
        middle_extended = middle_tip.y < landmarks[9].y
        
        # Check if index and middle are spread apart
        fingers_spread = abs(index_tip.x - middle_tip.x) > 0.05
        
        # Check if other fingers are closed
        ring_pinky_closed = (
            landmarks[16].y > landmarks[13].y and  # Ring closed
            landmarks[20].y > landmarks[17].y  # Pinky closed
        )
        
        # Check thumb position
        thumb_tip = landmarks[4]
        thumb_tucked = thumb_tip.x < landmarks[1].x
        
        # Analyze positioning for 'V' sign
        if not (index_extended and middle_extended):
            error_text = "Error: Extend your index and middle fingers upward"
        elif not fingers_spread:
            error_text = "Error: Spread your index and middle fingers apart to form a 'V'"
        elif not ring_pinky_closed:
            error_text = "Error: Close your ring and pinky fingers"
        elif not thumb_tucked:
            error_text = "Error: Tuck your thumb against your palm"
        else:
            error_text = "Good form! Keep it up"
            correct_position = True
        
        return error_text, correct_position
    
    def get_position_feedback_for_w(self, landmarks):
        """Special strict position checking for letter W."""
        error_text = "No feedback available"
        correct_position = False
        
        # For 'W', index, middle, and ring fingers are extended
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        
        # Check if three fingers are extended
        index_extended = index_tip.y < landmarks[5].y
        middle_extended = middle_tip.y < landmarks[9].y
        ring_extended = ring_tip.y < landmarks[13].y
        
        # Check if fingers are spread
        fingers_spread = (
            abs(index_tip.x - middle_tip.x) > 0.03 and 
            abs(middle_tip.x - ring_tip.x) > 0.03
        )
        
        # Check if pinky is closed
        pinky_closed = landmarks[20].y > landmarks[17].y
        
        # Check thumb position
        thumb_tip = landmarks[4]
        thumb_tucked = thumb_tip.x < landmarks[1].x
        
        # Analyze positioning for 'W' sign
        if not (index_extended and middle_extended and ring_extended):
            error_text = "Error: Extend your index, middle, and ring fingers"
        elif not fingers_spread:
            error_text = "Error: Spread your fingers to form a 'W' shape"
        elif not pinky_closed:
            error_text = "Error: Close your pinky finger"
        elif not thumb_tucked:
            error_text = "Error: Tuck your thumb against your palm"
        else:
            error_text = "Good form! Keep it up"
            correct_position = True
        
        return error_text, correct_position
    
    def get_position_feedback_for_x(self, landmarks):
        """Special strict position checking for letter X."""
        error_text = "No feedback available"
        correct_position = False
        
        # For 'X', index finger is bent at the middle joint
        index_tip = landmarks[8]
        index_pip = landmarks[6]  # Middle joint of index finger
        index_mcp = landmarks[5]  # Base of index finger
        
        # Check if index is bent at the middle joint
        index_bent = (
            index_tip.y > index_pip.y and  # Tip is lower than middle joint
            index_pip.y < index_mcp.y      # Middle joint is higher than base
        )
        
        # Check if other fingers are closed
        other_fingers_closed = (
            landmarks[12].y > landmarks[9].y and  # Middle closed
            landmarks[16].y > landmarks[13].y and  # Ring closed
            landmarks[20].y > landmarks[17].y      # Pinky closed
        )
        
        # Check thumb position
        thumb_tip = landmarks[4]
        thumb_tucked = thumb_tip.x < landmarks[1].x
        
        # Analyze positioning for 'X' sign
        if not index_bent:
            error_text = "Error: Bend your index finger at the middle joint"
        elif not other_fingers_closed:
            error_text = "Error: Close your middle, ring, and pinky fingers"
        elif not thumb_tucked:
            error_text = "Error: Tuck your thumb against your palm"
        else:
            error_text = "Good form! Keep it up"
            correct_position = True
        
        return error_text, correct_position
    
    def get_position_feedback_for_y(self, landmarks):
        """Special strict position checking for letter Y."""
        error_text = "No feedback available"
        correct_position = False
        
        # For 'Y', pinky and thumb are extended, other fingers closed
        thumb_tip = landmarks[4]
        pinky_tip = landmarks[20]
        
        # Check pinky extended
        pinky_extended = pinky_tip.y < landmarks[17].y
        
        # Check thumb extended
        thumb_extended = thumb_tip.y < landmarks[2].y
        
        # Check other fingers are closed
        other_fingers_closed = (
            landmarks[8].y > landmarks[5].y and   # Index closed
            landmarks[12].y > landmarks[9].y and  # Middle closed
            landmarks[16].y > landmarks[13].y     # Ring closed
        )
        
        # Analyze positioning for 'Y' sign
        if not pinky_extended:
            error_text = "Error: Extend your pinky finger"
        elif not thumb_extended:
            error_text = "Error: Extend your thumb"
        elif not other_fingers_closed:
            error_text = "Error: Close your index, middle, and ring fingers"
        else:
            error_text = "Good form! Keep it up"
            correct_position = True
        
        return error_text, correct_position
        
    def get_position_feedback(self, letter, landmarks):
        """Get feedback on hand position for specific letter."""
        # Use specialized feedback methods for each letter
        if letter == 'a':
            error_text = "No feedback available"
            correct_position = False
            
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
                
            return error_text, correct_position
                
        elif letter == 'b':
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
        
        return "No feedback available", False
    
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
        elif letter == 'k':
            return [
                "Hold your hand up with palm facing forward",
                "Extend your index and middle fingers upward",
                "Position your index finger straight up",
                "Position your middle finger at an angle to your index finger",
                "Keep your ring and pinky fingers closed",
                "Extend your thumb"
            ]
        elif letter == 'l':
            return [
                "Extend your thumb to the side",
                "Extend your index finger upward",
                "Keep your other fingers closed",
                "Form a clear 'L' shape with your thumb and index finger",
                "Keep your palm facing forward"
            ]
        elif letter == 'm':
            return [
                "Tuck all fingers into your palm",
                "Tuck your thumb between your fingers",
                "Turn your palm to face downward",
                "Create a compact fist with all digits tucked in"
            ]
        elif letter == 'n':
            return [
                "Tuck your index, middle, ring, and pinky fingers into your palm",
                "Tuck your thumb into your palm",
                "Turn your palm to face downward",
                "Similar to 'M' but with subtle differences in finger positioning"
            ]
        elif letter == 'o':
            return [
                "Curve all fingers to form a circular shape",
                "Touch your thumb and fingertips together",
                "Make a clear 'O' shape with your hand",
                "Keep your palm facing forward"
            ]
        elif letter == 'p':
            return [
                "Point your index finger downward",
                "Extend your thumb to the side",
                "Extend your middle, ring, and pinky fingers",
                "Turn your palm to face sideways"
            ]
        elif letter == 'q':
            return [
                "Point your index finger downward",
                "Position your thumb against your fingers",
                "Curl your middle, ring, and pinky fingers",
                "Turn your palm to face sideways"
            ]
        elif letter == 'r':
            return [
                "Extend your index and middle fingers upward and cross them",
                "Keep your ring and pinky fingers closed",
                "Tuck your thumb against your palm",
                "Your index finger should be in front of your middle finger"
            ]
        elif letter == 's':
            return [
                "Make a fist with all fingers closed",
                "Wrap your thumb across the front of your fist",
                "Keep your palm facing forward"
            ]
        elif letter == 't':
            return [
                "Make a fist with all fingers closed",
                "Place your thumb between your index and middle fingers",
                "Only the tip of your thumb should be visible",
                "Keep your palm facing forward"
            ]
        elif letter == 'u':
            return [
                "Extend your index and middle fingers upward",
                "Keep your index and middle fingers close together",
                "Close your ring and pinky fingers",
                "Tuck your thumb against your palm",
                "Keep your palm facing forward"
            ]
        elif letter == 'v':
            return [
                "Extend your index and middle fingers upward in a V shape",
                "Spread your index and middle fingers apart",
                "Close your ring and pinky fingers",
                "Tuck your thumb against your palm",
                "Keep your palm facing forward"
            ]
        elif letter == 'w':
            return [
                "Extend your index, middle, and ring fingers upward",
                "Spread your fingers slightly to form a 'W' shape",
                "Close your pinky finger",
                "Tuck your thumb against your palm",
                "Keep your palm facing forward"
            ]
        elif letter == 'x':
            return [
                "Bend your index finger at the middle joint",
                "Close your middle, ring, and pinky fingers",
                "Tuck your thumb against your palm",
                "Keep your palm facing forward"
            ]
        elif letter == 'y':
            return [
                "Extend your pinky finger",
                "Extend your thumb",
                "Close your index, middle, and ring fingers",
                "Keep your palm facing forward",
                "Your hand should form a 'hang loose' or 'Y' shape"
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
        print("The detector will automatically recognize signs from either hand")
        print("Press 'l' to toggle between letters A through Y (excluding J and Z)")
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
            
            # Draw target letter text
            cv2.putText(image, f"Target Letter: {self.target_letter.upper()} | Using: Both Hands", 
                        (image.shape[1] - 450, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Initialize a flag to track if any hand was found
            found_hand = False
            best_probability = 0
            best_result_text = ""
            best_color = (0, 0, 255)
            best_confidence_bar_width = 0
            best_confidence_color = (0, 0, 255)
            best_error_text = "No hand detected"
            
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
                    
                    # First determine if the sign is detected based on confidence threshold
                    sign_detected = target_probability >= self.confidence_threshold
                    
                    # If confidence is high enough, override any position feedback
                    if sign_detected:
                        correct_position = True
                        error_text = ""  # Clear any error message
                    
                    # Display result based on confidence
                    if sign_detected:
                        result_text = f"{self.target_letter.upper()} sign detected! ({hand_type.upper()} hand - Confidence: {target_probability:.2f})"
                        color = (0, 255, 0)  # Green for success
                    else:
                        # Only show "not an X sign" message when confidence is truly low
                        # Otherwise show "almost an X sign" for medium confidence
                        if target_probability < 0.5:
                            result_text = f"Not a{' n' if self.target_letter == 'a' else ' '}{self.target_letter.upper()} sign ({hand_type.upper()} hand - Confidence: {target_probability:.2f})"
                            color = (0, 0, 255)  # Red for low confidence
                        else:
                            result_text = f"Almost a{' n' if self.target_letter == 'a' else ' '}{self.target_letter.upper()} sign ({hand_type.upper()} hand - Confidence: {target_probability:.2f})"
                            color = (0, 165, 255)  # Orange for medium confidence
                    
                    # Calculate confidence bar visualization
                    confidence_bar_width = int(300 * target_probability)
                    
                    # Determine bar color based on the same logic as the text
                    if sign_detected:
                        confidence_color = (0, 255, 0)  # Green for success
                    elif target_probability < 0.5:
                        confidence_color = (0, 0, 255)  # Red for low confidence
                    else:
                        confidence_color = (0, 165, 255)  # Orange for medium confidence
                    
                    # Update variables for the best detection
                    found_hand = True
                    if target_probability > best_probability:
                        best_probability = target_probability
                        best_result_text = result_text
                        best_color = color
                        best_confidence_bar_width = confidence_bar_width
                        best_confidence_color = confidence_color
                        best_error_text = error_text
            
            # Display results from the best hand detection
            if found_hand:
                # Draw confidence bar background
                cv2.rectangle(image, (10, 185), (310, 200), (100, 100, 100), -1)
                # Draw filled portion
                cv2.rectangle(image, (10, 185), (10 + best_confidence_bar_width, 200), best_confidence_color, -1)
                # Draw threshold marker
                threshold_x = int(10 + 300 * self.confidence_threshold)
                cv2.line(image, (threshold_x, 180), (threshold_x, 205), (255, 255, 255), 2)
                
                cv2.putText(image, best_result_text, (10, 175), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, best_color, 2)
                
                # Display error/guidance text
                if best_error_text:  # Only display error text if there's an actual message
                    # Set color based on whether it's an error or just guidance
                    error_color = (0, 0, 255) if best_error_text.startswith("Error") else (0, 165, 255)
                    cv2.putText(image, best_error_text, (10, 140), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, error_color, 2)
            else:
                # No hand detected
                cv2.putText(image, "No hand detected", (10, 175), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Add instructions to screen
            cv2.putText(image, "Press 'l': Toggle letter (A-Y) | 'q' or ESC: Quit", 
                        (10, image.shape[0] - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display the image
            cv2.imshow('ASL Sign Detector', image)
            
            # Process key presses
            key = cv2.waitKey(5) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or q key
                break
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
    
    print("\nASL Sign Detector for A through Y (excluding J and Z)")
    print("================================================")
    print("This detector will automatically recognize signs from either hand")
    
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
            detector.load_model()
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
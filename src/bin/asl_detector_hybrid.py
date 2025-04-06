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
    parser = argparse.ArgumentParser(description='ASL Sign Language Detection using Hybrid Approach')
    parser.add_argument('--models_dir', type=str, default='./webcam_models',
                        help='Directory containing trained models')
    parser.add_argument('--camera_index', type=int, default=1,
                        help='Camera index to use (default: 1)')
    parser.add_argument('--rf_model', type=str, default='./improved_models/asl_randomforest.joblib', 
                        help='Random Forest model path')
    parser.add_argument('--xgb_model', type=str, default='./improved_models/asl_xgboost.joblib', 
                        help='XGBoost model path')
    parser.add_argument('--label_encoder', type=str, default='./improved_models/label_encoder.joblib', 
                        help='Label encoder path')
    parser.add_argument('--confidence_threshold', type=float, default=0.8,
                        help='Confidence threshold for detection (default: 0.8)')
    parser.add_argument('--use_preprocessing', action='store_true',
                        help='Apply preprocessing to images')
    parser.add_argument('--hand', choices=['right', 'left', 'both'], default='both',
                        help='Which hand to use for detection (default: both)')
    parser.add_argument('--target_letter', type=str, default='a', choices=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'],
                        help='Target letter to practice forming (default: a)')
    parser.add_argument('--rule_weight', type=float, default=0.3,
                        help='Weight for rule-based detection (0-1, default: 0.3)')
    parser.add_argument('--rf_weight', type=float, default=0.3,
                        help='Weight for Random Forest model (0-1, default: 0.3)')
    parser.add_argument('--xgb_weight', type=float, default=0.4,
                        help='Weight for XGBoost model (0-1, default: 0.4)')
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

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two landmarks."""
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def calculate_angle(p1, p2, p3):
    """Calculate angle between three landmarks."""
    vector1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    vector2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    
    unit_vector1 = vector1 / np.linalg.norm(vector1)
    unit_vector2 = vector2 / np.linalg.norm(vector2)
    
    dot_product = np.clip(np.dot(unit_vector1, unit_vector2), -1.0, 1.0)
    angle = np.arccos(dot_product)
    return angle * 180 / np.pi

def extract_features(landmarks, is_right_hand):
    """Extract features from hand landmarks for the model."""
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
    
    # Calculate hand size for normalization
    hand_size = 0
    for i in range(21):
        distance = calculate_distance(wrist, landmarks[i])
        if distance > hand_size:
            hand_size = distance
    
    # Add pairwise distances between fingertips (normalized by hand size)
    finger_tips = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky
    for i in range(len(finger_tips)):
        for j in range(i+1, len(finger_tips)):
            distance = calculate_distance(landmarks[finger_tips[i]], landmarks[finger_tips[j]])
            features.append(distance / hand_size if hand_size > 0 else 0)
    
    # Add angles between finger segments
    for finger in range(1, 6):  # 1=thumb, 2=index, 3=middle, 4=ring, 5=pinky
        # Get the base, middle and tip landmarks for each finger
        if finger == 1:  # thumb
            base, mid, tip = 1, 2, 4
        else:
            base = (finger - 2) * 4 + 5  # 5, 9, 13, 17
            mid = base + 1
            tip = base + 3
        
        # Calculate angle
        angle = calculate_angle(landmarks[base], landmarks[mid], landmarks[tip])
        features.append(angle / 180.0)  # Normalize to 0-1 range
    
    # Ensure we have exactly 94 features (fill with zeros if needed)
    while len(features) < 94:
        features.append(0.0)
    
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
        thumb_distance = calculate_distance(thumb_tip, index_mcp)
        
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
        if calculate_distance(landmarks[8], landmarks[12]) > 0.1:
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
        # Check for C shape
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        pinky_tip = landmarks[20]
        
        # Thumb should be positioned to the left of index finger
        if thumb_tip.x >= index_tip.x:
            feedback["finger_issues"].append("Position your thumb to the left to form the C shape")
        
        # Calculate average finger curl to detect curved position
        finger_bases = [5, 9, 13, 17]  # Index, middle, ring, pinky bases
        finger_tips = [8, 12, 16, 20]  # Index, middle, ring, pinky tips
        
        finger_curl_angles = []
        for i in range(4):
            base = finger_bases[i]
            mid = base + 1
            tip = finger_tips[i]
            angle = calculate_angle(landmarks[base], landmarks[mid], landmarks[tip])
            finger_curl_angles.append(angle)
        
        avg_curl = np.mean(finger_curl_angles)
        if avg_curl <= 100:
            feedback["finger_issues"].append("Don't curl your fingers too much, keep them slightly curved")
        elif avg_curl >= 160:
            feedback["finger_issues"].append("Curve your fingers more to form a C shape")
        
        # Set overall message
        if not feedback["finger_issues"]:
            feedback["correct"] = True
            feedback["message"] = "Perfect 'C' sign!"
        else:
            feedback["message"] = "Adjust your 'C' sign:"
    
    elif letter == 'l':
        # Check if thumb is extended to the side
        thumb_tip = landmarks[4]
        wrist = landmarks[0]
        if thumb_tip.y >= wrist.y:
            feedback["finger_issues"].append("Extend your thumb outward")
        if thumb_tip.x >= wrist.x:
            feedback["finger_issues"].append("Position your thumb more to the left")
        
        # Check index finger is extended up
        if landmarks[8].y >= landmarks[5].y:
            feedback["finger_issues"].append("Extend your index finger upward")
        
        # Check other fingers are curled
        if landmarks[12].y < landmarks[9].y:
            feedback["finger_issues"].append("Curl your middle finger into your palm")
        if landmarks[16].y < landmarks[13].y:
            feedback["finger_issues"].append("Curl your ring finger into your palm")
        if landmarks[20].y < landmarks[17].y:
            feedback["finger_issues"].append("Curl your pinky finger into your palm")
        
        # Set overall message
        if not feedback["finger_issues"]:
            feedback["correct"] = True
            feedback["message"] = "Perfect 'L' sign!"
        else:
            feedback["message"] = "Adjust your 'L' sign:"
    
    elif letter == 'o':
        # Check if fingers form a circle
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        # Thumb and index should be close to form a circle
        thumb_index_distance = calculate_distance(thumb_tip, index_tip)
        if thumb_index_distance > 0.05:
            feedback["finger_issues"].append(f"Bring your thumb and index finger closer (distance: {thumb_index_distance:.2f})")
        
        # Fingers should be together
        finger_tips = [8, 12, 16, 20]  # Fingertips
        for i in range(len(finger_tips)-1):
            distance = calculate_distance(landmarks[finger_tips[i]], landmarks[finger_tips[i+1]])
            if distance > 0.1:
                feedback["finger_issues"].append("Keep your fingertips together to form a circle")
                break
        
        # Set overall message
        if not feedback["finger_issues"]:
            feedback["correct"] = True
            feedback["message"] = "Perfect 'O' sign!"
        else:
            feedback["message"] = "Adjust your 'O' sign:"
    
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
            "Curve your fingers and thumb to form a C shape",
            "Keep your fingers together, not spread apart",
            "Palm should face to the side",
            "Thumb and fingers should be curved, not bent at sharp angles"
        ],
        'd': [
            "Make a circle with your thumb and middle finger",
            "Extend your index finger straight up",
            "Curl your ring and pinky fingers into your palm",
            "Keep your index finger pointing upward"
        ],
        'e': [
            "Curl all fingers into your palm",
            "Tuck your thumb across your fingers",
            "Keep palm facing forward",
            "All fingertips should be hidden"
        ],
        'l': [
            "Extend your thumb out to the side",
            "Extend your index finger straight up",
            "Curl your middle, ring, and pinky fingers into your palm",
            "Form a right angle between your thumb and index finger"
        ],
        'o': [
            "Form a circle with all five fingertips touching",
            "Keep fingers together, not spread apart",
            "Make sure thumb touches index finger to close the circle",
            "Palm should face forward"
        ],
        'v': [
            "Extend your index and middle fingers in a V-shape",
            "Keep your fingers spread apart",
            "Curl your ring and pinky fingers into your palm",
            "Tuck your thumb against your palm"
        ],
        'y': [
            "Extend your thumb and pinky finger",
            "Curl your index, middle, and ring fingers into your palm",
            "Keep your thumb and pinky spread apart",
            "Palm should face slightly inward"
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
            
            # For combined models, use a special key
            models['combined'] = model
            
            # If the model has classes_ attribute, extract characters from it
            if hasattr(model, 'classes_'):
                print(f"Model can predict these characters: {', '.join(model.classes_)}")
            
            return models
        except Exception as e:
            print(f"Error loading combined model: {str(e)}")
    
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
            # Extract character from filename
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
            print(f"Loaded model for character: '{char}' from {os.path.basename(model_file)}")
        except Exception as e:
            print(f"Error loading model {model_file}: {str(e)}")
    
    return models 

class HybridASLDetector:
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
            min_tracking_confidence=0.5,
            model_complexity=1  # Higher complexity for better accuracy
        )
        
        # XGBoost model only
        self.xgb_model = None
        self.label_encoder = None
        
        # Paths for individual models
        self.xgb_model_path = "./improved_models/asl_xgboost_tuned.joblib"
        self.label_encoder_path = "./improved_models/label_encoder.joblib"
        
        # Target letter and hand settings
        self.target_letter = 'a'  # Default to letter 'a'
        self.target_hand = "right"  # Default to right hand
        self.confidence_threshold = 0.5  # Lower threshold for better detection
        
        # Enable rule-based checking
        self.rule_based_enabled = True
        
        # Detection stability
        self.last_predictions = []
        self.prediction_history_size = 5
        self.stable_threshold = 0.6  # 60% of recent predictions must match
        
        # Load models
        self.load_models()
    
    def load_models(self):
        """Load XGBoost model for detection."""
        try:
            if os.path.exists(self.xgb_model_path):
                self.xgb_model = joblib.load(self.xgb_model_path)
                print(f"Successfully loaded XGBoost model from {self.xgb_model_path}")
            else:
                print(f"XGBoost model not found at {self.xgb_model_path}")
            
            if os.path.exists(self.label_encoder_path):
                self.label_encoder = joblib.load(self.label_encoder_path)
                print(f"Successfully loaded Label Encoder from {self.label_encoder_path}")
                if self.label_encoder is not None:
                    print(f"Available classes: {', '.join(self.label_encoder.classes_)}")
            else:
                print(f"Label encoder not found at {self.label_encoder_path}")
                
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def get_hand_type(self, hand_landmarks, results):
        """Determine if the hand is left or right."""
        handedness = results.multi_handedness
        for idx, classification in enumerate(handedness):
            if idx == results.multi_hand_landmarks.index(hand_landmarks):
                return classification.classification[0].label == "Right"
        return True  # Default to right if can't determine
    
    def rule_based_check(self, landmarks, letter):
        """Perform rule-based checks for specific letters."""
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
            
            # Check if thumb is positioned alongside the index finger
            thumb_alongside_index = thumb_tip.x >= index_base.x
            
            # Check if thumb is pointing upward
            thumb_pointing_up = thumb_tip.y <= landmarks[3].y  # compared to thumb IP
            
            return fingers_closed and thumb_alongside_index and thumb_pointing_up
            
        elif letter == 'b':
            # Check if fingers are extended
            fingers_extended = (
                landmarks[8].y < landmarks[5].y and  # Index finger
                landmarks[12].y < landmarks[9].y and  # Middle finger
                landmarks[16].y < landmarks[13].y and  # Ring finger
                landmarks[20].y < landmarks[17].y  # Pinky
            )
            
            # Check if thumb is tucked
            thumb_tucked = landmarks[4].x > landmarks[9].x  # Thumb to middle finger base
            
            return fingers_extended and thumb_tucked
        
        elif letter == 'c':
            # Check for C shape
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            
            # Thumb should be to the left of index
            thumb_position = thumb_tip.x < index_tip.x
            
            # Calculate finger curl
            finger_bases = [5, 9, 13, 17]  # Finger bases
            finger_tips = [8, 12, 16, 20]  # Fingertips
            
            finger_curl_angles = []
            for i in range(4):
                base = finger_bases[i]
                mid = base + 1
                tip = finger_tips[i]
                angle = calculate_angle(landmarks[base], landmarks[mid], landmarks[tip])
                finger_curl_angles.append(angle)
            
            avg_curl = np.mean(finger_curl_angles)
            fingers_curved = avg_curl > 100 and avg_curl < 160
            
            return thumb_position and fingers_curved
        
        elif letter == 'd':
            # Check if index finger is extended upward
            index_extended = landmarks[8].y < landmarks[5].y
            
            # Check if other fingers are curled
            other_fingers_curled = (
                landmarks[12].y > landmarks[9].y and  # Middle finger
                landmarks[16].y > landmarks[13].y and  # Ring finger
                landmarks[20].y > landmarks[17].y  # Pinky finger
            )
            
            # Check if thumb and middle finger form a circle
            thumb_tip = landmarks[4]
            middle_tip = landmarks[12]
            
            # Distance between thumb and middle fingertips - should be small for the circle
            thumb_middle_dist = calculate_distance(thumb_tip, middle_tip)
            
            # Check for proper D shape with index finger pointing up
            # and other fingers curled, with thumb-middle forming a circle
            thumb_middle_touching = thumb_middle_dist < 0.08
            
            return index_extended and other_fingers_curled and thumb_middle_touching
        
        return False  # Default if no rules match
    
    def extract_features(self, landmarks, is_right_hand=True):
        """Extract features from hand landmarks for the model."""
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
            
        # Make sure we have exactly 94 features - not more, not less
        if len(features) > 94:
            features = features[:94]  # Truncate to exactly 94 features
        else:
            # Add padding if needed
            while len(features) < 94:
                features.append(0.0)
        
        return features
    
    def toggle_target_hand(self):
        """Toggle between left and right hand."""
        if self.target_hand == "right":
            self.target_hand = "left"
        elif self.target_hand == "left":
            self.target_hand = "both"
        else:
            self.target_hand = "right"
        return self.target_hand
    
    def toggle_target_letter(self):
        """Toggle between supported letters."""
        available_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
        current_index = available_letters.index(self.target_letter)
        next_index = (current_index + 1) % len(available_letters)
        self.target_letter = available_letters[next_index]
        return self.target_letter
    
    def setup_camera(self):
        """Set up and return a camera capture object."""
        return setup_camera(1)
    
    def get_position_feedback(self, letter, landmarks):
        """Get feedback on hand position for specific letter."""
        # Use specialized feedback methods for different letters
        if letter == 'd':
            return self.get_position_feedback_for_d(landmarks)
        
        # For other letters, use the general feedback method
        feedback = get_asl_position_feedback(letter, landmarks)
        
        if feedback["correct"]:
            return "Good form! Keep it up", True
        elif feedback["finger_issues"]:
            # Return the first issue for now (could return more)
            return f"Error: {feedback['finger_issues'][0]}", False
        else:
            return f"Adjust your {letter.upper()} sign", False
    
    def get_position_feedback_for_d(self, landmarks):
        """Special strict position checking for letter D."""
        error_text = "No feedback available"
        correct_position = False
        
        # Check if thumb and middle finger form a circle
        thumb_tip = landmarks[4]
        middle_tip = landmarks[12]
        
        # Distance between thumb and middle tips - should be small for the circle
        thumb_middle_dist = calculate_distance(thumb_tip, middle_tip)
        
        # Check if index finger is extended upward
        index_extended = landmarks[8].y < landmarks[5].y
        
        # Check if other fingers are curled
        other_fingers_curled = (
            landmarks[16].y > landmarks[13].y and  # Ring finger
            landmarks[20].y > landmarks[17].y  # Pinky finger
        )
        
        # Very strict checking for D
        if not index_extended:
            error_text = "Error: Extend your index finger straight up for 'D'"
        elif not other_fingers_curled:
            error_text = "Error: Curl your ring and pinky fingers into your palm"
        elif thumb_middle_dist > 0.08:
            error_text = f"Error: Touch your thumb to your middle finger (distance: {thumb_middle_dist:.2f})"
        else:
            error_text = "Good form! Keep it up"
            correct_position = True
        
        return error_text, correct_position
    
    def get_letter_instructions(self, letter):
        """Return instructions for the current target letter."""
        return get_asl_instructions(letter)
    
    def draw_semitransparent_rect(self, image, start_point, end_point, color, alpha=0.3):
        """Draw a semi-transparent rectangle on the image."""
        return draw_semitransparent_rect(image, start_point, end_point, color, alpha)
    
    def predict_with_models(self, features, letter=None, landmarks=None):
        """Make predictions using only XGBoost model."""
        # Apply rule-based detection for specific letter if provided
        if letter and landmarks and self.rule_based_enabled:
            rule_result = self.rule_based_check(landmarks, letter)
            if rule_result:
                # If the rule matches, return with high confidence
                return letter, 0.9
        
        # Make sure XGBoost model and label encoder are available
        if self.xgb_model is None or self.label_encoder is None:
            print("XGBoost model or label encoder not loaded")
            return None, 0
        
        try:
            # Ensure exactly 94 features
            if len(features) != 94:
                if len(features) > 94:
                    features = features[:94]
                else:
                    while len(features) < 94:
                        features.append(0.0)
                        
            # Get predictions from XGBoost model
            predictions = self.xgb_model.predict_proba([features])[0]
            
            # Find the letter with highest confidence
            best_index = np.argmax(predictions)
            best_letter = self.label_encoder.classes_[best_index]
            confidence = predictions[best_index]
            
            return best_letter, confidence
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None, 0
    
    def detect_sign(self):
        """Run real-time detection to identify ASL signs."""
        if self.xgb_model is None or self.label_encoder is None:
            print("No models loaded. Please load at least one model first.")
            return
        
        print(f"Starting sign detection mode...")
        print(f"Current target letter: {self.target_letter.upper()}")
        print(f"Current target hand: {self.target_hand.upper()}")
        print("Press 'l' to toggle between letters")
        print("Press 'h' to toggle between left/right hand")
        print("Press 'q' or ESC to quit")
        
        # Setup camera
        cap = self.setup_camera()
        if cap is None:
            print("Failed to setup camera. Exiting detection.")
            return
        
        # Guidance title
        guide_title = f"How to form the '{self.target_letter.upper()}' sign correctly:"
        
        # Stable prediction and confidence
        stable_prediction = None
        stable_confidence = 0
        
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
                        
                        # Use our hybrid prediction method
                        prediction, confidence = self.predict_with_models(
                            features, self.target_letter, hand_landmarks.landmark
                        )
                        
                        # Add to prediction history for stability
                        if prediction is not None:
                            self.last_predictions.append(prediction)
                            # Keep history at fixed size
                            if len(self.last_predictions) > self.prediction_history_size:
                                self.last_predictions.pop(0)
                            
                            # Get the most common prediction in history
                            if self.last_predictions:
                                predictions_count = {}
                                for pred in self.last_predictions:
                                    predictions_count[pred] = predictions_count.get(pred, 0) + 1
                                
                                most_common = max(predictions_count.items(), key=lambda x: x[1])
                                if most_common[1] / len(self.last_predictions) >= self.stable_threshold:
                                    stable_prediction = most_common[0]
                                    stable_confidence = confidence
                                    print(f"Stable Prediction: {stable_prediction}, Confidence: {confidence:.2f}")
                        
                        # Display result based on stable confidence
                        sign_detected = stable_prediction == self.target_letter and stable_confidence >= self.confidence_threshold
                        
                        # Display result based on stable confidence
                        if sign_detected:
                            result_text = f"{self.target_letter.upper()} sign detected! (Confidence: {stable_confidence:.2f})"
                            color = (0, 255, 0)  # Green for success
                            error_text = ""  # Clear any error message
                        else:
                            # Not detected or wrong letter
                            if stable_confidence < 0.5:
                                result_text = f"Not a{' n' if self.target_letter == 'a' else ' '}{self.target_letter.upper()} sign"
                                if stable_prediction:
                                    result_text += f" (Detected: {stable_prediction.upper()}, {stable_confidence:.2f})"
                                color = (0, 0, 255)  # Red for low confidence
                            else:
                                result_text = f"Almost a{' n' if self.target_letter == 'a' else ' '}{self.target_letter.upper()} sign"
                                if stable_prediction and stable_prediction != self.target_letter:
                                    result_text += f" (Looks like: {stable_prediction.upper()}, {stable_confidence:.2f})"
                                color = (0, 165, 255)  # Orange for medium confidence
                        
                        # Add confidence bar visualization
                        confidence_bar_width = int(300 * stable_confidence)
                        
                        # Determine bar color based on the same logic as the text
                        if stable_prediction == self.target_letter and stable_confidence >= self.confidence_threshold:
                            confidence_color = (0, 255, 0)  # Green for success
                        elif stable_confidence < 0.5:
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
                        
                        # Display prediction text
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
            cv2.imshow('ASL Hybrid Detector', image)
            
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
    args = parse_args()
    detector = HybridASLDetector()
    
    # Set paths and settings from arguments
    detector.target_letter = args.target_letter
    detector.target_hand = args.hand
    detector.confidence_threshold = args.confidence_threshold
    detector.xgb_model_path = args.xgb_model
    detector.label_encoder_path = args.label_encoder
    
    print("\nXGBoost ASL Sign Detector")
    print("============================")
    
    # Load models
    detector.load_models()
    
    # Start detection directly
    if detector.xgb_model is not None and detector.label_encoder is not None:
        detector.detect_sign()
    else:
        print("XGBoost model or label encoder not loaded. Please check paths.")

if __name__ == "__main__":
    main() 
import cv2
import mediapipe as mp
import numpy as np
import os
import joblib
import time
import argparse
import json
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def parse_args():
    parser = argparse.ArgumentParser(description='ASL Model Training with Webcam (Improved Features)')
    parser.add_argument('--mode', choices=['collect', 'train'], default='collect',
                        help='Program mode: "collect" for data collection or "train" for model training')
    parser.add_argument('--data_dir', type=str, default='./webcam_data',
                        help='Directory to save/load collected feature data')
    parser.add_argument('--model_output', type=str, default='./asl_combined_model_improved.joblib',
                        help='Output file for the trained model')
    parser.add_argument('--character', type=str, 
                        help='Specific character to collect data for (in collect mode)')
    parser.add_argument('--hand', choices=['right', 'left', 'both'], default='right',
                        help='Which hand to use for collection/detection (default: right)')
    parser.add_argument('--camera_index', type=int, default=0,
                        help='Camera index to use (default: 0)')
    parser.add_argument('--use_preprocessing', action='store_true',
                        help='Apply preprocessing to images')
    parser.add_argument('--samples_per_char', type=int, default=30,
                        help='Minimum samples to collect per character (default: 30)')
    return parser.parse_args()

def calculate_angle_between_vectors(v1, v2):
    """Calculate angle between two vectors in radians."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Handle zero division
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    
    # Ensure value is in valid range for arccos
    cos_angle = min(1.0, max(-1.0, dot_product / (norm_v1 * norm_v2)))
    return np.arccos(cos_angle)

def calculate_palm_normal(wrist, index_mcp, pinky_mcp):
    """Calculate normal vector to the palm plane."""
    v1 = np.array([index_mcp.x - wrist.x, index_mcp.y - wrist.y, index_mcp.z - wrist.z])
    v2 = np.array([pinky_mcp.x - wrist.x, pinky_mcp.y - wrist.y, pinky_mcp.z - wrist.z])
    
    # Cross product gives normal vector
    normal = np.cross(v1, v2)
    
    # Normalize the vector
    norm = np.linalg.norm(normal)
    if norm == 0:
        return np.array([0, 0, 1])  # Default if degenerate
    return normal / norm

def perpendicular_distance_to_plane(point, plane_normal, plane_point):
    """Calculate perpendicular distance from point to plane."""
    v = np.array([point.x - plane_point.x, point.y - plane_point.y, point.z - plane_point.z])
    return abs(np.dot(v, plane_normal))

def extract_features(landmarks, is_right_hand):
    """Extract enhanced features from hand landmarks for the model."""
    features = []
    
    # Add a feature to indicate hand type (0 for left, 1 for right)
    features.append(1.0 if is_right_hand else 0.0)
    
    # Calculate hand bounding box size for normalization
    min_x = min(landmark.x for landmark in landmarks)
    max_x = max(landmark.x for landmark in landmarks)
    min_y = min(landmark.y for landmark in landmarks)
    max_y = max(landmark.y for landmark in landmarks)
    
    hand_width = max_x - min_x
    hand_height = max_y - min_y
    hand_size = max(hand_width, hand_height)  # Use max dimension for normalization
    
    # Handle case where hand size is too small (avoid division by zero)
    if hand_size < 0.01:
        hand_size = 0.01
    
    # Add normalized landmark positions (relative to wrist and normalized by hand size)
    wrist = landmarks[0]
    for landmark in landmarks:
        # Add normalized position
        features.append((landmark.x - wrist.x) / hand_size)
        features.append((landmark.y - wrist.y) / hand_size)
        features.append(landmark.z)  # Keep z as is, as it's already normalized by MediaPipe
    
    # Still include raw landmarks for compatibility with existing code
    for landmark in landmarks:
        features.extend([landmark.x, landmark.y, landmark.z])
    
    # Key points for feature calculation
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    
    # Finger base positions
    thumb_mcp = landmarks[1]
    index_mcp = landmarks[5]
    middle_mcp = landmarks[9]
    ring_mcp = landmarks[13]
    pinky_mcp = landmarks[17]
    
    # Calculate palm normal vector (for measuring how fingers bend relative to palm)
    palm_normal = calculate_palm_normal(wrist, index_mcp, pinky_mcp)
    
    # Distance between thumb and each fingertip (normalized)
    for finger_tip in [index_tip, middle_tip, ring_tip, pinky_tip]:
        dist = np.sqrt(
            (thumb_tip.x - finger_tip.x)**2 + 
            (thumb_tip.y - finger_tip.y)**2 +
            (thumb_tip.z - finger_tip.z)**2
        ) / hand_size
        features.append(dist)
    
    # Distance between adjacent fingertips (normalized)
    fingertips = [index_tip, middle_tip, ring_tip, pinky_tip]
    for i in range(len(fingertips)-1):
        dist = np.sqrt(
            (fingertips[i].x - fingertips[i+1].x)**2 + 
            (fingertips[i].y - fingertips[i+1].y)**2 +
            (fingertips[i].z - fingertips[i+1].z)**2
        ) / hand_size
        features.append(dist)
    
    # Check if fingers are closed (making a fist)
    # This uses the vertical position relative to the MCP joint
    index_closed = 1 if index_tip.y > index_mcp.y else 0
    middle_closed = 1 if middle_tip.y > middle_mcp.y else 0
    ring_closed = 1 if ring_tip.y > ring_mcp.y else 0
    pinky_closed = 1 if pinky_tip.y > pinky_mcp.y else 0
    
    features.extend([index_closed, middle_closed, ring_closed, pinky_closed])
    
    # Calculate angles between finger segments
    # For each finger, calculate the angles at the two joints
    for finger_idx, start_idx in [(1, 1), (2, 5), (3, 9), (4, 13), (5, 17)]:
        # Proximal joint angle
        v1 = [landmarks[start_idx].x - landmarks[0].x, 
              landmarks[start_idx].y - landmarks[0].y]
        v2 = [landmarks[start_idx+1].x - landmarks[start_idx].x, 
              landmarks[start_idx+1].y - landmarks[start_idx].y]
        angle = calculate_angle_between_vectors(v1, v2)
        features.append(angle)
        
        # Middle joint angle
        v1 = [landmarks[start_idx+1].x - landmarks[start_idx].x, 
              landmarks[start_idx+1].y - landmarks[start_idx].y]
        v2 = [landmarks[start_idx+2].x - landmarks[start_idx+1].x, 
              landmarks[start_idx+2].y - landmarks[start_idx+1].y]
        angle = calculate_angle_between_vectors(v1, v2)
        features.append(angle)
        
        # Distal joint angle
        v1 = [landmarks[start_idx+2].x - landmarks[start_idx+1].x, 
              landmarks[start_idx+2].y - landmarks[start_idx+1].y]
        v2 = [landmarks[start_idx+3].x - landmarks[start_idx+2].x, 
              landmarks[start_idx+3].y - landmarks[start_idx+2].y]
        angle = calculate_angle_between_vectors(v1, v2)
        features.append(angle)
    
    # Calculate angles between fingertips and wrist
    # This helps distinguish letters where finger orientation matters
    for finger_tip in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]:
        v = [finger_tip.x - wrist.x, finger_tip.y - wrist.y]
        # Calculate angle with vertical
        vertical = [0, -1]
        angle = calculate_angle_between_vectors(v, vertical)
        features.append(angle)
    
    # Add perpendicular distance from each fingertip to the palm plane
    for finger_tip in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]:
        dist = perpendicular_distance_to_plane(finger_tip, palm_normal, wrist) / hand_size
        features.append(dist)
    
    return features

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

def get_hand_type(hand_idx, results):
    """Determine if the hand is left or right."""
    handedness = results.multi_handedness[hand_idx].classification[0].label
    return handedness == "Right"  # True for right hand, False for left hand

def setup_camera(camera_index):
    """Set up and return a camera capture object, trying multiple indices if needed."""
    print("Trying to initialize camera...")
    cap = None
    
    # Try several camera indices
    for cam_index in [camera_index, 0, 1, -1, 2, 3]:
        if cam_index != camera_index and cam_index in [0, camera_index]:
            continue  # Skip duplicates
            
        print(f"Trying camera index {cam_index}...")
        cap = cv2.VideoCapture(cam_index)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None and test_frame.size > 0:
                print(f"Successfully opened camera at index {cam_index}")
                break
            else:
                print(f"Camera {cam_index} opened but couldn't read frames")
                cap.release()
                cap = None
        else:
            print(f"Failed to open camera {cam_index}")
    
    if cap is None:
        print("Error: Could not open any camera")
    
    return cap

def save_feature_data(character, X_positive_right, X_positive_left, X_negative, data_dir):
    """Save collected feature data to a file."""
    os.makedirs(data_dir, exist_ok=True)
    
    data_file = os.path.join(data_dir, f"features_{character}.pkl")
    
    # Prepare data dictionary
    data = {
        'character': character,
        'X_positive_right': X_positive_right,
        'X_positive_left': X_positive_left,
        'X_negative': X_negative,
        'timestamp': time.time()
    }
    
    # Save using pickle
    with open(data_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Saved {len(X_positive_right)} right hand, {len(X_positive_left)} left hand positive and {len(X_negative)} negative samples for '{character}' to {data_file}")
    return data_file

def load_feature_data(character, data_dir):
    """Load feature data from a file."""
    data_file = os.path.join(data_dir, f"features_{character}.pkl")
    
    if not os.path.exists(data_file):
        return None, None, None
    
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    # Check if we have the new format (with separate right/left hands)
    if 'X_positive_right' in data and 'X_positive_left' in data:
        print(f"Loaded {len(data['X_positive_right'])} right hand, {len(data['X_positive_left'])} left hand positive and {len(data['X_negative'])} negative samples for '{character}'")
        return data['X_positive_right'], data['X_positive_left'], data['X_negative']
    # Handle old format for backward compatibility
    elif 'X_positive' in data:
        print(f"Loaded {len(data['X_positive'])} positive (old format) and {len(data['X_negative'])} negative samples for '{character}'")
        return data['X_positive'], [], data['X_negative']
    else:
        return None, None, None

def collect_data(character, hand_mode='right', min_samples=30, camera_index=0, use_preprocessing=False):
    """Collect training data for a specific character."""
    print(f"Collecting data for character: {character}")
    print(f"Hand mode: {hand_mode} (right, left, or both)")
    print(f"Please make the sign for '{character.upper()}' with your {'right or left' if hand_mode == 'both' else hand_mode} hand")
    print(f"Minimum samples to collect: {min_samples} {'per hand' if hand_mode == 'both' else ''}")
    print("Press SPACE to capture a positive sample")
    print("Press 'n' to capture a negative sample")
    print("Press 'p' to toggle preprocessing")
    print("Press 'q' or ESC to finish collection")
    
    # Initialize MediaPipe hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Initialize hands object
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Setup camera
    cap = setup_camera(camera_index)
    if cap is None:
        return None, None, None
    
    # Prepare data storage
    X_positive_right = []  # Features for right hand positive samples
    X_positive_left = []   # Features for left hand positive samples
    X_negative = []        # Features for negative samples (either hand)
    
    # Variable for UI
    collecting = True
    
    # Target count to reach
    target_count = min_samples if hand_mode != 'both' else min_samples * 2
    
    while collecting and cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Failed to capture frame from camera")
            continue
        
        # Flip horizontally for selfie view
        image = cv2.flip(image, 1)
        
        # Create a copy for display
        display_image = image.copy()
        
        # Decide whether to use preprocessing
        if use_preprocessing:
            processed_image = preprocess_image(image)
            image_to_process = processed_image
            
            # Show small preview
            small_preview = cv2.resize(processed_image, (160, 120))
            display_image[10:10+120, 10:10+160] = small_preview
            cv2.rectangle(display_image, (10, 10), (10+160, 10+120), (255, 255, 255), 1)
            cv2.putText(display_image, "Processed", (15, 25),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            image_to_process = image
        
        # Process the frame with MediaPipe
        image_rgb = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        # Create a dark overlay for text area
        text_overlay = display_image.copy()
        cv2.rectangle(text_overlay, (0, display_image.shape[0]-200), 
                     (display_image.shape[1], display_image.shape[0]), (0, 0, 0), -1)
        display_image = cv2.addWeighted(text_overlay, 0.3, display_image, 0.7, 0)
        
        # Show preprocessing status
        status_text = f"Preprocessing: {'ON' if use_preprocessing else 'OFF'} | Hand Mode: {hand_mode.upper()}"
        cv2.putText(display_image, status_text, (display_image.shape[1] - 350, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show collection status
        cv2.putText(display_image, f"Collecting for: {character.upper()} (Improved Features)", 
                   (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        if hand_mode == 'right' or hand_mode == 'both':
            cv2.putText(display_image, f"Right hand samples: {len(X_positive_right)}/{min_samples}", 
                       (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        if hand_mode == 'left' or hand_mode == 'both':
            cv2.putText(display_image, f"Left hand samples: {len(X_positive_left)}/{min_samples}", 
                       (20, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.putText(display_image, f"Negative samples: {len(X_negative)}", 
                   (20, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Display instructions
        cv2.putText(display_image, "SPACE: Capture sample as positive", 
                   (20, display_image.shape[0] - 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        cv2.putText(display_image, "n: Capture sample as negative", 
                   (20, display_image.shape[0] - 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        cv2.putText(display_image, "p: Toggle preprocessing", 
                   (20, display_image.shape[0] - 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        cv2.putText(display_image, "h: Toggle hand mode (right/left/both)", 
                   (20, display_image.shape[0] - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        cv2.putText(display_image, "q/ESC: Finish collection", 
                   (20, display_image.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        # Process hand landmarks if detected
        detected_hands = []
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                is_right_hand = handedness.classification[0].label == "Right"
                hand_info = {
                    'landmarks': hand_landmarks,
                    'is_right_hand': is_right_hand,
                    'features': extract_features(hand_landmarks.landmark, is_right_hand)
                }
                detected_hands.append(hand_info)
                
                # Draw hand landmarks with different colors for left/right
                hand_color = (0, 255, 0) if is_right_hand else (0, 0, 255)  # Green for right, red for left
                
                mp_drawing.draw_landmarks(
                    display_image, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing.DrawingSpec(color=hand_color, thickness=2, circle_radius=4)
                )
                
                # Label the hand
                hand_text = "RIGHT" if is_right_hand else "LEFT"
                text_pos = (int(hand_landmarks.landmark[0].x * display_image.shape[1]), 
                           int(hand_landmarks.landmark[0].y * display_image.shape[0] - 20))
                cv2.putText(display_image, hand_text, text_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, hand_color, 2)
        
        # Show guidance if no hand detected
        if not detected_hands:
            cv2.putText(display_image, "No hand detected", 
                       (display_image.shape[1]//2 - 100, display_image.shape[0]//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # Show the image
        cv2.imshow(f'ASL Training - Character {character.upper()} (Improved)', display_image)
        
        # Check for key presses
        key = cv2.waitKey(5) & 0xFF
        
        if key == 27 or key == ord('q'):  # ESC or q key
            collecting = False
        elif key == ord('p'):  # Toggle preprocessing
            use_preprocessing = not use_preprocessing
            print(f"Preprocessing: {'ON' if use_preprocessing else 'OFF'}")
        elif key == ord('h'):  # Toggle hand mode
            if hand_mode == 'right':
                hand_mode = 'left'
            elif hand_mode == 'left':
                hand_mode = 'both'
            else:
                hand_mode = 'right'
            print(f"Hand mode: {hand_mode}")
        elif key == 32 and detected_hands:  # SPACE - capture positive sample
            for hand in detected_hands:
                if hand_mode == 'both' or (hand_mode == 'right' and hand['is_right_hand']) or (hand_mode == 'left' and not hand['is_right_hand']):
                    # Add as positive sample
                    if hand['is_right_hand']:
                        if hand_mode == 'right' or hand_mode == 'both':
                            X_positive_right.append(hand['features'])
                            print(f"Captured positive sample #{len(X_positive_right)} for '{character}' with RIGHT hand")
                    else:
                        if hand_mode == 'left' or hand_mode == 'both':
                            X_positive_left.append(hand['features'])
                            print(f"Captured positive sample #{len(X_positive_left)} for '{character}' with LEFT hand")
        elif key == ord('n') and detected_hands:  # N - capture negative sample
            for hand in detected_hands:
                X_negative.append(hand['features'])
                print(f"Captured negative sample #{len(X_negative)} with {'RIGHT' if hand['is_right_hand'] else 'LEFT'} hand")
        
        # Check if we have enough samples
        total_positives = len(X_positive_right) + len(X_positive_left)
        
        if hand_mode == 'right' and len(X_positive_right) >= min_samples:
            print(f"Collected minimum {min_samples} right hand samples for '{character}'")
        elif hand_mode == 'left' and len(X_positive_left) >= min_samples:
            print(f"Collected minimum {min_samples} left hand samples for '{character}'")
        elif hand_mode == 'both' and len(X_positive_right) >= min_samples and len(X_positive_left) >= min_samples:
            print(f"Collected minimum {min_samples} samples for both hands for '{character}'")
        
        if total_positives >= target_count and len(X_negative) >= min_samples:
            print("You have collected enough samples")
            print("You can continue collecting or press 'q' to finish")
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    
    return X_positive_right, X_positive_left, X_negative 

def train_combined_model(data_dir, output_file):
    """Train a single model on all collected data."""
    print(f"Training combined model using data from {data_dir}")
    
    # Get all character data files
    all_characters = [str(i) for i in range(10)] + [chr(i) for i in range(ord('a'), ord('z')+1)]
    
    # Find data files that exist
    X_train = []
    y_train = []
    hand_type_train = []  # 1 for right, 0 for left
    
    characters_with_data = []
    
    for char in all_characters:
        X_positive_right, X_positive_left, X_negative = load_feature_data(char, data_dir)
        
        if X_positive_right or X_positive_left:
            characters_with_data.append(char)
            
            # Add right hand positive samples
            if X_positive_right and len(X_positive_right) > 0:
                X_train.extend(X_positive_right)
                y_train.extend([char] * len(X_positive_right))
                
            # Add left hand positive samples
            if X_positive_left and len(X_positive_left) > 0:
                X_train.extend(X_positive_left)
                y_train.extend([char] * len(X_positive_left))
            
            # We don't need to add negative samples for each character
            # as positive samples for other characters serve as negatives
    
    if not characters_with_data:
        print("No training data found. Please collect data first.")
        return None
    
    print(f"Training with data from {len(characters_with_data)} characters: {', '.join(characters_with_data)}")
    print(f"Total samples: {len(X_train)}")
    print(f"Feature vector length: {len(X_train[0]) if X_train else 0}")
    
    if len(X_train) < 10:
        print("Not enough training data (need at least 10 samples)")
        return None
    
    # Split data into training and testing sets
    X_train_set, X_test, y_train_set, y_test = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)
    
    # Create and train the model
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=20,  # Increased for more complex feature space
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )
    
    model.fit(X_train_set, y_train_set)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
    
    # Print feature importance info
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\nTop 20 most important features:")
        for i in range(min(20, len(importances))):
            print(f"Feature {indices[i]}: {importances[indices[i]]:.4f}")
    
    # Save the model
    joblib.dump(model, output_file)
    print(f"Model saved to {output_file}")
    
    return model

def collect_data_for_character(character, args):
    """Collect data for a specific character and save it."""
    # Check if we have existing data
    X_pos_right_existing, X_pos_left_existing, X_neg_existing = load_feature_data(character, args.data_dir)
    
    # Collect new data
    X_positive_right, X_positive_left, X_negative = collect_data(
        character, 
        hand_mode=args.hand,
        min_samples=args.samples_per_char,
        camera_index=args.camera_index,
        use_preprocessing=args.use_preprocessing
    )
    
    if (X_positive_right or X_positive_left) and X_negative:
        # Combine with existing data if available
        if X_pos_right_existing:
            print(f"Combining with {len(X_pos_right_existing)} existing right hand samples for '{character}'")
            X_positive_right = X_pos_right_existing + (X_positive_right or [])
            
        if X_pos_left_existing:
            print(f"Combining with {len(X_pos_left_existing)} existing left hand samples for '{character}'")
            X_positive_left = X_pos_left_existing + (X_positive_left or [])
            
        if X_neg_existing:
            print(f"Combining with {len(X_neg_existing)} existing negative samples for '{character}'")
            X_negative = X_neg_existing + X_negative
        
        # Save the combined data
        save_feature_data(character, X_positive_right or [], X_positive_left or [], X_negative, args.data_dir)
        print(f"Successfully saved data for character '{character}'")
        return True
    
    return False

def collect_data_menu(args):
    """Display menu for data collection mode."""
    # Define all characters to train
    all_characters = [str(i) for i in range(10)] + [chr(i) for i in range(ord('a'), ord('z')+1)]
    
    print(f"ASL Data Collection for {len(all_characters)} characters: {', '.join(all_characters)}")
    print(f"Data will be saved to: {args.data_dir}")
    print(f"Hand mode: {args.hand}")
    print(f"NOTE: Using IMPROVED feature extraction - not compatible with regular web_trainer.py")
    
    while True:
        print("\nData Collection Options:")
        print("1. Collect data for a specific character")
        print("2. Collect data for all characters sequentially")
        print("3. Continue from a specific character")
        print("4. Train model with collected data")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            # Collect data for specific character
            char = input("Enter the character to collect data for (0-9, a-z): ").lower()
            if char in all_characters:
                collect_data_for_character(char, args)
            else:
                print(f"Invalid character: {char}. Must be 0-9 or a-z.")
        
        elif choice == '2':
            # Collect data for all characters
            for char in all_characters:
                print(f"\n=== Collecting data for '{char}' ===")
                success = collect_data_for_character(char, args)
                
                # Ask if user wants to continue
                if success and char != all_characters[-1]:
                    cont = input(f"Continue to next character? (y/n): ")
                    if cont.lower() != 'y':
                        break
        
        elif choice == '3':
            # Continue from specific character
            start_char = input("Enter the character to start from (0-9, a-z): ").lower()
            if start_char in all_characters:
                start_idx = all_characters.index(start_char)
                for char in all_characters[start_idx:]:
                    print(f"\n=== Collecting data for '{char}' ===")
                    success = collect_data_for_character(char, args)
                    
                    # Ask if user wants to continue
                    if success and char != all_characters[-1]:
                        cont = input(f"Continue to next character? (y/n): ")
                        if cont.lower() != 'y':
                            break
            else:
                print(f"Invalid character: {start_char}. Must be 0-9 or a-z.")
        
        elif choice == '4':
            # Train the model with collected data
            train_combined_model(args.data_dir, args.model_output)
        
        elif choice == '5':
            print("Exiting data collection mode.")
            break
        
        else:
            print("Invalid choice. Please enter 1-5.")

def main():
    args = parse_args()
    
    # Create data directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)
    
    if args.mode == 'collect':
        # If a specific character was provided, collect data for it
        if args.character:
            if args.character.lower() in [str(i) for i in range(10)] + [chr(i) for i in range(ord('a'), ord('z')+1)]:
                collect_data_for_character(args.character.lower(), args)
            else:
                print(f"Invalid character: {args.character}. Must be 0-9 or a-z.")
                return
        else:
            # Otherwise show the collection menu
            collect_data_menu(args)
    
    elif args.mode == 'train':
        # Train a model on all collected data
        train_combined_model(args.data_dir, args.model_output)

if __name__ == "__main__":
    main() 
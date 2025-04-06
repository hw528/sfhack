import cv2
import mediapipe as mp
import numpy as np
import os
import joblib
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='ASL Sign Position Guide for A and B')
    parser.add_argument('--model_path', type=str, default='./webcam_models/asl_combined_model.joblib',
                        help='Path to the trained ASL model')
    parser.add_argument('--camera_index', type=int, default=0,
                        help='Camera index to use (default: 0)')
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

def get_hand_position_feedback(letter, landmarks):
    """Analyze landmarks to provide correction feedback for specific signs."""
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
        finger_issues = []
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
    
    return feedback

def draw_semitransparent_rect(image, start_point, end_point, color, alpha=0.7):
    """Draw a semi-transparent rectangle on the image."""
    overlay = image.copy()
    cv2.rectangle(overlay, start_point, end_point, color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    return image

def setup_camera(camera_index):
    """Set up and return a camera capture object, trying multiple indices if needed."""
    print("Trying to initialize camera...")
    
    # Try the provided index first
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        ret, test_frame = cap.read()
        if ret and test_frame is not None and test_frame.size > 0:
            print(f"Successfully opened camera at index {camera_index}")
            return cap
        cap.release()
    
    # Try common indices if the provided index failed
    for idx in [0, 1, 2, 3, -1]:
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
        ]
    }
    
    return instructions.get(letter, ["No instructions available for this letter"])

def main():
    args = parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found")
        return
    
    # Load ASL model
    try:
        model = joblib.load(args.model_path)
        print(f"Successfully loaded model from {args.model_path}")
        
        # Check if model has classes attribute
        if hasattr(model, 'classes_'):
            print(f"Model can predict: {', '.join(model.classes_)}")
            if 'a' not in model.classes_ or 'b' not in model.classes_:
                print("Warning: Model doesn't contain both 'a' and 'b' classes")
        else:
            print("Warning: Model doesn't have classes attribute")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Initialize MediaPipe hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Initialize hands object
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Setup camera
    cap = setup_camera(args.camera_index)
    if cap is None:
        return
    
    # Variables for UI state
    target_letter = 'a'  # Start with letter 'a'
    target_hand = "right"  # Start with right hand
    use_preprocessing = True
    confidence_threshold = 0.6
    last_prediction_time = 0
    prediction_cooldown = 0.5  # Increased from 0.2 to 0.5 seconds to reduce flickering
    
    # Add variables for prediction smoothing
    prediction_history = []
    max_history_length = 5  # Number of frames to average over
    current_message = ""
    current_issues = []
    message_hold_frames = 0
    
    print(f"Starting ASL Position Guide for letters A and B")
    print(f"Default target letter: {target_letter.upper()}")
    print(f"Default target hand: {target_hand.upper()}")
    print("Press 'a' to switch to letter A")
    print("Press 'b' to switch to letter B")
    print("Press 'h' to toggle between left/right hand")
    print("Press 'p' to toggle preprocessing")
    print("Press '+'/'-' to adjust confidence threshold")
    print("Press 'ESC' or 'q' to exit")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Failed to grab frame")
            continue
        
        # Flip image for selfie view
        image = cv2.flip(image, 1)
        
        # Create a copy for processing if using preprocessing
        if use_preprocessing:
            processed_image = preprocess_image(image)
            # Show small preview of processed image
            small_preview = cv2.resize(processed_image, (160, 120))
            image[10:10+120, 10:10+160] = small_preview
            cv2.rectangle(image, (10, 10), (10+160, 10+120), (255, 255, 255), 1)
            cv2.putText(image, "Processed", (15, 25), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Use processed image for detection
            detection_image = processed_image
        else:
            detection_image = image
        
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(detection_image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        # Create semi-transparent overlay for instruction area
        instruction_height = 150
        draw_semitransparent_rect(image, (0, 0), (image.shape[1], instruction_height), (0, 0, 0))
        
        # Draw title and target letter
        title = f"Form the ASL letter '{target_letter.upper()}' sign"
        cv2.putText(image, title, (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw instructions for target letter
        instructions = get_asl_instructions(target_letter)
        for i, instruction in enumerate(instructions):
            cv2.putText(image, f"• {instruction}", (30, 60 + i*22), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Draw settings bar
        settings_y = image.shape[0] - 90
        draw_semitransparent_rect(image, (0, settings_y), (image.shape[1], image.shape[0]), (0, 0, 0))
        settings_text = f"Target: '{target_letter.upper()}' | Hand: {target_hand.upper()} | " \
                       f"Preprocessing: {'ON' if use_preprocessing else 'OFF'} | " \
                       f"Threshold: {confidence_threshold:.1f}"
        cv2.putText(image, settings_text, (20, settings_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Draw controls
        controls_text = "a/b: Change letter | h: Toggle hand | p: Toggle preprocessing | +/-: Threshold | ESC: Exit"
        cv2.putText(image, controls_text, (20, settings_y + 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Process hand detection
        current_prediction = None
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                # Get hand type (left or right)
                is_right_hand = handedness.classification[0].label == "Right"
                hand_type = "right" if is_right_hand else "left"
                
                # Only proceed if this is the target hand or we're accepting both hands
                if target_hand != "both" and hand_type != target_hand:
                    continue
                
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    image, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Extract features for prediction
                features = extract_features(hand_landmarks.landmark, is_right_hand)
                
                # Make prediction if it's time
                current_time = time.time()
                if current_time - last_prediction_time > prediction_cooldown:
                    last_prediction_time = current_time
                    
                    # Predict letter
                    if hasattr(model, 'predict') and hasattr(model, 'predict_proba'):
                        try:
                            letter_prediction = model.predict([features])[0]
                            letter_probas = model.predict_proba([features])[0]
                            
                            # Find probability for target letter
                            target_idx = -1
                            for i, letter in enumerate(model.classes_):
                                if letter == target_letter:
                                    target_idx = i
                                    break
                            
                            if target_idx >= 0:
                                target_probability = letter_probas[target_idx]
                                
                                # Get feedback on hand position
                                feedback = get_hand_position_feedback(target_letter, hand_landmarks.landmark)
                                
                                # Store prediction for smoothing
                                prediction_history.append({
                                    'letter': letter_prediction,
                                    'target_prob': target_probability,
                                    'feedback': feedback,
                                    'time': current_time
                                })
                                
                                # Trim old predictions
                                prediction_history = [p for p in prediction_history 
                                                     if current_time - p['time'] < 2.0]
                                
                                # Keep only most recent history
                                if len(prediction_history) > max_history_length:
                                    prediction_history = prediction_history[-max_history_length:]
                                
                                # Set current prediction
                                current_prediction = {
                                    'letter': letter_prediction,
                                    'target_prob': target_probability,
                                    'feedback': feedback
                                }
                        except Exception as e:
                            print(f"Error making prediction: {str(e)}")
        
        # Display feedback based on current and historical predictions
        feedback_y = 200
        
        if current_prediction:
            # Calculate average probability
            avg_probability = sum(p['target_prob'] for p in prediction_history) / len(prediction_history)
            
            # Count correct positions
            correct_positions = sum(1 for p in prediction_history if p['feedback']['correct'])
            
            # Determine if it's stable enough for a match
            is_stable_match = (len(prediction_history) >= 3 and 
                             avg_probability >= confidence_threshold and
                             correct_positions >= len(prediction_history) * 0.6)
            
            # Get most common issues
            all_issues = []
            for p in prediction_history:
                all_issues.extend(p['feedback']['finger_issues'])
            
            # Count issue occurrences
            issue_counts = {}
            for issue in all_issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
            
            # Get most common issues
            common_issues = []
            if issue_counts:
                # Sort by frequency
                sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
                # Take issues that appear in at least 1/3 of frames
                threshold = len(prediction_history) / 3
                common_issues = [issue for issue, count in sorted_issues if count >= threshold]
            
            # Update message if needed or hold current message
            if message_hold_frames > 0:
                message_hold_frames -= 1
            else:
                if is_stable_match:
                    current_message = f"Perfect '{target_letter.upper()}' sign! ({avg_probability:.2f})"
                    current_issues = []
                elif avg_probability >= confidence_threshold * 0.8:
                    # A high chance of the right letter, but position issues
                    current_message = f"Close to '{target_letter.upper()}' sign ({avg_probability:.2f})"
                    current_issues = common_issues
                else:
                    # Low confidence or wrong letter
                    if avg_probability < confidence_threshold * 0.5:
                        current_message = f"Low confidence for '{target_letter.upper()}' ({avg_probability:.2f})"
                    else:
                        # Determine most common predicted letter
                        letter_counts = {}
                        for p in prediction_history:
                            letter_counts[p['letter']] = letter_counts.get(p['letter'], 0) + 1
                        most_common_letter = max(letter_counts.items(), key=lambda x: x[1])[0]
                        
                        if most_common_letter != target_letter:
                            current_message = f"Detected '{most_common_letter.upper()}' instead of '{target_letter.upper()}'"
                        else:
                            current_message = f"Positioning needs work for '{target_letter.upper()}'"
                    
                    current_issues = common_issues
                
                # Hold the message for a few frames to prevent flickering
                message_hold_frames = 10
            
            # Draw main feedback message with background
            feedback_color = (0, 255, 0) if is_stable_match else (0, 165, 255)
            text_size = cv2.getTextSize(current_message, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            draw_semitransparent_rect(
                image, 
                (int(image.shape[1]/2 - text_size[0]/2 - 10), feedback_y - 30),
                (int(image.shape[1]/2 + text_size[0]/2 + 10), feedback_y + 10),
                (0, 0, 0)
            )
            cv2.putText(
                image, 
                current_message, 
                (int(image.shape[1]/2 - text_size[0]/2), feedback_y),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                feedback_color, 
                2
            )
            
            # Draw specific finger position issues
            if current_issues:
                for i, issue in enumerate(current_issues[:3]):  # Limit to top 3 issues
                    y_pos = feedback_y + 40 + i*30
                    text_size = cv2.getTextSize(issue, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
                    draw_semitransparent_rect(
                        image, 
                        (int(image.shape[1]/2 - text_size[0]/2 - 10), y_pos - 20),
                        (int(image.shape[1]/2 + text_size[0]/2 + 10), y_pos + 5),
                        (0, 0, 0)
                    )
                    cv2.putText(
                        image, 
                        issue, 
                        (int(image.shape[1]/2 - text_size[0]/2), y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (0, 165, 255), 
                        1
                    )
        elif len(prediction_history) > 0:
            # No current prediction, but we have history
            message = "Hand lost - Keep your hand in view"
            text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            draw_semitransparent_rect(
                image, 
                (int(image.shape[1]/2 - text_size[0]/2 - 10), feedback_y - 30),
                (int(image.shape[1]/2 + text_size[0]/2 + 10), feedback_y + 10),
                (0, 0, 0)
            )
            cv2.putText(
                image, 
                message, 
                (int(image.shape[1]/2 - text_size[0]/2), feedback_y),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (100, 100, 255), 
                2
            )
            
            # Clear prediction history if we haven't seen a hand for a while
            if time.time() - prediction_history[-1]['time'] > 2.0:
                prediction_history = []
                current_message = ""
                current_issues = []
        else:
            # No hand detected and no history
            message = "No hand detected"
            text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            draw_semitransparent_rect(
                image, 
                (int(image.shape[1]/2 - text_size[0]/2 - 10), feedback_y - 30),
                (int(image.shape[1]/2 + text_size[0]/2 + 10), feedback_y + 10),
                (0, 0, 0)
            )
            cv2.putText(
                image, 
                message, 
                (int(image.shape[1]/2 - text_size[0]/2), feedback_y),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (100, 100, 255), 
                2
            )
        
        # Display the image
        cv2.imshow('ASL Position Guide', image)
        
        # Process key presses
        key = cv2.waitKey(5) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or q key
            break
        elif key == ord('a'):  # Switch to letter 'a'
            target_letter = 'a'
            print(f"Switched to letter '{target_letter}'")
            # Clear prediction history when changing letter
            prediction_history = []
            current_message = ""
            current_issues = []
        elif key == ord('b'):  # Switch to letter 'b'
            target_letter = 'b'
            print(f"Switched to letter '{target_letter}'")
            # Clear prediction history when changing letter
            prediction_history = []
            current_message = ""
            current_issues = []
        elif key == ord('h'):  # Toggle hand
            if target_hand == "right":
                target_hand = "left"
            elif target_hand == "left":
                target_hand = "both"
            else:
                target_hand = "right"
            print(f"Switched to {target_hand} hand")
            # Clear prediction history when changing hand
            prediction_history = []
            current_message = ""
            current_issues = []
        elif key == ord('p'):  # Toggle preprocessing
            use_preprocessing = not use_preprocessing
            print(f"Preprocessing: {'ON' if use_preprocessing else 'OFF'}")
        elif key == ord('+') or key == ord('='):  # Increase threshold
            confidence_threshold = min(1.0, confidence_threshold + 0.05)
            print(f"Increased threshold to {confidence_threshold:.2f}")
        elif key == ord('-'):  # Decrease threshold
            confidence_threshold = max(0.1, confidence_threshold - 0.05)
            print(f"Decreased threshold to {confidence_threshold:.2f}")
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
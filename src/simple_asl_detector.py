import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import glob
from convert_features import extract_optimized_features
from datetime import datetime

class SimpleASLDetector:
    def __init__(self):
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Let user select and load model
        self.model, self.label_encoder = self.choose_and_load_model()
        
        # Initialize camera
        self.cap = self.setup_camera()
        
        # Settings
        self.show_landmarks = True
        self.target_hand = "right"
        
        # Get ASL letters in alphabetical order (excluding 'j' and 'z' which require motion)
        self.asl_letters = sorted([l for l in self.label_encoder.classes_ if l not in ['j', 'z']])
        if not self.asl_letters:
            self.asl_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 
                                'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']
            
        # Set initial target letter
        self.target_letter_idx = 0
        self.target_letter = self.asl_letters[self.target_letter_idx]
        
    def choose_and_load_model(self):
        """Let user choose which model to load from available directories"""
        # Find all model directories
        models_root = '/Users/wuhaodong/SFhack/models'
        model_dirs = []
        
        # Look for all potential model directories
        for pattern in ['new_run_*', 'optimized_run_*']:
            model_dirs.extend(glob.glob(os.path.join(models_root, pattern)))
        
        if not model_dirs:
            raise FileNotFoundError("No model directories found. Please train a model first.")
        
        # Sort by creation time (newest first)
        model_dirs.sort(key=os.path.getctime, reverse=True)
        
        # Display available model directories
        print("\nAvailable model directories:")
        for i, model_dir in enumerate(model_dirs):
            dir_name = os.path.basename(model_dir)
            created_time = os.path.getctime(model_dir)
            created_str = datetime.fromtimestamp(created_time).strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{i+1}] {dir_name} (created: {created_str})")
        
        # Ask user to select a directory
        while True:
            try:
                choice = int(input("\nSelect a model directory number (or 0 for most recent): "))
                if choice == 0:
                    selected_dir = model_dirs[0]
                    break
                elif 1 <= choice <= len(model_dirs):
                    selected_dir = model_dirs[choice-1]
                    break
                else:
                    print(f"Invalid choice. Please enter a number between 0 and {len(model_dirs)}")
            except ValueError:
                print("Please enter a valid number")
        
        print(f"\nSelected model directory: {os.path.basename(selected_dir)}")
        
        # Find model in the selected directory
        model_path = None
        
        # First try best_model.joblib
        best_model_path = os.path.join(selected_dir, 'best_model.joblib')
        if os.path.exists(best_model_path):
            model_path = best_model_path
        else:
            # List all available models in this directory
            model_files = glob.glob(os.path.join(selected_dir, '*_model_*.joblib'))
            if not model_files:
                raise FileNotFoundError(f"No model files found in {selected_dir}")
            
            # If there are multiple models, let the user choose
            if len(model_files) > 1:
                print("\nAvailable models in this directory:")
                for i, model_file in enumerate(model_files):
                    print(f"[{i+1}] {os.path.basename(model_file)}")
                
                while True:
                    try:
                        choice = int(input("\nSelect a model number: "))
                        if 1 <= choice <= len(model_files):
                            model_path = model_files[choice-1]
                            break
                        else:
                            print(f"Invalid choice. Please enter a number between 1 and {len(model_files)}")
                    except ValueError:
                        print("Please enter a valid number")
            else:
                # If there's only one model, use it
                model_path = model_files[0]
        
        # Find encoder in the selected directory
        encoder_path = os.path.join(selected_dir, 'label_encoder.joblib')
        if not os.path.exists(encoder_path):
            # Try to find any encoder
            encoder_files = glob.glob(os.path.join(selected_dir, 'label_encoder_*.joblib'))
            if not encoder_files:
                raise FileNotFoundError(f"No encoder files found in {selected_dir}")
            encoder_path = encoder_files[0]
        
        # Load model and encoder
        print(f"\nLoading model from: {os.path.basename(model_path)}")
        print(f"Loading encoder from: {os.path.basename(encoder_path)}")
        
        model = joblib.load(model_path)
        label_encoder = joblib.load(encoder_path)
        
        print(f"Model loaded. Can detect {len(label_encoder.classes_)} letters: {', '.join(label_encoder.classes_)}")
        return model, label_encoder
    
    def setup_camera(self):
        """Set up and return a camera capture object"""
        for camera_index in range(5):
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    print(f"Successfully opened camera {camera_index}")
                    return cap
                cap.release()
        
        raise RuntimeError("Failed to find a working camera")
    
    def get_hand_type(self, hand_landmarks, results):
        """Determine if the hand is left or right"""
        handedness = results.multi_handedness
        for idx, classification in enumerate(handedness):
            if idx == results.multi_hand_landmarks.index(hand_landmarks):
                return classification.classification[0].label == "Right"
        return True  # Default to right if can't determine
    
    def toggle_target_hand(self):
        """Toggle between left and right hand"""
        self.target_hand = "left" if self.target_hand == "right" else "right"
        return self.target_hand
    
    def next_target_letter(self):
        """Move to the next target letter"""
        self.target_letter_idx = (self.target_letter_idx + 1) % len(self.asl_letters)
        self.target_letter = self.asl_letters[self.target_letter_idx]
        return self.target_letter
    
    def prev_target_letter(self):
        """Move to the previous target letter"""
        self.target_letter_idx = (self.target_letter_idx - 1) % len(self.asl_letters)
        self.target_letter = self.asl_letters[self.target_letter_idx]
        return self.target_letter
    
    def draw_semitransparent_rect(self, image, start_point, end_point, color, alpha=0.5):
        """Draw a semi-transparent rectangle on the image"""
        overlay = image.copy()
        cv2.rectangle(overlay, start_point, end_point, color, -1)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        return image
    
    def detect_signs(self):
        """Main method to detect ASL signs and display confidence scores"""
        if self.cap is None or not self.cap.isOpened():
            print("No camera available")
            return
        
        if self.model is None or self.label_encoder is None:
            print("No model loaded")
            return
        
        print("\nControls:")
        print("  Press 'n' to go to next target letter")
        print("  Press 'p' to go to previous target letter")
        print("  Press 'h' to toggle hand (left/right)")
        print("  Press 'l' to toggle landmarks")
        print("  Press ESC to exit")
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Flip the frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)
            
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame and detect hands
            results = self.hands.process(rgb_frame)
            
            # Create background for text
            frame_h, frame_w, _ = frame.shape
            self.draw_semitransparent_rect(frame, (0, 0), (frame_w, 100), (0, 0, 0))
            
            # Draw target letter and controls info
            cv2.putText(frame, f"Target letter: {self.target_letter.upper()}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, f"Hand: {self.target_hand}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Check if hands are detected
            confidence_score = 0
            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Determine if hand is left or right
                    is_right_hand = self.get_hand_type(hand_landmarks, results)
                    hand_type = "Right" if is_right_hand else "Left"
                    
                    # Check if this hand matches our target hand type
                    if (self.target_hand == "right" and not is_right_hand) or \
                       (self.target_hand == "left" and is_right_hand):
                        continue  # Skip this hand if it doesn't match target
                    
                    # Draw hand landmarks if enabled
                    if self.show_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Convert landmarks to dictionary format
                    landmarks_dict = []
                    for landmark in hand_landmarks.landmark:
                        landmarks_dict.append({
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z
                        })
                    
                    # Extract optimized features
                    features = extract_optimized_features(landmarks_dict)
                    
                    # Convert features to numpy array
                    feature_values = np.array(list(features.values()))
                    
                    # Make prediction
                    try:
                        # Get prediction probabilities
                        probabilities = self.model.predict_proba([feature_values])[0]
                        
                        # Get confidence score for the target letter
                        target_letter_idx = np.where(self.label_encoder.classes_ == self.target_letter)[0][0]
                        confidence_score = probabilities[target_letter_idx]
                        
                        # Get the predicted letter
                        prediction_idx = np.argmax(probabilities)
                        predicted_letter = self.label_encoder.classes_[prediction_idx]
                        prediction_confidence = probabilities[prediction_idx]
                        
                        # Draw predicted letter
                        cv2.putText(frame, f"Predicted: {predicted_letter.upper()} ({prediction_confidence:.2f})", 
                                   (frame_w - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    except Exception as e:
                        print(f"Error making prediction: {e}")
            
            # Draw confidence meter for target letter
            meter_width = int(frame_w * 0.7)
            meter_height = 50
            meter_x = int((frame_w - meter_width) / 2)
            meter_y = frame_h - meter_height - 50
            
            # Draw meter background
            cv2.rectangle(frame, (meter_x, meter_y), (meter_x + meter_width, meter_y + meter_height), (50, 50, 50), -1)
            
            # Draw confidence level
            filled_width = int(meter_width * confidence_score)
            
            # Color changes from red (low) to yellow (medium) to green (high)
            if confidence_score < 0.33:
                color = (0, 0, 255)  # Red
            elif confidence_score < 0.67:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 255, 0)  # Green
                
            cv2.rectangle(frame, (meter_x, meter_y), (meter_x + filled_width, meter_y + meter_height), color, -1)
            
            # Draw confidence text
            cv2.putText(frame, f"Confidence for '{self.target_letter.upper()}': {confidence_score:.2f}", 
                       (meter_x, meter_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw threshold markers
            for threshold in [0.25, 0.5, 0.75]:
                threshold_x = meter_x + int(meter_width * threshold)
                cv2.line(frame, (threshold_x, meter_y), (threshold_x, meter_y + meter_height), (200, 200, 200), 2)
                cv2.putText(frame, f"{threshold:.1f}", (threshold_x - 10, meter_y + meter_height + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Display the resulting frame
            cv2.imshow('ASL Detector - Target Letter', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('n'):  # Next letter
                self.next_target_letter()
                print(f"Target letter: {self.target_letter.upper()}")
            elif key == ord('p'):  # Previous letter
                self.prev_target_letter()
                print(f"Target letter: {self.target_letter.upper()}")
            elif key == ord('h'):  # Change hand
                self.toggle_target_hand()
                print(f"Target hand: {self.target_hand}")
            elif key == ord('l'):  # Toggle landmarks
                self.show_landmarks = not self.show_landmarks
                print(f"Show landmarks: {self.show_landmarks}")
        
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = SimpleASLDetector()
    detector.detect_signs() 
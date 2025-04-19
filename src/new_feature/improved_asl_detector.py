import cv2
import mediapipe as mp
import numpy as np
import joblib
from improved_feature_extraction import extract_optimized_features, get_feature_names
import time
import os

class ImprovedASLDetector:
    def __init__(self):
        """Initialize the improved ASL detector with optimized feature extraction."""
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Load pretrained models - these will need to be trained with the new features
        self.model = None
        self.label_encoder = None
        
        # Define confidence thresholds
        self.confidence_threshold = 0.5
        
        # Initialize camera
        self.cap = self.setup_camera()
        
        # Target settings
        self.target_hand = "right"
        self.target_letter = "a"
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        
        # Visualization settings
        self.show_landmarks = True
        self.show_feature_metrics = False
        
        print("Improved ASL Detector initialized")
        print(f"Using {len(get_feature_names())} optimized features for detection")
    
    def load_model(self, model_path, encoder_path):
        """Load a pretrained model and label encoder."""
        try:
            self.model = joblib.load(model_path)
            self.label_encoder = joblib.load(encoder_path)
            print(f"Model loaded from {model_path}")
            print(f"Encoder loaded from {encoder_path}")
            print(f"Model can detect letters: {', '.join(self.label_encoder.classes_)}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_hand_type(self, hand_landmarks, results):
        """Determine if the hand is left or right."""
        handedness = results.multi_handedness
        for idx, classification in enumerate(handedness):
            if idx == results.multi_hand_landmarks.index(hand_landmarks):
                return classification.classification[0].label == "Right"
        return True  # Default to right if can't determine
    
    def setup_camera(self):
        """Set up and return a camera capture object."""
        print("Setting up camera...")
        # Try different camera indices
        for camera_index in range(5):
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None and test_frame.size > 0:
                    print(f"Successfully opened camera {camera_index}")
                    time.sleep(1)  # Give the camera time to initialize
                    return cap
                cap.release()
        
        print("Failed to find a working camera")
        return None
    
    def toggle_target_hand(self):
        """Toggle between left and right hand."""
        self.target_hand = "left" if self.target_hand == "right" else "right"
        return self.target_hand
    
    def toggle_features_display(self):
        """Toggle the display of feature metrics."""
        self.show_feature_metrics = not self.show_feature_metrics
        return self.show_feature_metrics
    
    def detect_sign(self):
        """Main method to detect ASL signs from the webcam."""
        if self.cap is None:
            print("No camera available")
            return
        
        # Check if model is loaded
        if self.model is None or self.label_encoder is None:
            print("No model loaded. Please load a model first.")
            return
        
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
            
            # Calculate FPS
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 1:
                self.fps = self.frame_count / elapsed_time
                self.frame_count = 0
                self.start_time = time.time()
            
            # Draw FPS on frame
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw target hand
            cv2.putText(frame, f"Hand: {self.target_hand}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Check if hands are detected
            if results.multi_hand_landmarks:
                # Process each detected hand
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
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style())
                    
                    # Extract optimized features
                    landmarks = hand_landmarks.landmark
                    features = extract_optimized_features(landmarks, is_right_hand)
                    
                    # Make prediction
                    try:
                        probabilities = self.model.predict_proba([features])[0]
                        prediction_idx = np.argmax(probabilities)
                        confidence = probabilities[prediction_idx]
                        
                        # Get the predicted letter
                        predicted_letter = self.label_encoder.inverse_transform([prediction_idx])[0]
                        
                        # Draw prediction and confidence on frame
                        cv2.putText(frame, f"Detected: {predicted_letter.upper()}", (10, 110), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 150), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        # Show additional information about features if enabled
                        if self.show_feature_metrics:
                            # Display some key features
                            feature_names = get_feature_names()
                            for i, feature_idx in enumerate([10, 25, 35, 45, 55]):  # Show a sample of features
                                if feature_idx < len(features):
                                    cv2.putText(frame, f"{feature_names[feature_idx]}: {features[feature_idx]:.2f}", 
                                                (10, 190 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
                    except Exception as e:
                        print(f"Error making prediction: {e}")
            
            # Display the resulting frame
            cv2.imshow('Improved ASL Detector', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('h'):  # Change hand
                self.toggle_target_hand()
                print(f"Target hand: {self.target_hand}")
            elif key == ord('f'):  # Toggle features display
                self.toggle_features_display()
                print(f"Show features: {self.show_feature_metrics}")
            elif key == ord('l'):  # Toggle landmarks display
                self.show_landmarks = not self.show_landmarks
                print(f"Show landmarks: {self.show_landmarks}")
        
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()
    
    def train_new_model(self, data_path, output_dir=None):
        """
        Train a new model using the optimized features.
        
        Args:
            data_path: Path to dataset of preprocessed landmarks
            output_dir: Directory to save model files (default: current directory)
        """
        try:
            # This is a placeholder for model training logic
            # In a real implementation, this would:
            # 1. Load training data
            # 2. Extract optimized features for each sample
            # 3. Train models (RandomForest, XGBoost, etc.)
            # 4. Evaluate and save the best model
            
            print("Model training feature not implemented yet.")
            print("To train a model with these features:")
            print("1. Extract landmarks from your ASL dataset")
            print("2. Process each landmark set with extract_optimized_features()")
            print("3. Use scikit-learn to train a classifier on the features")
            print("4. Save the model using joblib.dump()")
            
            return False
        except Exception as e:
            print(f"Error training model: {e}")
            return False


if __name__ == "__main__":
    # Create and start the ASL detector
    detector = ImprovedASLDetector()
    
    # Define model and encoder paths - these will need to be trained first
    model_dir = '/Users/wuhaodong/SFhack/models'
    
    # Check if model exists (this is a placeholder - real path would be used)
    sample_model_path = os.path.join(model_dir, 'improved_model.joblib')
    sample_encoder_path = os.path.join(model_dir, 'improved_encoder.joblib')
    
    if os.path.exists(sample_model_path) and os.path.exists(sample_encoder_path):
        # Load model if available
        detector.load_model(sample_model_path, sample_encoder_path)
        detector.detect_sign()
    else:
        print("No trained model available with improved features.")
        print("You need to train a model first using the improved feature extraction.")
        print("\nFeature names for reference:")
        for i, name in enumerate(get_feature_names()):
            print(f"{i+1}. {name}")
        print(f"\nTotal: {len(get_feature_names())} optimized features") 
import cv2
import mediapipe as mp
import numpy as np
import time
import os
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class ASLRandomForestDetector:
    def __init__(self):
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Higher confidence for detection
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Setup for Random Forest model
        self.model = None
        self.model_file = "asl_random_forest_model.joblib"
        
        # Training data
        self.X = []  # Features
        self.y = []  # Labels (1 for A sign, 0 for not A sign)
        
        # Paths for saving collected data
        self.data_dir = "asl_training_data"
        self.X_file = os.path.join(self.data_dir, "X_data.npy")
        self.y_file = os.path.join(self.data_dir, "y_data.npy")
        
        # Ensure data directory exists
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # Hand preference (left or right)
        self.target_hand = "right"
        
        # Load existing model if available
        self.load_model()
    
    def extract_features(self, landmarks):
        """Extract features from hand landmarks for the model."""
        # Flatten coordinates into a 1D array
        features = []
        
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
    
    def collect_data(self):
        """Collect training data from webcam."""
        print("Starting data collection mode...")
        print(f"Current target hand: {self.target_hand.upper()}")
        print("Press 'a' to capture an 'A' sign sample")
        print("Press 'n' to capture a 'NOT A' sign sample")
        print("Press 'h' to toggle between left/right hand")
        print("Press 'q' or ESC to quit")
        
        # Try to load existing data if available
        if os.path.exists(self.X_file) and os.path.exists(self.y_file):
            try:
                self.X = list(np.load(self.X_file, allow_pickle=True))
                self.y = list(np.load(self.y_file, allow_pickle=True))
                print(f"Loaded {len(self.X)} existing samples")
            except Exception as e:
                print(f"Error loading existing data: {e}")
                self.X = []
                self.y = []
        
        # Setup camera
        cap = self.setup_camera()
        if cap is None:
            print("Failed to setup camera. Exiting data collection.")
            return
        
        a_count = 0
        not_a_count = 0
        collecting = True
        
        # Count existing samples
        a_samples = sum(1 for label in self.y if label == 1)
        not_a_samples = sum(1 for label in self.y if label == 0)
        print(f"Current samples: {a_samples} 'A' signs, {not_a_samples} 'NOT A' signs")
        
        while collecting and cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Failed to grab frame")
                continue
            
            # Flip image for selfie view
            image = cv2.flip(image, 1)
            
            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)
            
            # Draw hand information text
            cv2.putText(image, f"Target Hand: {self.target_hand.upper()}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display counts
            cv2.putText(image, f"'A' samples: {a_samples + a_count}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"'NOT A' samples: {not_a_samples + not_a_count}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
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
            
            # Add instructions to screen
            cv2.putText(image, "Press 'a': Capture 'A' sign", (10, image.shape[0] - 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(image, "Press 'n': Capture 'NOT A' sign", (10, image.shape[0] - 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(image, "Press 'h': Toggle hand | 's': Save | 'q': Quit", 
                        (10, image.shape[0] - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display the image
            cv2.imshow('ASL Data Collection', image)
            
            # Process key presses
            key = cv2.waitKey(5) & 0xFF
            
            if key == 27 or key == ord('q'):  # ESC or q key
                collecting = False
            elif key == ord('h'):  # Toggle hand
                self.toggle_target_hand()
                print(f"Switched to {self.target_hand} hand")
            elif key == ord('s'):  # Save data
                self._save_data()
                print("Data saved!")
            elif (key == ord('a') or key == ord('n')) and results.multi_hand_landmarks:
                # Find the target hand among detected hands
                target_hand_found = False
                for hand_landmarks in results.multi_hand_landmarks:
                    is_right_hand = self.get_hand_type(hand_landmarks, results)
                    hand_type = "right" if is_right_hand else "left"
                    
                    # Only use the specified hand
                    if hand_type == self.target_hand:
                        target_hand_found = True
                        # Extract features
                        features = self.extract_features(hand_landmarks.landmark)
                        
                        # Add to training data
                        if key == ord('a'):  # 'A' sign
                            self.X.append(features)
                            self.y.append(1)  # Label 1 for 'A'
                            a_count += 1
                            print(f"Captured 'A' sign sample #{a_samples + a_count}")
                        else:  # NOT 'A' sign
                            self.X.append(features)
                            self.y.append(0)  # Label 0 for 'NOT A'
                            not_a_count += 1
                            print(f"Captured 'NOT A' sign sample #{not_a_samples + not_a_count}")
                        
                        # Automatic save after every 10 samples
                        if (a_count + not_a_count) % 10 == 0:
                            self._save_data()
                            print("Data autosaved")
                        
                        break  # Only capture one hand per frame
                
                if not target_hand_found:
                    print(f"No {self.target_hand} hand detected. Please show your {self.target_hand} hand.")
        
        # Save before exiting
        if a_count > 0 or not_a_count > 0:
            self._save_data()
            print(f"Data collection finished. Total samples: {len(self.X)}")
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
    
    def _save_data(self):
        """Save collected data to files."""
        np.save(self.X_file, np.array(self.X))
        np.save(self.y_file, np.array(self.y))
    
    def train_model(self):
        """Train the Random Forest model on collected data."""
        # Check if we have enough data
        if len(self.X) < 10 or len(self.y) < 10:
            print("Not enough training data. Please collect more samples.")
            return False
        
        if len(self.X) != len(self.y):
            print("Data mismatch. Features and labels have different counts.")
            return False
        
        print(f"Training model on {len(self.X)} samples...")
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42)
        
        # Create and train the model
        self.model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.2f}")
        print(classification_report(y_test, y_pred))
        
        # Save the model
        joblib.dump(self.model, self.model_file)
        print(f"Model saved to {self.model_file}")
        
        return True
    
    def load_model(self):
        """Load a pre-trained model if available."""
        if os.path.exists(self.model_file):
            try:
                self.model = joblib.load(self.model_file)
                print(f"Loaded model from {self.model_file}")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
        
        return False
    
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
    
    def detect_sign(self):
        """Run real-time detection using the trained model."""
        if self.model is None:
            print("No model loaded. Please train a model first.")
            return
        
        print("Starting sign detection mode...")
        print(f"Current target hand: {self.target_hand.upper()}")
        print("Press 'h' to toggle between left/right hand")
        print("Press 'q' or ESC to quit")
        
        # Setup camera
        cap = self.setup_camera()
        if cap is None:
            print("Failed to setup camera. Exiting detection.")
            return
        
        # Guidance text for correct 'A' sign
        guide_title = "How to form the 'A' sign correctly:"
        guide_instructions = [
            "1. Make a fist with all fingers closed",
            "2. Place your thumb against the side of your index finger",
            "3. Keep thumb pointing upward, not across palm"
        ]
        error_text = ""
        
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
            guide_overlay = image.copy()
            cv2.rectangle(guide_overlay, (0, 0), (image.shape[1], 150), (0, 0, 0), -1)
            image = cv2.addWeighted(guide_overlay, 0.3, image, 0.7, 0)
            
            # Draw guide title
            cv2.putText(image, guide_title, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw guide instructions
            for i, line in enumerate(guide_instructions):
                cv2.putText(image, line, (20, 50 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Draw hand information text
            cv2.putText(image, f"Target Hand: {self.target_hand.upper()}", (image.shape[1] - 250, 25), 
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
                    if hand_type == self.target_hand:
                        found_target_hand = True
                        
                        # Extract features
                        features = self.extract_features(hand_landmarks.landmark)
                        
                        # Check specific hand positions to provide better feedback
                        landmarks = hand_landmarks.landmark
                        thumb_tip = landmarks[4]
                        index_tip = landmarks[8]
                        middle_tip = landmarks[12]
                        ring_tip = landmarks[16]
                        pinky_tip = landmarks[20]
                        index_base = landmarks[5]
                        
                        # Check finger positions for more specific feedback
                        fingers_closed = (
                            index_tip.y > index_base.y and
                            middle_tip.y > landmarks[9].y and
                            ring_tip.y > landmarks[13].y and
                            pinky_tip.y > landmarks[17].y
                        )
                        
                        # Calculate distance for thumb position
                        thumb_distance = np.sqrt(
                            (thumb_tip.x - index_base.x)**2 + 
                            (thumb_tip.y - index_base.y)**2 +
                            (thumb_tip.z - index_base.z)**2
                        )
                        
                        # Analyze positioning
                        if not fingers_closed:
                            error_text = "Error: Close all your fingers into a fist"
                        elif thumb_distance > 0.08:
                            error_text = "Error: Thumb should touch the side of your index finger"
                        elif thumb_tip.y > landmarks[3].y:  # thumb_ip
                            error_text = "Error: Thumb should point upward, not downward"
                        else:
                            error_text = "Good form! Keep it up"
                        
                        # Make prediction
                        prediction = self.model.predict([features])[0]
                        confidence = self.model.predict_proba([features])[0]
                        
                        # Display result
                        if prediction == 1 and confidence[1] >= 0.7:  # Only consider it an 'A' if 70% confident
                            prob = confidence[1]
                            result_text = f"A sign detected! ({prob:.2f})"
                            color = (0, 255, 0)
                        else:
                            prob = confidence[0] if prediction == 0 else confidence[1]
                            result_text = f"Not an A sign ({prob:.2f})"
                            color = (0, 0, 255)
                        
                        cv2.putText(image, result_text, (10, 175), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if not found_target_hand:
                # Target hand not detected
                cv2.putText(image, f"No {self.target_hand} hand detected", (10, 175), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Display error/guidance text
            cv2.putText(image, error_text, (10, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            # Add instructions to screen
            cv2.putText(image, "Press 'h': Toggle hand | 'q' or ESC: Quit", 
                        (10, image.shape[0] - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display the image
            cv2.imshow('ASL Sign Detector (Random Forest)', image)
            
            # Process key presses
            key = cv2.waitKey(5) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or q key
                break
            elif key == ord('h'):  # Toggle hand
                self.toggle_target_hand()
                print(f"Switched to {self.target_hand} hand")
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()


def main():
    detector = ASLRandomForestDetector()
    
    print("\nASL 'A' Sign Random Forest Detector")
    print("===================================")
    print("1. Collect training data")
    print("2. Train model")
    print("3. Detect signs (requires trained model)")
    print("4. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            detector.collect_data()
        elif choice == '2':
            detector.train_model()
        elif choice == '3':
            detector.detect_sign()
        elif choice == '4':
            print("Exiting program")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main() 
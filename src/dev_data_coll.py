import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time
import signal
import sys
from datetime import datetime

class ASLRawDataCollector:
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
        
        # Initialize camera
        self.cap = self.setup_camera()
        
        # Settings
        self.show_landmarks = True
        self.target_hand = "right"
        self.sample_type = "positive"  # positive or negative
        
        # Define ASL letters (excluding J and Z which require motion)
        self.asl_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 
                           'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']
        
        # Set initial target letter
        self.target_letter_idx = 0
        self.target_letter = self.asl_letters[self.target_letter_idx]
        
        # Create output directory
        self.output_dir = os.path.join('/Users/wuhaodong/SFhack', 'raw_asl_data')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Data collection variables
        self.collecting = False
        self.collection_start_time = 0
        self.collected_frames = 0
        self.collection_mode = "none"  # none, single, continuous
        
        # Dataset
        self.all_data = []
        
        # Register signal handler for clean exit
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def signal_handler(self, sig, frame):
        """Handle CTRL+C and other exit signals"""
        print("\nExiting and saving data...")
        self.save_data()
        self.cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)
        
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
                return classification.classification[0].label.lower()
        return "right"  # Default to right if can't determine
    
    def toggle_target_hand(self):
        """Toggle between left and right hand"""
        self.target_hand = "left" if self.target_hand == "right" else "right"
        return self.target_hand
    
    def toggle_sample_type(self):
        """Toggle between positive and negative samples"""
        self.sample_type = "negative" if self.sample_type == "positive" else "positive"
        return self.sample_type
    
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
    
    def landmarks_to_dataframe_row(self, landmarks, hand_type, sample_label, sample_type):
        """Convert landmarks to a dataframe row with raw coordinates"""
        features = {}
        
        # Store basic info
        features['hand_type'] = hand_type
        features['label'] = sample_label
        features['timestamp'] = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        features['sample_type'] = sample_type
        
        # Store raw landmark data (all relative to wrist - landmark 0)
        wrist_x = landmarks[0]['x']
        wrist_y = landmarks[0]['y']
        wrist_z = landmarks[0]['z']
        
        # Store all 21 landmarks raw x, y, z coordinates (relative to wrist)
        for i in range(21):
            features[f'landmark_{i}_x'] = landmarks[i]['x'] - wrist_x
            features[f'landmark_{i}_y'] = landmarks[i]['y'] - wrist_y
            features[f'landmark_{i}_z'] = landmarks[i]['z'] - wrist_z
        
        return features

    def save_data(self):
        """Save collected data to CSV file"""
        if not self.all_data:
            print("No data to save")
            return
        
        # Create DataFrame
        df = pd.DataFrame(self.all_data)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'raw_asl_data_{timestamp}.csv'
        filepath = os.path.join(self.output_dir, filename)
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        print(f"Saved {len(df)} samples to {filepath}")
        
        # Stats about the dataset
        print("\nData distribution:")
        
        # Count by letter
        letter_counts = df['label'].value_counts()
        print("\nBy letter:")
        for letter, count in letter_counts.items():
            print(f"  {letter}: {count} samples")
        
        # Count by sample type
        type_counts = df['sample_type'].value_counts()
        print("\nBy type:")
        for type_name, count in type_counts.items():
            print(f"  {type_name}: {count} samples")
        
        # Count by hand type
        hand_counts = df['hand_type'].value_counts()
        print("\nBy hand:")
        for hand, count in hand_counts.items():
            print(f"  {hand}: {count} samples")
        
        # Reset data collection
        self.all_data = []
        
    def collect_data(self):
        """Main method to collect raw hand landmark data"""
        if self.cap is None or not self.cap.isOpened():
            print("No camera available")
            return
        
        print("\nControls:")
        print("  Press 'n' to go to next target letter")
        print("  Press 'p' to go to previous target letter")
        print("  Press 'h' to toggle hand (left/right)")
        print("  Press 't' to toggle sample type (positive/negative)")
        print("  Press 'l' to toggle landmarks display")
        print("  Press 'c' to collect a single frame of data")
        print("  Press 'space' to start/stop continuous data collection")
        print("  Press 's' to save all collected data to CSV")
        print("  Press 'r' to reset collected data")
        print("  Press ESC to exit (will auto-save)")
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Flip the frame horizontally for a selfie-view display
            frame = cv2.flip(frame, 1)
            
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame and detect hands
            results = self.hands.process(rgb_frame)
            
            # Create background for text
            frame_h, frame_w, _ = frame.shape
            self.draw_semitransparent_rect(frame, (0, 0), (frame_w, 140), (0, 0, 0))
            
            # Sample type color
            sample_type_color = (0, 255, 0) if self.sample_type == "positive" else (0, 0, 255)
            
            # Draw target letter and controls info
            cv2.putText(frame, f"Target letter: {self.target_letter.upper()}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, f"Target hand: {self.target_hand}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Sample type: {self.sample_type}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, sample_type_color, 2)
            cv2.putText(frame, f"Collected frames: {self.collected_frames}", (frame_w - 250, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Collection mode indicator
            if self.collection_mode == "continuous":
                # Show recording indicator (red circle)
                cv2.circle(frame, (frame_w - 30, 70), 15, (0, 0, 255), -1)
                # Show recording time
                elapsed_time = time.time() - self.collection_start_time
                cv2.putText(frame, f"Recording: {elapsed_time:.1f}s", (frame_w - 200, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Check if hands are detected
            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Determine if hand is left or right
                    hand_type = self.get_hand_type(hand_landmarks, results)
                    
                    # Check if this hand matches our target hand type
                    if self.target_hand != hand_type:
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
                    
                    # Collect data if in collection mode
                    if self.collection_mode == "single":
                        # Add to dataset
                        data_row = self.landmarks_to_dataframe_row(
                            landmarks_dict, hand_type, self.target_letter, self.sample_type)
                        self.all_data.append(data_row)
                        self.collected_frames += 1
                        print(f"Collected {self.sample_type} frame for {self.target_letter.upper()} ({hand_type} hand)")
                        
                        # Reset collection mode
                        self.collection_mode = "none"
                    
                    elif self.collection_mode == "continuous":
                        # Collect at a reasonable rate (every 5 frames)
                        if self.collected_frames % 5 == 0:
                            data_row = self.landmarks_to_dataframe_row(
                                landmarks_dict, hand_type, self.target_letter, self.sample_type)
                            self.all_data.append(data_row)
                            print(f"Collected {self.sample_type} frame for {self.target_letter.upper()} ({hand_type} hand)")
                        
                        self.collected_frames += 1
            
            # Display the resulting frame
            cv2.imshow('ASL Raw Data Collector', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC - exit and save
                print("Exiting and saving data...")
                self.save_data()
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
            elif key == ord('t'):  # Toggle sample type
                self.toggle_sample_type()
                print(f"Sample type: {self.sample_type}")
            elif key == ord('l'):  # Toggle landmarks
                self.show_landmarks = not self.show_landmarks
                print(f"Show landmarks: {self.show_landmarks}")
            elif key == ord('c'):  # Collect single frame
                self.collection_mode = "single"
                print(f"Collecting single {self.sample_type} frame for {self.target_letter.upper()}")
            elif key == 32:  # Space - Start/stop continuous collection
                if self.collection_mode != "continuous":
                    self.collection_mode = "continuous"
                    self.collection_start_time = time.time()
                    print(f"Started continuous {self.sample_type} collection for {self.target_letter.upper()}")
                else:
                    self.collection_mode = "none"
                    elapsed = time.time() - self.collection_start_time
                    print(f"Stopped continuous collection. Collected for {elapsed:.1f} seconds")
            elif key == ord('s'):  # Save data
                self.save_data()
            elif key == ord('r'):  # Reset data
                self.all_data = []
                self.collected_frames = 0
                print("Reset collected data")
        
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    collector = ASLRawDataCollector()
    collector.collect_data()
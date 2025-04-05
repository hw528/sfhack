import cv2
import mediapipe as mp
import numpy as np
import os
import joblib
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import glob
from tqdm import tqdm  # For progress bars

def parse_args():
    parser = argparse.ArgumentParser(description='Train ASL model from image dataset')
    parser.add_argument('--dataset_dir', type=str, required=True, 
                        help='Directory containing the image dataset. Should have subdirectories for each letter.')
    parser.add_argument('--output_model', type=str, default='asl_random_forest_model.joblib',
                        help='Path to save the trained model')
    parser.add_argument('--target_letter', type=str, default='A',
                        help='Target letter to train (default: A)')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Size to resize images to (default: 224)')
    return parser.parse_args()

def extract_features(landmarks):
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

def process_dataset(dataset_dir, target_letter, image_size=224):
    """Process all images in the dataset directory."""
    # Initialize MediaPipe hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,  # Use static mode for images
        max_num_hands=1,         # Only need one hand per image
        min_detection_confidence=0.5
    )
    
    features = []
    labels = []
    
    print(f"Processing images from {dataset_dir}")
    print(f"Target letter: {target_letter}")
    
    # Get all letter directories
    letter_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    
    total_processed = 0
    total_detected = 0
    
    for letter in letter_dirs:
        letter_path = os.path.join(dataset_dir, letter)
        # Set the label: 1 for target letter, 0 for others
        is_target = (letter.upper() == target_letter.upper())
        label = 1 if is_target else 0
        
        # Get all images for this letter
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.gif']:
            image_files.extend(glob.glob(os.path.join(letter_path, ext)))
        
        print(f"Processing {len(image_files)} images for letter {letter} (Target: {is_target})")
        
        for image_file in tqdm(image_files):
            # Read image
            image = cv2.imread(image_file)
            if image is None:
                continue
                
            # Resize image
            image = cv2.resize(image, (image_size, image_size))
                
            # Convert to RGB (MediaPipe requires RGB)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
            # Process with MediaPipe
            results = hands.process(image_rgb)
                
            # Check if hand landmarks detected
            if results.multi_hand_landmarks:
                total_detected += 1
                hand_landmarks = results.multi_hand_landmarks[0]  # Get the first hand
                
                # Extract features
                feature_vector = extract_features(hand_landmarks.landmark)
                
                # Add to dataset
                features.append(feature_vector)
                labels.append(label)
            
            total_processed += 1
    
    print(f"Processed {total_processed} images, detected hands in {total_detected} images")
    print(f"Final dataset: {len(features)} samples, {sum(labels)} positive, {len(features) - sum(labels)} negative")
    
    return features, labels

def train_model(features, labels, model_file):
    """Train a Random Forest model on the dataset."""
    print("Training model...")
    
    if len(features) < 10:
        print("Not enough training data (need at least 10 samples)")
        return False
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42)
    
    # Create and train the model
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
    
    # Save the model
    joblib.dump(model, model_file)
    print(f"Model saved to {model_file}")
    
    return True

def main():
    args = parse_args()
    
    features, labels = process_dataset(
        args.dataset_dir, 
        args.target_letter, 
        args.image_size
    )
    
    if features:
        train_model(features, labels, args.output_model)
    else:
        print("No features extracted from dataset. Check if the images contain visible hands.")

if __name__ == "__main__":
    main() 
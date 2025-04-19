import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

def calculate_angle(p1, p2, p3):
    """Calculate angle between three points in 3D space."""
    v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y'], p1['z'] - p2['z']])
    v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y'], p3['z'] - p2['z']])
    
    # Normalize vectors
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    # Handle zero vectors
    if v1_norm < 1e-6 or v2_norm < 1e-6:
        return 0.0
    
    v1 = v1 / v1_norm
    v2 = v2 / v2_norm
    
    # Calculate angle using dot product
    dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle = np.arccos(dot_product)
    
    return angle

def normalized_distance(p1, p2, hand_scale):
    """Calculate normalized Euclidean distance between two points."""
    dist = np.sqrt(
        (p1['x'] - p2['x'])**2 + 
        (p1['y'] - p2['y'])**2 + 
        (p1['z'] - p2['z'])**2
    )
    return dist / hand_scale

def normalized_distance_to_point(p, x, y, z, hand_scale):
    """Calculate normalized Euclidean distance from a landmark to a point."""
    dist = np.sqrt(
        (p['x'] - x)**2 + 
        (p['y'] - y)**2 + 
        (p['z'] - z)**2
    )
    return dist / hand_scale

def calculate_finger_extension(landmarks, indices):
    """
    Calculate how extended a finger is based on landmark indices.
    Returns a ratio between 0 (completely curled) and 1 (fully extended).
    """
    # Calculate direct distance from base to tip
    direct_distance = np.sqrt(
        (landmarks[indices[0]]['x'] - landmarks[indices[3]]['x'])**2 + 
        (landmarks[indices[0]]['y'] - landmarks[indices[3]]['y'])**2 + 
        (landmarks[indices[0]]['z'] - landmarks[indices[3]]['z'])**2
    )
    
    # Calculate sum of all segments
    segment_sum = 0
    for i in range(len(indices) - 1):
        segment_sum += np.sqrt(
            (landmarks[indices[i]]['x'] - landmarks[indices[i+1]]['x'])**2 + 
            (landmarks[indices[i]]['y'] - landmarks[indices[i+1]]['y'])**2 + 
            (landmarks[indices[i]]['z'] - landmarks[indices[i+1]]['z'])**2
        )
    
    # Return the ratio (0-1)
    if segment_sum < 1e-6:  # Avoid division by zero
        return 0.0
    
    return direct_distance / segment_sum

def extract_optimized_features(landmarks):
    """
    Extract 65 optimized features from 21 hand landmarks for better ASL recognition.
    """
    features = {}
    
    # Calculate hand scale for normalization
    hand_scale = np.sqrt(
        (landmarks[5]['x'] - landmarks[17]['x'])**2 + 
        (landmarks[5]['y'] - landmarks[17]['y'])**2 + 
        (landmarks[5]['z'] - landmarks[17]['z'])**2
    )
    
    # Define palm center
    palm_center_x = sum(landmarks[i]['x'] for i in [1, 5, 9, 13, 17]) / 5
    palm_center_y = sum(landmarks[i]['y'] for i in [1, 5, 9, 13, 17]) / 5
    palm_center_z = sum(landmarks[i]['z'] for i in [1, 5, 9, 13, 17]) / 5
    
    # 1. JOINT ANGLE FEATURES (19 features)
    # PIP joint angles (4 features)
    features["index_pip_angle"] = calculate_angle(landmarks[5], landmarks[6], landmarks[7])
    features["middle_pip_angle"] = calculate_angle(landmarks[9], landmarks[10], landmarks[11])
    features["ring_pip_angle"] = calculate_angle(landmarks[13], landmarks[14], landmarks[15])
    features["pinky_pip_angle"] = calculate_angle(landmarks[17], landmarks[18], landmarks[19])
    
    # DIP joint angles (4 features)
    features["index_dip_angle"] = calculate_angle(landmarks[6], landmarks[7], landmarks[8])
    features["middle_dip_angle"] = calculate_angle(landmarks[10], landmarks[11], landmarks[12])
    features["ring_dip_angle"] = calculate_angle(landmarks[14], landmarks[15], landmarks[16])
    features["pinky_dip_angle"] = calculate_angle(landmarks[18], landmarks[19], landmarks[20])
    
    # MCP joint angles (4 features)
    features["index_mcp_angle"] = calculate_angle(landmarks[0], landmarks[5], landmarks[6])
    features["middle_mcp_angle"] = calculate_angle(landmarks[0], landmarks[9], landmarks[10])
    features["ring_mcp_angle"] = calculate_angle(landmarks[0], landmarks[13], landmarks[14])
    features["pinky_mcp_angle"] = calculate_angle(landmarks[0], landmarks[17], landmarks[18])
    
    # Thumb joint angles (3 features)
    features["thumb_cmc_angle"] = calculate_angle(landmarks[0], landmarks[1], landmarks[2])
    features["thumb_mcp_angle"] = calculate_angle(landmarks[1], landmarks[2], landmarks[3])
    features["thumb_ip_angle"] = calculate_angle(landmarks[2], landmarks[3], landmarks[4])
    
    # Inter-finger angles (4 features)
    features["index_middle_angle"] = calculate_angle(landmarks[8], landmarks[5], landmarks[12])
    features["middle_ring_angle"] = calculate_angle(landmarks[12], landmarks[9], landmarks[16])
    features["ring_pinky_angle"] = calculate_angle(landmarks[16], landmarks[13], landmarks[20])
    features["thumb_index_angle"] = calculate_angle(landmarks[4], landmarks[1], landmarks[8])
    
    # 2. FINGERTIP RELATIVE POSITIONS (15 features)
    # Normalized distances from fingertips to wrist (5 features)
    features["thumb_wrist_dist"] = normalized_distance(landmarks[4], landmarks[0], hand_scale)
    features["index_wrist_dist"] = normalized_distance(landmarks[8], landmarks[0], hand_scale)
    features["middle_wrist_dist"] = normalized_distance(landmarks[12], landmarks[0], hand_scale)
    features["ring_wrist_dist"] = normalized_distance(landmarks[16], landmarks[0], hand_scale)
    features["pinky_wrist_dist"] = normalized_distance(landmarks[20], landmarks[0], hand_scale)
    
    # Normalized distances from fingertips to palm center (5 features)
    features["thumb_palm_dist"] = normalized_distance_to_point(landmarks[4], palm_center_x, palm_center_y, palm_center_z, hand_scale)
    features["index_palm_dist"] = normalized_distance_to_point(landmarks[8], palm_center_x, palm_center_y, palm_center_z, hand_scale)
    features["middle_palm_dist"] = normalized_distance_to_point(landmarks[12], palm_center_x, palm_center_y, palm_center_z, hand_scale)
    features["ring_palm_dist"] = normalized_distance_to_point(landmarks[16], palm_center_x, palm_center_y, palm_center_z, hand_scale)
    features["pinky_palm_dist"] = normalized_distance_to_point(landmarks[20], palm_center_x, palm_center_y, palm_center_z, hand_scale)
    
    # Fingertip height differences (5 features)
    features["index_thumb_height"] = (landmarks[8]['y'] - landmarks[4]['y']) / hand_scale
    features["middle_thumb_height"] = (landmarks[12]['y'] - landmarks[4]['y']) / hand_scale
    features["ring_thumb_height"] = (landmarks[16]['y'] - landmarks[4]['y']) / hand_scale
    features["pinky_thumb_height"] = (landmarks[20]['y'] - landmarks[4]['y']) / hand_scale
    features["pinky_index_height"] = (landmarks[20]['y'] - landmarks[8]['y']) / hand_scale
    
    # 3. FINGER STATE FEATURES (10 features)
    # Finger extension ratios (5 features)
    features["index_extension_ratio"] = calculate_finger_extension(landmarks, [5, 6, 7, 8])  # Index finger
    features["middle_extension_ratio"] = calculate_finger_extension(landmarks, [9, 10, 11, 12])  # Middle finger
    features["ring_extension_ratio"] = calculate_finger_extension(landmarks, [13, 14, 15, 16])  # Ring finger
    features["pinky_extension_ratio"] = calculate_finger_extension(landmarks, [17, 18, 19, 20])  # Pinky finger
    features["thumb_extension_ratio"] = calculate_finger_extension(landmarks, [1, 2, 3, 4])  # Thumb
    
    # Binary finger states (5 features)
    features["index_extended"] = 1 if features["index_extension_ratio"] > 0.7 else 0
    features["middle_extended"] = 1 if features["middle_extension_ratio"] > 0.7 else 0
    features["ring_extended"] = 1 if features["ring_extension_ratio"] > 0.7 else 0
    features["pinky_extended"] = 1 if features["pinky_extension_ratio"] > 0.7 else 0
    features["thumb_extended"] = 1 if features["thumb_extension_ratio"] > 0.7 else 0
    
    # 4. RELATIONAL FEATURES (10 features)
    # Normalized distances between fingertips (10 features)
    features["thumb_index_dist"] = normalized_distance(landmarks[4], landmarks[8], hand_scale)
    features["thumb_middle_dist"] = normalized_distance(landmarks[4], landmarks[12], hand_scale)
    features["thumb_ring_dist"] = normalized_distance(landmarks[4], landmarks[16], hand_scale)
    features["thumb_pinky_dist"] = normalized_distance(landmarks[4], landmarks[20], hand_scale)
    features["index_middle_dist"] = normalized_distance(landmarks[8], landmarks[12], hand_scale)
    features["index_ring_dist"] = normalized_distance(landmarks[8], landmarks[16], hand_scale)
    features["index_pinky_dist"] = normalized_distance(landmarks[8], landmarks[20], hand_scale)
    features["middle_ring_dist"] = normalized_distance(landmarks[12], landmarks[16], hand_scale)
    features["middle_pinky_dist"] = normalized_distance(landmarks[12], landmarks[20], hand_scale)
    features["ring_pinky_dist"] = normalized_distance(landmarks[16], landmarks[20], hand_scale)
    
    # 5. HAND ORIENTATION FEATURES (5 features)
    # Calculate palm normal vector
    v1 = np.array([landmarks[5]['x'] - landmarks[0]['x'], landmarks[5]['y'] - landmarks[0]['y'], landmarks[5]['z'] - landmarks[0]['z']])
    v2 = np.array([landmarks[17]['x'] - landmarks[0]['x'], landmarks[17]['y'] - landmarks[0]['y'], landmarks[17]['z'] - landmarks[0]['z']])
    palm_normal = np.cross(v1, v2)
    palm_normal_norm = np.linalg.norm(palm_normal)
    if palm_normal_norm > 0:
        palm_normal = palm_normal / palm_normal_norm
    
    # Palm normal vector components (3 features)
    features["palm_normal_x"] = palm_normal[0]
    features["palm_normal_y"] = palm_normal[1]
    features["palm_normal_z"] = palm_normal[2]
    
    # Palm direction and inclination (2 features)
    palm_direction_x = np.mean([landmarks[5]['x'], landmarks[9]['x'], landmarks[13]['x'], landmarks[17]['x']]) - landmarks[0]['x']
    palm_direction_y = np.mean([landmarks[5]['y'], landmarks[9]['y'], landmarks[13]['y'], landmarks[17]['y']]) - landmarks[0]['y']
    features["palm_direction"] = np.arctan2(palm_direction_y, palm_direction_x)
    
    palm_inclination = np.arctan2(
        np.mean([landmarks[5]['y'], landmarks[9]['y'], landmarks[13]['y'], landmarks[17]['y']]) - landmarks[0]['y'],
        np.mean([landmarks[5]['z'], landmarks[9]['z'], landmarks[13]['z'], landmarks[17]['z']]) - landmarks[0]['z']
    )
    features["palm_inclination"] = palm_inclination
    
    # 6. ADVANCED TOPOLOGICAL FEATURES (6 features)
    # Finger spread angles (4 features)
    features["thumb_index_spread"] = calculate_angle(landmarks[4], landmarks[0], landmarks[8])
    features["index_middle_spread"] = calculate_angle(landmarks[8], landmarks[0], landmarks[12])
    features["middle_ring_spread"] = calculate_angle(landmarks[12], landmarks[0], landmarks[16])
    features["ring_pinky_spread"] = calculate_angle(landmarks[16], landmarks[0], landmarks[20])
    
    # Curvature of the palm (2 features)
    features["palm_curve_1"] = calculate_angle(landmarks[5], landmarks[9], landmarks[13])
    features["palm_curve_2"] = calculate_angle(landmarks[9], landmarks[13], landmarks[17])
    
    return features

def convert_features(input_path, output_path=None):
    """Convert raw landmark features to optimized features and save as CSV."""
    print(f"Loading dataset from {input_path}...")
    
    # Load original dataset
    df = pd.read_csv(input_path)
    
    # Check if we have landmark columns
    landmark_cols = {}
    for i in range(21):
        for coord in ['x', 'y', 'z']:
            col_name = f'landmark_{i}_{coord}'
            if col_name not in df.columns:
                print(f"Error: Expected column {col_name} not found in dataset!")
                return None
            landmark_cols[(i, coord)] = col_name
    
    print("Converting landmarks to optimized features...")
    # Process each row and extract optimized features
    optimized_data = []
    
    # Check if 'label' column exists
    if 'label' not in df.columns:
        print("Error: 'label' column not found in dataset!")
        return None
    
    # Convert landmarks to optimized features
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Create landmark dictionary for each hand
        landmarks = []
        for i in range(21):
            landmark = {
                'x': row[landmark_cols[(i, 'x')]],
                'y': row[landmark_cols[(i, 'y')]],
                'z': row[landmark_cols[(i, 'z')]]
            }
            landmarks.append(landmark)
        
        # Extract optimized features
        features = extract_optimized_features(landmarks)
        
        # Add label and hand type
        features['label'] = row['label']
        if 'hand_type' in df.columns:
            features['hand_type'] = row['hand_type']
        
        # Add sample_type if it exists in the original dataset
        if 'sample_type' in df.columns:
            features['sample_type'] = row['sample_type']
        
        # Add to dataset
        optimized_data.append(features)
    
    # Convert to DataFrame
    optimized_df = pd.DataFrame(optimized_data)
    
    # Set default output path if none provided
    if output_path is None:
        timestamp = datetime.now().strftime('%m%d_%H%M%S')
        output_path = f"/Users/wuhaodong/SFhack/asl_optimized_features_{timestamp}.csv"
    
    # Save optimized features to CSV
    optimized_df.to_csv(output_path, index=False)
    print(f"Successfully converted {len(optimized_df)} samples to optimized features")
    print(f"Optimized features saved to: {output_path}")
    
    # Print dataset details
    print(f"\nDataset information:")
    print(f"Number of samples: {len(optimized_df)}")
    
    # Calculate number of features (excluding metadata columns)
    excluded_cols = ['label', 'hand_type', 'sample_type']
    feature_count = len(optimized_df.columns) - sum(col in optimized_df.columns for col in excluded_cols)
    print(f"Number of features: {feature_count}")
    
    print(f"Number of classes: {optimized_df['label'].nunique()}")
    print(f"Classes: {', '.join(sorted(optimized_df['label'].unique()))}")
    
    # If sample_type exists, show distribution
    if 'sample_type' in optimized_df.columns:
        sample_type_counts = optimized_df['sample_type'].value_counts()
        print("\nSample Type Distribution:")
        for sample_type, count in sample_type_counts.items():
            print(f"  {sample_type}: {count} samples ({count/len(optimized_df)*100:.2f}%)")
    
    return optimized_df

if __name__ == "__main__":
    input_path = "/Users/wuhaodong/SFhack/asl_features.csv"
    output_path = None  # Will generate a timestamped filename
    
    # Convert features
    convert_features(input_path, output_path) 
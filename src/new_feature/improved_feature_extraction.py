import numpy as np

def extract_optimized_features(landmarks, is_right_hand=True):
    """
    Extract 65 optimized features from 21 hand landmarks for better ASL recognition.
    
    Args:
        landmarks: List of 21 MediaPipe hand landmarks
        is_right_hand: Boolean indicating if the hand is right (True) or left (False)
    
    Returns:
        features: Numpy array of 65 features
    """
    features = []
    
    # Calculate hand scale for normalization
    hand_scale = np.sqrt(
        (landmarks[5].x - landmarks[17].x)**2 + 
        (landmarks[5].y - landmarks[17].y)**2 + 
        (landmarks[5].z - landmarks[17].z)**2
    )
    
    # Define palm center
    palm_center_x = sum(landmarks[i].x for i in [1, 5, 9, 13, 17]) / 5
    palm_center_y = sum(landmarks[i].y for i in [1, 5, 9, 13, 17]) / 5
    palm_center_z = sum(landmarks[i].z for i in [1, 5, 9, 13, 17]) / 5
    
    # 1. JOINT ANGLE FEATURES (19 features)
    # PIP joint angles (4 features)
    features.append(calculate_angle(landmarks[5], landmarks[6], landmarks[7]))  # index PIP
    features.append(calculate_angle(landmarks[9], landmarks[10], landmarks[11]))  # middle PIP
    features.append(calculate_angle(landmarks[13], landmarks[14], landmarks[15]))  # ring PIP
    features.append(calculate_angle(landmarks[17], landmarks[18], landmarks[19]))  # pinky PIP
    
    # DIP joint angles (4 features)
    features.append(calculate_angle(landmarks[6], landmarks[7], landmarks[8]))  # index DIP
    features.append(calculate_angle(landmarks[10], landmarks[11], landmarks[12]))  # middle DIP
    features.append(calculate_angle(landmarks[14], landmarks[15], landmarks[16]))  # ring DIP
    features.append(calculate_angle(landmarks[18], landmarks[19], landmarks[20]))  # pinky DIP
    
    # MCP joint angles (4 features)
    features.append(calculate_angle(landmarks[0], landmarks[5], landmarks[6]))  # index MCP
    features.append(calculate_angle(landmarks[0], landmarks[9], landmarks[10]))  # middle MCP
    features.append(calculate_angle(landmarks[0], landmarks[13], landmarks[14]))  # ring MCP
    features.append(calculate_angle(landmarks[0], landmarks[17], landmarks[18]))  # pinky MCP
    
    # Thumb joint angles (3 features)
    features.append(calculate_angle(landmarks[0], landmarks[1], landmarks[2]))  # CMC
    features.append(calculate_angle(landmarks[1], landmarks[2], landmarks[3]))  # MCP
    features.append(calculate_angle(landmarks[2], landmarks[3], landmarks[4]))  # IP
    
    # Inter-finger angles (4 features)
    features.append(calculate_angle(landmarks[8], landmarks[5], landmarks[12]))  # index-middle
    features.append(calculate_angle(landmarks[12], landmarks[9], landmarks[16]))  # middle-ring
    features.append(calculate_angle(landmarks[16], landmarks[13], landmarks[20]))  # ring-pinky
    features.append(calculate_angle(landmarks[4], landmarks[1], landmarks[8]))  # thumb-index
    
    # 2. FINGERTIP RELATIVE POSITIONS (15 features)
    # Normalized distances from fingertips to wrist (5 features)
    features.append(normalized_distance(landmarks[4], landmarks[0], hand_scale))  # thumb-wrist
    features.append(normalized_distance(landmarks[8], landmarks[0], hand_scale))  # index-wrist
    features.append(normalized_distance(landmarks[12], landmarks[0], hand_scale))  # middle-wrist
    features.append(normalized_distance(landmarks[16], landmarks[0], hand_scale))  # ring-wrist
    features.append(normalized_distance(landmarks[20], landmarks[0], hand_scale))  # pinky-wrist
    
    # Normalized distances from fingertips to palm center (5 features)
    features.append(normalized_distance_to_point(landmarks[4], palm_center_x, palm_center_y, palm_center_z, hand_scale))  # thumb-palm
    features.append(normalized_distance_to_point(landmarks[8], palm_center_x, palm_center_y, palm_center_z, hand_scale))  # index-palm
    features.append(normalized_distance_to_point(landmarks[12], palm_center_x, palm_center_y, palm_center_z, hand_scale))  # middle-palm
    features.append(normalized_distance_to_point(landmarks[16], palm_center_x, palm_center_y, palm_center_z, hand_scale))  # ring-palm
    features.append(normalized_distance_to_point(landmarks[20], palm_center_x, palm_center_y, palm_center_z, hand_scale))  # pinky-palm
    
    # Fingertip height differences (5 features)
    features.append((landmarks[8].y - landmarks[4].y) / hand_scale)  # index-thumb height
    features.append((landmarks[12].y - landmarks[4].y) / hand_scale)  # middle-thumb height
    features.append((landmarks[16].y - landmarks[4].y) / hand_scale)  # ring-thumb height
    features.append((landmarks[20].y - landmarks[4].y) / hand_scale)  # pinky-thumb height
    features.append((landmarks[20].y - landmarks[8].y) / hand_scale)  # pinky-index height
    
    # 3. FINGER STATE FEATURES (10 features)
    # Finger extension ratios (5 features)
    features.append(calculate_finger_extension(landmarks, 'index'))
    features.append(calculate_finger_extension(landmarks, 'middle'))
    features.append(calculate_finger_extension(landmarks, 'ring'))
    features.append(calculate_finger_extension(landmarks, 'pinky'))
    features.append(calculate_finger_extension(landmarks, 'thumb'))
    
    # Binary finger states (5 features)
    features.append(1 if features[-5] > 0.7 else 0)  # index extended
    features.append(1 if features[-4] > 0.7 else 0)  # middle extended
    features.append(1 if features[-3] > 0.7 else 0)  # ring extended
    features.append(1 if features[-2] > 0.7 else 0)  # pinky extended
    features.append(1 if features[-1] > 0.7 else 0)  # thumb extended
    
    # 4. RELATIONAL FEATURES (10 features)
    # Normalized distances between fingertips (10 features)
    features.append(normalized_distance(landmarks[4], landmarks[8], hand_scale))  # thumb-index
    features.append(normalized_distance(landmarks[4], landmarks[12], hand_scale))  # thumb-middle
    features.append(normalized_distance(landmarks[4], landmarks[16], hand_scale))  # thumb-ring
    features.append(normalized_distance(landmarks[4], landmarks[20], hand_scale))  # thumb-pinky
    features.append(normalized_distance(landmarks[8], landmarks[12], hand_scale))  # index-middle
    features.append(normalized_distance(landmarks[8], landmarks[16], hand_scale))  # index-ring
    features.append(normalized_distance(landmarks[8], landmarks[20], hand_scale))  # index-pinky
    features.append(normalized_distance(landmarks[12], landmarks[16], hand_scale))  # middle-ring
    features.append(normalized_distance(landmarks[12], landmarks[20], hand_scale))  # middle-pinky
    features.append(normalized_distance(landmarks[16], landmarks[20], hand_scale))  # ring-pinky
    
    # 5. HAND ORIENTATION FEATURES (5 features)
    # Calculate palm normal vector
    v1 = np.array([landmarks[5].x - landmarks[0].x, landmarks[5].y - landmarks[0].y, landmarks[5].z - landmarks[0].z])
    v2 = np.array([landmarks[17].x - landmarks[0].x, landmarks[17].y - landmarks[0].y, landmarks[17].z - landmarks[0].z])
    palm_normal = np.cross(v1, v2)
    palm_normal = palm_normal / np.linalg.norm(palm_normal)
    
    # Palm normal vector components (3 features)
    features.append(palm_normal[0])  # palm normal x
    features.append(palm_normal[1])  # palm normal y
    features.append(palm_normal[2])  # palm normal z
    
    # Palm direction and inclination (2 features)
    palm_direction_x = np.mean([landmarks[5].x, landmarks[9].x, landmarks[13].x, landmarks[17].x]) - landmarks[0].x
    palm_direction_y = np.mean([landmarks[5].y, landmarks[9].y, landmarks[13].y, landmarks[17].y]) - landmarks[0].y
    palm_direction = np.arctan2(palm_direction_y, palm_direction_x)
    features.append(palm_direction)
    
    palm_inclination = np.arctan2(
        np.mean([landmarks[5].y, landmarks[9].y, landmarks[13].y, landmarks[17].y]) - landmarks[0].y,
        np.mean([landmarks[5].z, landmarks[9].z, landmarks[13].z, landmarks[17].z]) - landmarks[0].z
    )
    features.append(palm_inclination)
    
    # 6. ADVANCED TOPOLOGICAL FEATURES (6 features)
    # Finger spread angles (4 features)
    features.append(calculate_angle(landmarks[4], landmarks[0], landmarks[8]))  # thumb-index spread
    features.append(calculate_angle(landmarks[8], landmarks[0], landmarks[12]))  # index-middle spread
    features.append(calculate_angle(landmarks[12], landmarks[0], landmarks[16]))  # middle-ring spread
    features.append(calculate_angle(landmarks[16], landmarks[0], landmarks[20]))  # ring-pinky spread
    
    # Curvature of the palm (2 features)
    features.append(calculate_angle(landmarks[5], landmarks[9], landmarks[13]))  # palm curve 1
    features.append(calculate_angle(landmarks[9], landmarks[13], landmarks[17]))  # palm curve 2
    
    return np.array(features)

def calculate_angle(p1, p2, p3):
    """Calculate angle between three points in 3D space."""
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    
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
        (p1.x - p2.x)**2 + 
        (p1.y - p2.y)**2 + 
        (p1.z - p2.z)**2
    )
    return dist / hand_scale

def normalized_distance_to_point(p, x, y, z, hand_scale):
    """Calculate normalized Euclidean distance from a landmark to a point."""
    dist = np.sqrt(
        (p.x - x)**2 + 
        (p.y - y)**2 + 
        (p.z - z)**2
    )
    return dist / hand_scale

def calculate_finger_extension(landmarks, finger):
    """
    Calculate how extended a finger is.
    
    Returns a ratio between 0 (completely curled) and 1 (fully extended).
    """
    if finger == 'thumb':
        # Indices for thumb landmarks: CMC, MCP, IP, TIP
        indices = [1, 2, 3, 4]
    elif finger == 'index':
        # Indices for index finger landmarks: MCP, PIP, DIP, TIP
        indices = [5, 6, 7, 8]
    elif finger == 'middle':
        # Indices for middle finger landmarks: MCP, PIP, DIP, TIP
        indices = [9, 10, 11, 12]
    elif finger == 'ring':
        # Indices for ring finger landmarks: MCP, PIP, DIP, TIP
        indices = [13, 14, 15, 16]
    elif finger == 'pinky':
        # Indices for pinky finger landmarks: MCP, PIP, DIP, TIP
        indices = [17, 18, 19, 20]
    else:
        raise ValueError(f"Invalid finger: {finger}")
    
    # Calculate direct distance from base to tip
    direct_distance = np.sqrt(
        (landmarks[indices[0]].x - landmarks[indices[3]].x)**2 + 
        (landmarks[indices[0]].y - landmarks[indices[3]].y)**2 + 
        (landmarks[indices[0]].z - landmarks[indices[3]].z)**2
    )
    
    # Calculate sum of all segments
    segment_sum = 0
    for i in range(len(indices) - 1):
        segment_sum += np.sqrt(
            (landmarks[indices[i]].x - landmarks[indices[i+1]].x)**2 + 
            (landmarks[indices[i]].y - landmarks[indices[i+1]].y)**2 + 
            (landmarks[indices[i]].z - landmarks[indices[i+1]].z)**2
        )
    
    # Return the ratio (0-1)
    if segment_sum < 1e-6:  # Avoid division by zero
        return 0.0
    
    return direct_distance / segment_sum

def get_feature_names():
    """Return the names of all 65 features for reference."""
    return [
        # 1. Joint Angle Features (19)
        "index_pip_angle", "middle_pip_angle", "ring_pip_angle", "pinky_pip_angle",
        "index_dip_angle", "middle_dip_angle", "ring_dip_angle", "pinky_dip_angle",
        "index_mcp_angle", "middle_mcp_angle", "ring_mcp_angle", "pinky_mcp_angle",
        "thumb_cmc_angle", "thumb_mcp_angle", "thumb_ip_angle",
        "index_middle_angle", "middle_ring_angle", "ring_pinky_angle", "thumb_index_angle",
        
        # 2. Fingertip Relative Positions (15)
        "thumb_wrist_dist", "index_wrist_dist", "middle_wrist_dist", "ring_wrist_dist", "pinky_wrist_dist",
        "thumb_palm_dist", "index_palm_dist", "middle_palm_dist", "ring_palm_dist", "pinky_palm_dist",
        "index_thumb_height", "middle_thumb_height", "ring_thumb_height", "pinky_thumb_height", "pinky_index_height",
        
        # 3. Finger State Features (10)
        "index_extension_ratio", "middle_extension_ratio", "ring_extension_ratio", "pinky_extension_ratio", "thumb_extension_ratio",
        "index_extended", "middle_extended", "ring_extended", "pinky_extended", "thumb_extended",
        
        # 4. Relational Features (10)
        "thumb_index_dist", "thumb_middle_dist", "thumb_ring_dist", "thumb_pinky_dist",
        "index_middle_dist", "index_ring_dist", "index_pinky_dist",
        "middle_ring_dist", "middle_pinky_dist", "ring_pinky_dist",
        
        # 5. Hand Orientation Features (5)
        "palm_normal_x", "palm_normal_y", "palm_normal_z", "palm_direction", "palm_inclination",
        
        # 6. Advanced Topological Features (6)
        "thumb_index_spread", "index_middle_spread", "middle_ring_spread", "ring_pinky_spread",
        "palm_curve_1", "palm_curve_2"
    ]

# Example of how to use the feature extraction
if __name__ == "__main__":
    # This is just a placeholder for testing - real implementation would use MediaPipe landmarks
    # Example with dummy landmarks to ensure code compiles correctly
    class Landmark:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z
    
    # Generate a test set of landmarks
    landmarks = [Landmark(i/10, i/20, i/30) for i in range(21)]
    
    # Extract features
    features = extract_optimized_features(landmarks)
    
    # Print feature names and values
    feature_names = get_feature_names()
    for name, value in zip(feature_names, features):
        print(f"{name}: {value:.4f}")

    print(f"Total number of features: {len(features)}") 
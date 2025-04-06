import cv2
import numpy as np
import os
import argparse
import glob
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import joblib

def parse_args():
    parser = argparse.ArgumentParser(description='Extract MediaPipe hand landmarks from visualization images (dots only)')
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='Directory containing MediaPipe landmark visualization images')
    parser.add_argument('--output_dir', type=str, default='./data/processed', 
                        help='Directory to save extracted features')
    parser.add_argument('--format', type=str, choices=['csv', 'npy'], default='csv',
                        help='Output format (default: csv)')
    parser.add_argument('--debug', action='store_true', 
                        help='Save debug images with detected points')
    parser.add_argument('--sample', type=int, default=0,
                        help='Process only a sample of images (0 for all)')
    return parser.parse_args()

def extract_landmarks_from_image(image_path, debug=False, output_dir=None):
    """Extract MediaPipe hand landmarks from visualization image, focusing on dots only."""
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return None
    
    # Define exact RGB colors for each type of landmark based on color picker results
    # Note: OpenCV loads images as BGR
    # We now have multiple color profiles for each landmark type to handle variations
    color_profiles = {
        'red': [
            (48, 48, 255),    # Red profile 1
            (30, 30, 220),    # Red profile 2 (darker)
            (70, 70, 255),    # Red profile 3 (lighter)
            (20, 20, 180)     # Red profile 4 (much darker)
        ],
        'blue': [
            (192, 101, 21),   # Blue profile 1
            (180, 80, 10),    # Blue profile 2 (darker)
            (210, 120, 40),   # Blue profile 3 (lighter)
            (160, 60, 0)      # Blue profile 4 (much darker)
        ],
        'green': [
            (48, 255, 48),    # Green profile 1
            (30, 230, 30),    # Green profile 2 (darker)
            (70, 255, 70),    # Green profile 3 (lighter)
            (20, 200, 20)     # Green profile 4 (much darker)
        ],
        'yellow': [
            (0, 204, 255),    # Yellow profile 1
            (0, 180, 230),    # Yellow profile 2 (darker)
            (30, 220, 255),   # Yellow profile 3 (lighter)
            (0, 150, 200)     # Yellow profile 4 (much darker)
        ],
        'purple': [
            (128, 64, 128),   # Purple profile 1
            (110, 50, 110),   # Purple profile 2 (darker)
            (150, 80, 150),   # Purple profile 3 (lighter)
            (90, 40, 90)      # Purple profile 4 (much darker)
        ],
        'cream': [
            (180, 229, 255),  # Cream profile 1
            (160, 210, 235),  # Cream profile 2 (darker)
            (200, 240, 255),  # Cream profile 3 (lighter)
            (140, 190, 215)   # Cream profile 4 (much darker)
        ],
        'gray': [
            (128, 128, 128)   # Gray for lines
        ]
    }
    
    # Store the landmark point data
    all_points = []
    landmark_points = {}  # Store detected points by color
    height, width = image.shape[:2]
    
    # Debug image
    if debug:
        debug_image = image.copy()
    
    # Try adaptive brightness enhancement for images that might be too dark
    enhanced_image = None
    
    # Function to process an image with color profiles
    def process_with_color_profiles(img, color_name, profiles, threshold=25):
        points_found = []
        
        for bgr_value in profiles:
            # Create a mask that matches this exact color (or close)
            lower_bound = np.array([max(0, c - threshold) for c in bgr_value])
            upper_bound = np.array([min(255, c + threshold) for c in bgr_value])
            
            mask = cv2.inRange(img, lower_bound, upper_bound)
            
            # Find contours of the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # For each contour, check if it's a circular dot
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Only consider contours within a reasonable size range for dots
                if 5 < area < 500:
                    # Get the center of the contour
                    M = cv2.moments(contour)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Check if the contour is roughly circular
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            if circularity > 0.4:  # Relaxed filter for circular shapes
                                # Check for duplicates (same point detected by different profiles)
                                is_duplicate = False
                                for existing_x, existing_y in points_found:
                                    # If point is very close to an existing one, skip it
                                    if abs(cx - existing_x) < 5 and abs(cy - existing_y) < 5:
                                        is_duplicate = True
                                        break
                                
                                if not is_duplicate:
                                    points_found.append((cx, cy))
                                    
                                    # Add this point to global collections
                                    all_points.append((cx, cy, color_name))
                                    
                                    if color_name not in landmark_points:
                                        landmark_points[color_name] = []
                                    landmark_points[color_name].append((cx, cy))
                                    
                                    # Draw on debug image
                                    if debug:
                                        # Pick the first profile color for visualization
                                        display_color = color_profiles[color_name][0]
                                        cv2.circle(debug_image, (cx, cy), 5, display_color, -1)
                                        cv2.putText(debug_image, f"{color_name[0]}", (cx+5, cy), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return points_found
    
    # Process the original image with all color profiles
    for color_name, profiles in color_profiles.items():
        # Skip gray as it's used for lines, not landmarks
        if color_name == 'gray':
            continue
            
        process_with_color_profiles(image, color_name, profiles)
    
    # If we didn't find enough points, try with enhanced image
    if len(all_points) < 10:
        # Create an enhanced version with better contrast
        enhanced_image = image.copy()
        
        # Apply contrast enhancement
        lab = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Try again with enhanced image and larger threshold
        for color_name, profiles in color_profiles.items():
            # Skip gray as it's used for lines, not landmarks
            if color_name == 'gray':
                continue
                
            process_with_color_profiles(enhanced_image, color_name, profiles, threshold=40)
    
    # If we still don't have enough points, try automatic color clustering
    if len(all_points) < 10:
        try:
            # Convert to a list of all pixel colors (excluding background)
            pixels = []
            # Use a simple threshold to exclude background (usually white or black)
            mask = np.zeros((height, width), dtype=np.uint8)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)  # Invert to get non-white pixels
            
            # Dilate slightly to include more of the dot
            kernel = np.ones((3,3), np.uint8)
            binary = cv2.dilate(binary, kernel, iterations=1)
            
            # Find contours in the binary image
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process each contour
            for contour in contours:
                area = cv2.contourArea(contour)
                if 5 < area < 500:  # Filter by area
                    # Create a mask for this contour
                    contour_mask = np.zeros_like(binary)
                    cv2.drawContours(contour_mask, [contour], 0, 255, -1)
                    
                    # Find the average color in this contour
                    contour_avg_color = cv2.mean(image, mask=contour_mask)[:3]  # BGR
                    
                    # Find the centroid
                    M = cv2.moments(contour)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Add to our list if not a duplicate
                        is_duplicate = False
                        for x, y, _ in all_points:
                            if abs(cx - x) < 5 and abs(cy - y) < 5:
                                is_duplicate = True
                                break
                                
                        if not is_duplicate:
                            # Determine closest color profile
                            min_dist = float('inf')
                            best_color = None
                            
                            for color_name, profiles in color_profiles.items():
                                if color_name == 'gray':
                                    continue
                                    
                                for profile in profiles:
                                    dist = np.sum(np.abs(np.array(contour_avg_color) - np.array(profile)))
                                    if dist < min_dist:
                                        min_dist = dist
                                        best_color = color_name
                            
                            # Only add if we have a reasonable match
                            if min_dist < 300:
                                all_points.append((cx, cy, best_color))
                                
                                if best_color not in landmark_points:
                                    landmark_points[best_color] = []
                                landmark_points[best_color].append((cx, cy))
                                
                                # Draw on debug image
                                if debug:
                                    display_color = color_profiles[best_color][0]
                                    cv2.circle(debug_image, (cx, cy), 5, display_color, -1)
                                    cv2.putText(debug_image, f"{best_color[0]}", (cx+5, cy), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        except Exception as e:
            print(f"Error during color clustering: {str(e)}")
    
    # Check if we found enough points
    total_points = len(all_points)
    # Reduce minimum threshold to 6 points instead of 10
    min_required_points = 6  
    
    if total_points < min_required_points:
        print(f"Found only {total_points} landmarks in {image_path}, expected around 21")
        
        # Count by color for debugging
        color_counts = {}
        for _, _, color in all_points:
            if color not in color_counts:
                color_counts[color] = 0
            color_counts[color] += 1
            
        for color, count in color_counts.items():
            print(f"  {color}: {count}")
        
        if debug:
            # Save debug image showing detected points
            debug_dir = os.path.join(output_dir or ".", "debug")
            os.makedirs(debug_dir, exist_ok=True)
            class_name = os.path.basename(os.path.dirname(image_path))
            file_name = os.path.basename(image_path)
            debug_path = os.path.join(debug_dir, f"failed_{class_name}_{file_name}")
            cv2.imwrite(debug_path, debug_image)
            
            # If we used enhanced image, save that too
            if enhanced_image is not None:
                enhanced_debug_path = os.path.join(debug_dir, f"enhanced_{class_name}_{file_name}")
                cv2.imwrite(enhanced_debug_path, enhanced_image)
            
        # Try fallback method for green-only points with numbers
        if total_points < 5 and debug:
            # This could be a numbered green points visualization
            print(f"Attempting fallback method for numbered green points in {image_path}")
            return extract_numbered_landmarks(image_path, debug, output_dir)
            
        return None
    
    # If we have points, try to organize them into the MediaPipe format
    # First, determine if we have a complete hand representation
    # or just partial landmarks
    is_partial = total_points < 15
    
    # Look for a likely wrist point (typically lowest point)
    all_xy_points = [(x, y) for x, y, _ in all_points]
    sorted_by_y = sorted(all_xy_points, key=lambda p: p[1], reverse=True)
    
    # The wrist is likely one of the lowest points
    wrist_candidates = sorted_by_y[:3]  # Take the 3 lowest points
    
    # Among the candidates, the wrist is likely most central horizontally
    wrist = sorted(wrist_candidates, key=lambda p: abs(p[0] - width/2))[0]
    
    # Start building the ordered landmarks with the wrist
    landmarks_ordered = [wrist]
    remaining_points = [p for p in all_xy_points if p != wrist]
    
    # If we have color information, use it to structure the hand
    if 'red' in landmark_points and len(landmark_points['red']) > 0:
        # Hand typically has red points for the palm joints
        palm_points = sorted(landmark_points['red'], key=lambda p: p[1], reverse=True)
        
        # Wrist (point 0) should be the lowest red point
        if palm_points:
            wrist = palm_points[0]
            landmarks_ordered = [wrist]
            
            # If we have thumb points (cream/beige)
            if 'cream' in landmark_points and len(landmark_points['cream']) > 0:
                # Sort thumb points from base to tip (away from wrist)
                thumb_points = sorted(landmark_points['cream'], 
                                      key=lambda p: np.sqrt((p[0] - wrist[0])**2 + (p[1] - wrist[1])**2))
                landmarks_ordered.extend(thumb_points)
            
            # Fill in with 4 dummy points if thumb wasn't found
            while len(landmarks_ordered) < 5:
                landmarks_ordered.append(wrist)  # Just duplicate wrist as placeholder
            
            # Add rest of the palm and fingers based on available colors
            finger_colors = ['blue', 'yellow', 'green', 'purple']
            for i, color in enumerate(finger_colors):
                # Next palm point if available
                if i + 1 < len(palm_points):
                    landmarks_ordered.append(palm_points[i + 1])
                else:
                    landmarks_ordered.append(wrist)  # Placeholder
                    
                # Finger points for this finger
                if color in landmark_points:
                    finger_points = sorted(landmark_points[color], 
                                          key=lambda p: np.sqrt((p[0] - landmarks_ordered[-1][0])**2 + 
                                                               (p[1] - landmarks_ordered[-1][1])**2))
                    landmarks_ordered.extend(finger_points)
                
                # Fill with placeholders to ensure 3 points per finger
                while len(landmarks_ordered) < 5 + (i + 1) * 4:
                    landmarks_ordered.append(landmarks_ordered[-1])
    else:
        # Fallback: just use distance-based organization
        for _ in range(min(20, len(remaining_points))):
            last_point = landmarks_ordered[-1]
            
            # Find closest point to last point
            closest_idx = 0
            closest_dist = float('inf')
            
            for i, point in enumerate(remaining_points):
                dist = np.sqrt((point[0] - last_point[0])**2 + (point[1] - last_point[1])**2)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_idx = i
            
            # Add this point and remove from remaining
            next_point = remaining_points.pop(closest_idx)
            landmarks_ordered.append(next_point)
    
    # Ensure exactly 21 points
    while len(landmarks_ordered) < 21:
        landmarks_ordered.append(landmarks_ordered[-1])
    
    landmarks_ordered = landmarks_ordered[:21]  # Truncate if more than 21
    
    # For partial landmarks (less than 15 points), add a feature to indicate this
    is_partial_landmarks = 1 if total_points < 15 else 0
    
    # Normalize coordinates to 0-1 range
    normalized_landmarks = []
    for x, y in landmarks_ordered:
        normalized_landmarks.append([x / width, y / height, 0.0])  # Z is set to 0
    
    # Add metadata about this extraction
    metadata = {
        "total_detected_points": total_points,
        "colors_detected": list(landmark_points.keys()),
        "is_partial": is_partial_landmarks
    }
    
    # Draw ordered landmarks with indices for debugging
    if debug:
        ordered_debug = image.copy()
        # Draw connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # index finger
            (0, 9), (9, 10), (10, 11), (11, 12),  # middle finger
            (0, 13), (13, 14), (14, 15), (15, 16),  # ring finger
            (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
            (0, 5), (5, 9), (9, 13), (13, 17)  # palm
        ]
        
        for connection in connections:
            if connection[0] < len(landmarks_ordered) and connection[1] < len(landmarks_ordered):
                start_point = landmarks_ordered[connection[0]]
                end_point = landmarks_ordered[connection[1]]
                cv2.line(ordered_debug, 
                        (int(start_point[0]), int(start_point[1])),
                        (int(end_point[0]), int(end_point[1])),
                        (100, 100, 100), 1)
        
        # Color points based on landmark index
        for i, (x, y) in enumerate(landmarks_ordered):
            if i == 0:  # Wrist
                color = (0, 0, 255)  # Red
            elif 1 <= i <= 4:  # Thumb
                color = (224, 224, 255)  # Cream
            elif 5 <= i <= 8:  # Index finger
                color = (255, 0, 0)  # Blue
            elif 9 <= i <= 12:  # Middle finger
                color = (0, 255, 255)  # Yellow
            elif 13 <= i <= 16:  # Ring finger
                color = (0, 255, 0)  # Green
            elif 17 <= i <= 20:  # Pinky
                color = (255, 0, 255)  # Purple
            
            cv2.circle(ordered_debug, (x, y), 5, color, -1)
            cv2.putText(ordered_debug, str(i), (x+5, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add metadata text to image
        y_pos = 20
        cv2.putText(ordered_debug, f"Points: {total_points}/21", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 20
        cv2.putText(ordered_debug, f"Colors: {', '.join(landmark_points.keys())}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 20
        cv2.putText(ordered_debug, f"Partial: {is_partial_landmarks}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save ordered debug image
        debug_dir = os.path.join(output_dir or ".", "debug")
        os.makedirs(debug_dir, exist_ok=True)
        class_name = os.path.basename(os.path.dirname(image_path))
        file_name = os.path.basename(image_path)
        debug_path = os.path.join(debug_dir, f"ordered_{class_name}_{file_name}")
        cv2.imwrite(debug_path, ordered_debug)
    
    return normalized_landmarks, metadata

def extract_numbered_landmarks(image_path, debug=False, output_dir=None):
    """Alternative method for extracting landmarks from images with numbered green points."""
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    height, width = image.shape[:2]
    
    # Convert to HSV for better green detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Green color range (adjust as needed)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    # Create mask for green pixels
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Clean up mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get centroids of all green blobs
    green_points = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 5 < area < 500:  # Adjust thresholds as needed
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                green_points.append((cx, cy))
    
    print(f"Found {len(green_points)} green points in {image_path}")
    
    if len(green_points) < 6:  # Reduced minimum threshold to match main function
        return None
    
    # Debug image
    if debug:
        debug_image = image.copy()
        for i, (x, y) in enumerate(green_points):
            cv2.circle(debug_image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(debug_image, str(i), (x+5, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        debug_dir = os.path.join(output_dir or ".", "debug")
        os.makedirs(debug_dir, exist_ok=True)
        class_name = os.path.basename(os.path.dirname(image_path))
        file_name = os.path.basename(image_path)
        debug_path = os.path.join(debug_dir, f"green_{class_name}_{file_name}")
        cv2.imwrite(debug_path, debug_image)
    
    # Try to order the points
    # First find the wrist (likely at the bottom of the hand)
    sorted_by_y = sorted(green_points, key=lambda p: p[1], reverse=True)
    wrist_candidates = sorted_by_y[:3]
    wrist = sorted(wrist_candidates, key=lambda p: abs(p[0] - width/2))[0]
    
    # Order by distance from previous point
    landmarks_ordered = [wrist]
    remaining_points = [p for p in green_points if p != wrist]
    
    for _ in range(min(20, len(remaining_points))):
        last_point = landmarks_ordered[-1]
        
        # Find closest point to last point
        closest_idx = 0
        closest_dist = float('inf')
        
        for i, point in enumerate(remaining_points):
            dist = np.sqrt((point[0] - last_point[0])**2 + (point[1] - last_point[1])**2)
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = i
        
        # Add this point and remove from remaining
        next_point = remaining_points.pop(closest_idx)
        landmarks_ordered.append(next_point)
    
    # Ensure exactly 21 points
    while len(landmarks_ordered) < 21:
        landmarks_ordered.append(landmarks_ordered[-1])
    
    landmarks_ordered = landmarks_ordered[:21]
    
    # Normalize coordinates
    normalized_landmarks = []
    for x, y in landmarks_ordered:
        normalized_landmarks.append([x / width, y / height, 0.0])
    
    # Create metadata similar to main extraction function
    metadata = {
        "total_detected_points": len(green_points),
        "colors_detected": ["green"],
        "is_partial": 1 if len(green_points) < 15 else 0
    }
    
    # Debug visualization with ordered landmarks
    if debug:
        ordered_debug = image.copy()
        # Draw connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # index finger
            (0, 9), (9, 10), (10, 11), (11, 12),  # middle finger
            (0, 13), (13, 14), (14, 15), (15, 16),  # ring finger
            (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
            (0, 5), (5, 9), (9, 13), (13, 17)  # palm
        ]
        
        for connection in connections:
            if connection[0] < len(landmarks_ordered) and connection[1] < len(landmarks_ordered):
                start_point = landmarks_ordered[connection[0]]
                end_point = landmarks_ordered[connection[1]]
                cv2.line(ordered_debug, 
                        (int(start_point[0]), int(start_point[1])),
                        (int(end_point[0]), int(end_point[1])),
                        (100, 100, 100), 1)
        
        # Draw points
        for i, (x, y) in enumerate(landmarks_ordered):
            cv2.circle(ordered_debug, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(ordered_debug, str(i), (x+5, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
        # Add metadata text
        y_pos = 20
        cv2.putText(ordered_debug, f"Points: {len(green_points)}/21", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 20
        cv2.putText(ordered_debug, f"Colors: green", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 20
        cv2.putText(ordered_debug, f"Partial: {metadata['is_partial']}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save ordered debug image
        debug_dir = os.path.join(output_dir or ".", "debug")
        os.makedirs(debug_dir, exist_ok=True)
        class_name = os.path.basename(os.path.dirname(image_path))
        file_name = os.path.basename(image_path)
        debug_path = os.path.join(debug_dir, f"ordered_green_{class_name}_{file_name}")
        cv2.imwrite(debug_path, ordered_debug)
    
    return normalized_landmarks, metadata

def calculate_features(landmarks, metadata=None):
    """Calculate features from normalized landmarks."""
    features = []
    
    # Add raw landmark coordinates (x, y, z for each of the 21 landmarks)
    for landmark in landmarks:
        features.extend(landmark)  # x, y, z
    
    # Add calculated features (distances between key points)
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    wrist = landmarks[0]
    
    # Distance between fingertips and wrist
    for tip in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]:
        dist = np.sqrt((tip[0] - wrist[0])**2 + (tip[1] - wrist[1])**2 + (tip[2] - wrist[2])**2)
        features.append(dist)
    
    # Distance between adjacent fingertips
    adjacent_pairs = [(thumb_tip, index_tip), (index_tip, middle_tip), 
                      (middle_tip, ring_tip), (ring_tip, pinky_tip)]
    for tip1, tip2 in adjacent_pairs:
        dist = np.sqrt((tip1[0] - tip2[0])**2 + (tip1[1] - tip2[1])**2 + (tip1[2] - tip2[2])**2)
        features.append(dist)
    
    # Distance from thumb to each other fingertip
    for tip in [index_tip, middle_tip, ring_tip, pinky_tip]:
        dist = np.sqrt((thumb_tip[0] - tip[0])**2 + (thumb_tip[1] - tip[1])**2 + (thumb_tip[2] - tip[2])**2)
        features.append(dist)
    
    # Add relative positions of fingers to wrist
    for i in [4, 8, 12, 16, 20]:  # Fingertips
        features.append(landmarks[i][0] - wrist[0])  # x relative to wrist
        features.append(landmarks[i][1] - wrist[1])  # y relative to wrist
    
    # Add metadata if available
    if metadata:
        # Add a confidence score based on number of detected points (0-1)
        detected_ratio = metadata["total_detected_points"] / 21.0
        features.append(detected_ratio)
        
        # Add partial landmark flag
        features.append(metadata["is_partial"])
        
        # Add color presence flags (1 if color is present, 0 if not)
        color_flags = []
        for color in ['red', 'blue', 'green', 'yellow', 'purple', 'cream']:
            color_flags.append(1 if color in metadata["colors_detected"] else 0)
        features.extend(color_flags)
    
    return features

def process_images(input_dir, output_dir, output_format='csv', debug=False, sample=0):
    """Process images in the input directory and extract landmarks."""
    # Find all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))
    
    # Filter out images in debug directories
    filtered_files = []
    for img_file in image_files:
        # Skip images in debug directories
        if '/debug/' not in img_file and '\\debug\\' not in img_file:
            filtered_files.append(img_file)
    
    print(f"Found {len(image_files)} images in total")
    print(f"After filtering out debug images: {len(filtered_files)} images to process")
    
    image_files = filtered_files
    
    if not image_files:
        print("No images found!")
        return
    
    # If sample is specified, take a subset of images
    if sample > 0 and sample < len(image_files):
        print(f"Processing a sample of {sample} images")
        np.random.shuffle(image_files)
        image_files = image_files[:sample]
    
    all_features = []
    all_labels = []
    all_paths = []
    all_confidence = []  # Add confidence metrics
    
    # Counters for statistics
    total_processed = 0
    successful_extractions = 0
    failed_extractions = 0
    partial_extractions = 0
    complete_extractions = 0
    
    # Process each image
    for image_file in tqdm(image_files, desc="Processing images"):
        total_processed += 1
        try:
            # Extract class label from directory name
            class_label = os.path.basename(os.path.dirname(image_file)).lower()
            
            # Extract landmarks with output_dir for debug images
            result = extract_landmarks_from_image(image_file, debug, output_dir)
            
            if result is not None:
                landmarks, metadata = result
                successful_extractions += 1
                
                # Check if we have all 21 landmarks properly detected (not just placeholders)
                is_complete = metadata["total_detected_points"] >= 15  # At least 15 real points
                
                if is_complete:
                    complete_extractions += 1
                else:
                    partial_extractions += 1
                
                if landmarks and len(landmarks) == 21:  # Ensure we have all 21 landmarks
                    # Only save data if we have a complete set or close to it
                    if is_complete:
                        # Calculate features with metadata
                        features = calculate_features(landmarks, metadata)
                        
                        # Determine confidence based on metadata
                        confidence = metadata["total_detected_points"] / 21.0
                        
                        # Store features
                        if output_format == 'npy':
                            # Create output directory for this class
                            class_output_dir = os.path.join(output_dir, class_label)
                            os.makedirs(class_output_dir, exist_ok=True)
                            
                            # Save as NPY file
                            base_name = os.path.splitext(os.path.basename(image_file))[0]
                            output_file = os.path.join(class_output_dir, f"{base_name}.npy")
                            np.save(output_file, features)
                        else:
                            # Collect for CSV
                            all_features.append(features)
                            all_labels.append(class_label)
                            all_paths.append(image_file)
                            all_confidence.append(confidence)
            else:
                failed_extractions += 1
                if debug:
                    print(f"Failed to extract valid landmarks from {image_file}")
        except Exception as e:
            failed_extractions += 1
            print(f"Error processing {image_file}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Print extraction statistics
    print("\nExtraction Statistics:")
    print(f"  Total images processed: {total_processed}")
    print(f"  Successful extractions: {successful_extractions} ({successful_extractions/total_processed*100:.1f}%)")
    print(f"  Complete extractions (15+ landmarks): {complete_extractions} ({complete_extractions/total_processed*100:.1f}%)")
    print(f"  Partial extractions (<15 landmarks): {partial_extractions} ({partial_extractions/total_processed*100:.1f}%)")
    print(f"  Failed extractions: {failed_extractions} ({failed_extractions/total_processed*100:.1f}%)")
    
    # Save results to CSV if needed
    if output_format == 'csv' and all_features:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create DataFrame
        if len(all_features) > 0:
            # Get feature count from first sample
            feature_count = len(all_features[0])
            feature_columns = [f'f{i}' for i in range(feature_count)]
            
            df = pd.DataFrame(all_features, columns=feature_columns)
            df['label'] = all_labels
            df['confidence'] = all_confidence
            df['file_path'] = all_paths
            
            # Save to CSV
            output_file = os.path.join(output_dir, "mediapipe_features.csv")
            df.to_csv(output_file, index=False)
            print(f"Saved features for {len(all_features)} complete landmark extractions to {output_file}")
            
            # Save a small sample of the data for inspection
            sample_size = min(10, len(all_features))
            sample_df = df.sample(sample_size) if len(df) > sample_size else df
            sample_file = os.path.join(output_dir, "sample_features.csv")
            sample_df.to_csv(sample_file, index=False)
            print(f"Saved sample of {sample_size} features to {sample_file}")
            
            # Get statistics on labels
            label_counts = df['label'].value_counts()
            print("\nLabel distribution:")
            for label, count in label_counts.items():
                print(f"  {label}: {count}")
                
            # Get statistics on confidence
            print(f"\nConfidence statistics:")
            print(f"  Mean: {df['confidence'].mean():.2f}")
            print(f"  Min: {df['confidence'].min():.2f}")
            print(f"  Max: {df['confidence'].max():.2f}")
            
            # Save confidence statistics per label
            stats_file = os.path.join(output_dir, "confidence_stats.csv")
            confidence_stats = df.groupby('label')['confidence'].agg(['mean', 'min', 'max', 'count'])
            confidence_stats.to_csv(stats_file)
            print(f"Saved confidence statistics to {stats_file}")
        else:
            print("No valid features extracted from images")

def main():
    args = parse_args()
    
    # Process images
    process_images(args.input_dir, args.output_dir, args.format, args.debug, args.sample)
    
    print("Processing complete!")

if __name__ == "__main__":
    main() 
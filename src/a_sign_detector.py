import cv2
import mediapipe as mp
import numpy as np
import time
import sys

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize MediaPipe Hands with better detection settings
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Allow detection of both hands
    min_detection_confidence=0.7,  # Higher confidence threshold
    min_tracking_confidence=0.5
)

# Global variable to store which hand to track
target_hand = "right"  # Default to right hand

def check_a_sign(landmarks, is_right_hand):
    """Check if hand is making ASL 'A' sign and provide feedback."""
    # Get key landmarks
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]  # Interphalangeal joint
    index_tip = landmarks[8]
    index_base = landmarks[5]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    wrist = landmarks[0]
    index_mcp = landmarks[5]  # Metacarpophalangeal joint (where index connects to hand)
    
    # Check if fingers are closed (making a fist)
    index_closed = index_tip.y > index_base.y
    middle_closed = middle_tip.y > landmarks[9].y
    ring_closed = ring_tip.y > landmarks[13].y
    pinky_closed = pinky_tip.y > landmarks[17].y
    fingers_closed = index_closed and middle_closed and ring_closed and pinky_closed
    
    # Calculate distance between thumb tip and side of index finger
    # For 3D distance: sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)
    if is_right_hand:
        # Thumb should be close to the side of the index finger (left side of hand)
        # For right hand, thumb x should be less than index_mcp x
        thumb_correct_side = thumb_tip.x < index_mcp.x
        
        # Calculate distance between thumb tip and index MCP joint
        distance = np.sqrt(
            (thumb_tip.x - index_mcp.x)**2 + 
            (thumb_tip.y - index_mcp.y)**2 +
            (thumb_tip.z - index_mcp.z)**2
        )
    else:
        # For left hand, thumb x should be greater than index_mcp x
        thumb_correct_side = thumb_tip.x > index_mcp.x
        
        # Calculate distance between thumb tip and index MCP joint
        distance = np.sqrt(
            (thumb_tip.x - index_mcp.x)**2 + 
            (thumb_tip.y - index_mcp.y)**2 +
            (thumb_tip.z - index_mcp.z)**2
        )
    
    # Threshold for considering thumb close enough to be "touching" the fist
    # This value may need tuning based on testing
    touch_threshold = 0.08
    thumb_touching = distance < touch_threshold
    
    # Thumb should be pointing up along the side of the fist
    thumb_pointing_up = thumb_tip.y < thumb_ip.y
    
    # Generate specific feedback
    if not fingers_closed:
        return False, "Close your fingers into a fist"
    elif not thumb_correct_side:
        side_text = "left" if is_right_hand else "right"
        return False, f"Position thumb on the {side_text} side of your fist"
    elif not thumb_touching:
        return False, "Thumb should touch the side of your fist"
    elif not thumb_pointing_up:
        return False, "Thumb should point upward"
    else:
        return True, "Perfect! That's the 'A' sign"

def try_all_cameras():
    """Try to find a working camera by testing indices 0-10."""
    for i in range(10):  # Try indices 0 through 9
        print(f"Trying camera index {i}...")
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Try to read a frame to make sure it's actually working
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                print(f"Successfully opened camera {i}")
                return cap
            else:
                print(f"Camera {i} opened but couldn't read frames")
                cap.release()
        else:
            print(f"Failed to open camera {i}")
    
    return None

def get_hand_type(hand_landmarks, results):
    """Determine if the hand is left or right based on MediaPipe classification."""
    # Get handedness classification
    handedness = results.multi_handedness
    for idx, classification in enumerate(handedness):
        if idx == results.multi_hand_landmarks.index(hand_landmarks):
            # Return True if right hand, False if left hand
            return classification.classification[0].label == "Right"
    return True  # Default to right if can't determine

def toggle_target_hand():
    """Toggle between left and right hand."""
    global target_hand
    if target_hand == "right":
        target_hand = "left"
    else:
        target_hand = "right"
    return target_hand

# Main function to process webcam feed
def main():
    global target_hand
    
    # First try the default camera
    print("Attempting to open camera...")
    cap = cv2.VideoCapture(0)
    
    # Test if default camera works by reading a frame
    if cap.isOpened():
        ret, test_frame = cap.read()
        if not ret or test_frame is None or test_frame.size == 0:
            print("Default camera opened but couldn't read frames, trying alternatives...")
            cap.release()
            cap = None
    else:
        print("Failed to open default camera, trying alternatives...")
        cap = None
    
    # If default didn't work, try scanning for a working camera
    if cap is None:
        cap = try_all_cameras()
    
    # If still no working camera, exit
    if cap is None:
        print("Error: Could not find any working camera.")
        print("Please check your camera connection and permissions.")
        print("For macOS: System Preferences > Security & Privacy > Camera")
        return
    
    # Add a small delay to let the camera initialize properly
    print("Camera opened successfully. Starting video stream...")
    print("Press 'h' to toggle between left and right hand detection")
    time.sleep(2)
    
    frame_count = 0
    consecutive_failures = 0
    
    while cap.isOpened():
        success, image = cap.read()
        
        # Handle frame grab failures with more tolerance
        if not success or image is None or image.size == 0:
            consecutive_failures += 1
            print(f"Failed to grab frame ({consecutive_failures})")
            
            # If we've had too many failures in a row, exit
            if consecutive_failures > 10:
                print("Too many consecutive frame capture failures. Exiting.")
                break
                
            # Wait a bit and try again
            time.sleep(0.1)
            continue
        
        # Reset failure counter on success
        consecutive_failures = 0
        frame_count += 1
            
        # Flip the image horizontally for a more natural selfie-view
        image = cv2.flip(image, 1)
        
        # Convert to RGB and process with MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        # Draw hand information text
        cv2.putText(image, f"Target Hand: {target_hand.upper()}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw landmarks and check 'A' sign
        found_target_hand = False
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Determine if this is a left or right hand
                is_right_hand = get_hand_type(hand_landmarks, results)
                hand_type = "right" if is_right_hand else "left"
                
                # Different colors for each hand
                color = (0, 255, 0) if is_right_hand else (255, 0, 0)
                
                # Draw landmarks with custom styling
                mp_drawing.draw_landmarks(
                    image, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Only evaluate the target hand type
                if hand_type == target_hand:
                    found_target_hand = True
                    
                    # Convert landmarks to array for our function
                    landmarks = hand_landmarks.landmark
                    
                    # Check if 'A' sign and get feedback
                    is_a_sign, feedback = check_a_sign(landmarks, is_right_hand)
                    
                    # Display result
                    result_color = (0, 255, 0) if is_a_sign else (0, 0, 255)
                    cv2.putText(image, feedback, (10, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, result_color, 2)
                    
                    # Display indicator
                    indicator_text = "✓ Correct!" if is_a_sign else "✗ Try Again"
                    cv2.putText(image, indicator_text, (10, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, result_color, 2)
                
                # Display which hand was detected
                hand_pos_x = int(hand_landmarks.landmark[0].x * image.shape[1])
                hand_pos_y = int(hand_landmarks.landmark[0].y * image.shape[0])
                cv2.putText(image, hand_type.upper(), (hand_pos_x - 20, hand_pos_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        if not found_target_hand:
            # Target hand not detected
            cv2.putText(image, f"No {target_hand} hand detected", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Display target sign
        cv2.putText(image, "Target: 'A' sign", (image.shape[1] - 200, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add help text
        cv2.putText(image, "Press 'h': Toggle hand | 'q' or ESC: Quit", 
                    (10, image.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Show the image
        cv2.imshow('ASL Sign Detector', image)
        
        # Print occasional status (every 30 frames)
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")
        
        # Handle key presses
        key = cv2.waitKey(5) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or q key
            break
        elif key == ord('h'):  # h key to toggle hand
            new_hand = toggle_target_hand()
            print(f"Switched to {new_hand} hand detection")
    
    # Clean up
    print("Releasing camera and closing windows...")
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {e}")
        sys.exit(1)
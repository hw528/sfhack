from flask import Flask, render_template_string, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
import os
import time
import joblib
import glob

app = Flask(__name__)

# ---------------------------
# Utility Functions
# ---------------------------
def preprocess_image(image):
    """Apply preprocessing to improve hand detection."""
    image = cv2.resize(image, (640, 480))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    return filtered

def extract_features(landmarks, is_right_hand):
    """Extract features from hand landmarks for the model."""
    features = []
    features.append(1.0 if is_right_hand else 0.0)
    for landmark in landmarks:
        features.extend([landmark.x, landmark.y, landmark.z])
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    wrist = landmarks[0]
    thumb_index_dist = np.sqrt(
        (thumb_tip.x - index_tip.x)**2 +
        (thumb_tip.y - index_tip.y)**2 +
        (thumb_tip.z - index_tip.z)**2
    )
    features.append(thumb_index_dist)
    thumb_side_dist = np.sqrt(
        (thumb_tip.x - landmarks[17].x)**2 +
        (thumb_tip.y - landmarks[17].y)**2 +
        (thumb_tip.z - landmarks[17].z)**2
    )
    features.append(thumb_side_dist)
    index_closed = 1 if index_tip.y > landmarks[5].y else 0
    middle_closed = 1 if middle_tip.y > landmarks[9].y else 0
    ring_closed = 1 if ring_tip.y > landmarks[13].y else 0
    pinky_closed = 1 if pinky_tip.y > landmarks[17].y else 0
    features.extend([index_closed, middle_closed, ring_closed, pinky_closed])
    for i in [4, 8, 12, 16, 20]:
        features.append(landmarks[i].x - wrist.x)
        features.append(landmarks[i].y - wrist.y)
        features.append(landmarks[i].z - wrist.z)
    return features

def get_asl_position_feedback(letter, landmarks):
    """Analyze landmarks to provide correction feedback for specific ASL signs."""
    feedback = {"correct": False, "message": "", "finger_issues": []}
    if letter == 'a':
        if landmarks[8].y < landmarks[5].y:
            feedback["finger_issues"].append("Close your index finger into a fist")
        if landmarks[12].y < landmarks[9].y:
            feedback["finger_issues"].append("Close your middle finger into a fist")
        if landmarks[16].y < landmarks[13].y:
            feedback["finger_issues"].append("Close your ring finger into a fist")
        if landmarks[20].y < landmarks[17].y:
            feedback["finger_issues"].append("Close your pinky finger into a fist")
        thumb_tip = landmarks[4]
        index_mcp = landmarks[5]
        thumb_distance = np.sqrt(
            (thumb_tip.x - index_mcp.x)**2 +
            (thumb_tip.y - index_mcp.y)**2 +
            (thumb_tip.z - index_mcp.z)**2
        )
        if thumb_tip.x < index_mcp.x:
            feedback["finger_issues"].append("Keep your thumb alongside your fingers, not across them")
        elif thumb_distance > 0.08:
            feedback["finger_issues"].append("Thumb should touch the side of your index finger")
        if not feedback["finger_issues"]:
            feedback["correct"] = True
            feedback["message"] = "Perfect 'A' sign!"
        else:
            feedback["message"] = "Adjust your 'A' sign:"
    elif letter == 'b':
        if landmarks[8].y > landmarks[5].y:
            feedback["finger_issues"].append("Extend your index finger upward")
        if landmarks[12].y > landmarks[9].y:
            feedback["finger_issues"].append("Extend your middle finger upward")
        if landmarks[16].y > landmarks[13].y:
            feedback["finger_issues"].append("Extend your ring finger upward")
        if landmarks[20].y > landmarks[17].y:
            feedback["finger_issues"].append("Extend your pinky finger upward")
        if abs(landmarks[8].x - landmarks[12].x) > 0.1:
            feedback["finger_issues"].append("Keep your fingers together")
        thumb_tip = landmarks[4]
        palm_center_x = (landmarks[0].x + landmarks[9].x) / 2
        if thumb_tip.x > palm_center_x:
            feedback["finger_issues"].append("Tuck your thumb against your palm")
        if not feedback["finger_issues"]:
            feedback["correct"] = True
            feedback["message"] = "Perfect 'B' sign!"
        else:
            feedback["message"] = "Adjust your 'B' sign:"
    return feedback

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
    return instructions.get(letter.lower(), ["No instructions available for this letter"])

def draw_semitransparent_rect(image, start_point, end_point, color, alpha=0.7):
    """Draw a semi-transparent rectangle on the image."""
    overlay = image.copy()
    cv2.rectangle(overlay, start_point, end_point, color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    return image

# ---------------------------
# ASLDetector Class
# ---------------------------
class ASLDetector:
    def __init__(self):
        # Initialize MediaPipe Hands and drawing utilities
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # Model paths and settings
        self.models_dir = "./webcam_models"
        self.model = None
        self.model_classes = []
        # Target letter and hand settings for classification
        self.target_letter = 'a'
        self.target_hand = "right"
        self.confidence_threshold = 0.98
        self.load_model()
    
    def load_model(self):
        """Load the combined ASL model."""
        combined_model_paths = [
            os.path.join(self.models_dir, "asl_combined_model.joblib"),
            os.path.join(self.models_dir, "combined_model.joblib")
        ]
        combined_model_path = None
        for path in combined_model_paths:
            if os.path.exists(path):
                combined_model_path = path
                break
        if combined_model_path:
            try:
                self.model = joblib.load(combined_model_path)
                print(f"Successfully loaded model from {combined_model_path}")
                if hasattr(self.model, 'classes_'):
                    self.model_classes = self.model.classes_
                    print(f"Model can predict: {', '.join(self.model_classes)}")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print(f"No model found in {self.models_dir}")
        return False

    def extract_features(self, landmarks, is_right_hand=True):
        """Extract features from hand landmarks for the model."""
        return extract_features(landmarks, is_right_hand)
    
    def get_hand_type(self, hand_landmarks, results):
        """Determine if the hand is left or right."""
        handedness = results.multi_handedness
        for idx, classification in enumerate(handedness):
            if idx == results.multi_hand_landmarks.index(hand_landmarks):
                return classification.classification[0].label == "Right"
        return True  # Default to right if undetermined

    def toggle_target_hand(self):
        """Toggle between left and right hand."""
        self.target_hand = "left" if self.target_hand == "right" else "right"
        return self.target_hand

    def toggle_target_letter(self):
        """Toggle between letters A and B."""
        self.target_letter = "b" if self.target_letter == "a" else "a"
        return self.target_letter

    def setup_camera(self):
        """Set up and return a camera capture object."""
        print("Setting up camera...")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if not ret or test_frame is None or test_frame.size == 0:
                print("Camera opened but couldn't read frames, trying alternatives...")
                cap.release()
                cap = None
        else:
            print("Failed to open camera, trying alternatives...")
            cap = None
        if cap is None:
            for i in range(1, 5):
                print(f"Trying camera index {i}...")
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        print(f"Successfully opened camera at index {i}")
                        time.sleep(1)
                        return cap
                    cap.release()
            print("Failed to find a working camera")
            return None
        time.sleep(1)
        return cap

    def get_position_feedback(self, letter, landmarks):
        return get_asl_position_feedback(letter, landmarks)
    
    def get_letter_instructions(self, letter):
        return get_asl_instructions(letter)
    
# Create a global ASLDetector instance
detector = ASLDetector()

# ---------------------------
# Frame Processing Function for Web
# ---------------------------
def process_frame(frame):
    """
    Flip the frame (mirror view), process it using MediaPipe Hands,
    perform classification using the loaded model, overlay landmarks,
    feedback, and instructions, and return the processed frame.
    """
    # Mirror the frame
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            detector.mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                detector.mp_hands.HAND_CONNECTIONS,
                detector.mp_drawing_styles.get_default_hand_landmarks_style(),
                detector.mp_drawing_styles.get_default_hand_connections_style()
            )
            # Determine hand type and process if it matches the target
            is_right_hand = detector.get_hand_type(hand_landmarks, results)
            hand_type = "right" if is_right_hand else "left"
            if hand_type == detector.target_hand or detector.target_hand == "both":
                # Extract features and run prediction if a model is loaded
                features = detector.extract_features(hand_landmarks.landmark, is_right_hand)
                if detector.model is not None:
                    prediction = detector.model.predict([features])[0]
                    probabilities = detector.model.predict_proba([features])[0]
                    letter_index = np.where(detector.model_classes == detector.target_letter)[0][0]
                    target_probability = probabilities[letter_index]
                    feedback_dict = detector.get_position_feedback(detector.target_letter, hand_landmarks.landmark)
                    correct_position = feedback_dict.get("correct", False)

                    if (prediction == detector.target_letter and target_probability >= detector.confidence_threshold and correct_position) or target_probability >= 0.99:
                        result_text = f"{detector.target_letter.upper()} sign detected! (Confidence: {target_probability:.2f})"
                        color = (0, 255, 0)
                        feedback = ""
                    else:
                        if target_probability < 0.5:
                            result_text = f"Not a {detector.target_letter.upper()} sign (Confidence: {target_probability:.2f})"
                            color = (0, 0, 255)
                        else:
                            result_text = f"Almost a {detector.target_letter.upper()} sign (Confidence: {target_probability:.2f})"
                            color = (0, 165, 255)
                    cv2.putText(frame, result_text, (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    # Draw confidence bar
                    confidence_bar_width = int(300 * target_probability)
                    if (prediction == detector.target_letter and target_probability >= detector.confidence_threshold and correct_position) or target_probability >= 0.99:
                        confidence_color = (0, 255, 0)
                    elif target_probability < 0.5:
                        confidence_color = (0, 0, 255)
                    else:
                        confidence_color = (0, 165, 255)
                    cv2.rectangle(frame, (10, 185), (310, 200), (100, 100, 100), -1)
                    cv2.rectangle(frame, (10, 185), (10 + confidence_bar_width, 200), confidence_color, -1)
                    threshold_x = int(10 + 300 * detector.confidence_threshold)
                    cv2.line(frame, (threshold_x, 180), (threshold_x, 205), (255, 255, 255), 2)
                    # Display instructions
                    instructions = detector.get_letter_instructions(detector.target_letter)
                    for i, line in enumerate(instructions):
                        cv2.putText(frame, f"{i+1}. {line}", (20, 220 + i*25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                else:
                    cv2.putText(frame, "No model loaded", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Hand detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No hand detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return frame

# ---------------------------
# MJPEG Stream Generator
# ---------------------------
def gen_frames():
    cap = detector.setup_camera()  # Open the camera
    if cap is None:
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

# ---------------------------
# Flask Endpoints
# ---------------------------
@app.route('/')
def index():
    html = '''
    <!doctype html>
    <html>
    <head>
      <title>ASL Detection Real-Time Video</title>
    </head>
    <body>
      <h1>ASL Detection Real-Time Video</h1>
      <img src="/video_feed" width="640" height="480">
    </body>
    </html>
    '''
    return render_template_string(html)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------------------------
# Run the Flask App
# ---------------------------
if __name__ == '__main__':
    app.run(debug=True)

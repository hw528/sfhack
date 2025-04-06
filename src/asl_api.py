from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os
import time
import argparse
from asl_detector import ASLDetector

app = Flask(__name__)
# Enable CORS for all routes
CORS(app)

# Global detector instance
detector = None

@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        'status': 'ok',
        'model_loaded': detector.model is not None,
        'available_letters': list(detector.model_classes) if detector.model is not None else [],
        'confidence_threshold': detector.confidence_threshold if detector.model is not None else 0.7
    })

@app.route('/api/detect', methods=['POST'])
def detect_sign():
    """
    Endpoint to process an image and detect ASL signs.
    
    Expected JSON format:
    {
        "image": "base64_encoded_image_data",
        "target_letter": "a"  # Optional, defaults to current target letter
    }
    
    Returns:
    {
        "target_letter": "a",
        "detected_sign": "a" or null,
        "confidence": 0.95,
        "position_feedback": "Good form! Keep it up",
        "hand_type": "right" or "left",
        "is_correct_position": true/false
    }
    """
    # Check if model is loaded
    if detector.model is None:
        return jsonify({
            'error': 'No model loaded',
            'status': 'error'
        }), 500
    
    try:
        # Get request data
        data = request.json
        
        # Check if target letter is provided
        if 'target_letter' in data and data['target_letter']:
            target_letter = data['target_letter'].lower()
            # Validate the target letter is in the model's classes
            if target_letter in detector.model_classes:
                detector.target_letter = target_letter
        
        # Decode image
        if 'image' not in data or not data['image']:
            return jsonify({
                'error': 'No image provided',
                'status': 'error'
            }), 400
        
        # Decode base64 image
        try:
            image_data = base64.b64decode(data['image'])
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None or image.size == 0:
                return jsonify({
                    'error': 'Invalid image data',
                    'status': 'error'
                }), 400
                
            # Apply pre-processing to improve hand detection
            # 1. Resize if too large (to speed up processing)
            max_width = 1280
            if image.shape[1] > max_width:
                scale_factor = max_width / image.shape[1]
                new_height = int(image.shape[0] * scale_factor)
                image = cv2.resize(image, (max_width, new_height))
                
            # 2. Apply contrast enhancement to improve feature visibility
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l_channel)
            enhanced_lab = cv2.merge((cl, a, b))
            image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
        except Exception as e:
            return jsonify({
                'error': f'Error decoding image: {str(e)}',
                'status': 'error'
            }), 400
        
        # Process the image for hand detection
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = detector.hands.process(image_rgb)
        
        # Prepare response
        response = {
            'target_letter': detector.target_letter,
            'detected_hands': [],
            'status': 'success',
            'confidence_threshold': detector.confidence_threshold
        }
        
        # Process detected hands
        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                # Determine hand type
                is_right_hand = detector.get_hand_type(hand_landmarks, results)
                hand_type = "right" if is_right_hand else "left"
                
                # Extract features
                features = detector.extract_features(hand_landmarks.landmark, is_right_hand)
                
                # Get position feedback
                position_feedback, is_correct_position = detector.get_position_feedback(
                    detector.target_letter, hand_landmarks.landmark
                )
                
                # Make prediction
                features_array = np.array(features).reshape(1, -1)
                prediction = detector.model.predict(features_array)[0]
                probabilities = detector.model.predict_proba(features_array)[0]
                
                # Find probability for target letter
                try:
                    letter_indices = np.where(detector.model_classes == detector.target_letter)[0]
                    if len(letter_indices) > 0:
                        letter_index = letter_indices[0]
                        confidence = float(probabilities[letter_index])
                    else:
                        # Letter not in model, use highest probability
                        letter_index = np.argmax(probabilities)
                        confidence = float(probabilities[letter_index])
                except Exception as e:
                    # Fallback if any issue with finding the letter index
                    print(f"Error finding letter index: {e}")
                    letter_index = np.argmax(probabilities)
                    confidence = float(probabilities[letter_index])
                
                # Determine if sign is detected
                sign_detected = confidence >= detector.confidence_threshold
                
                # Sync feedback with confidence value
                if sign_detected:
                    # High confidence, use positive feedback
                    is_correct_position = True
                    position_feedback = "Correct sign detected!"
                else:
                    # Low confidence, ensure feedback isn't overly positive
                    # If position feedback says "Good form" but confidence is too low, update it
                    if confidence < 0.4 and ("Good form" in position_feedback or "Perfect" in position_feedback):
                        is_correct_position = False
                        position_feedback = f"Keep adjusting your hand position. Confidence too low: {int(confidence*100)}%"
                    elif confidence < detector.confidence_threshold and ("Good form" in position_feedback or "Perfect" in position_feedback):
                        position_feedback = f"Almost there! {position_feedback} Increase confidence: {int(confidence*100)}%"
                
                # Add hand detection result to response
                hand_result = {
                    'hand_type': hand_type,
                    'confidence': confidence,
                    'position_feedback': position_feedback,
                    'is_correct_position': is_correct_position,
                    'detected_sign': detector.target_letter if sign_detected else None,
                    'sign_detected': sign_detected,
                    'landmark_positions': [
                        {'x': landmark.x, 'y': landmark.y, 'z': landmark.z}
                        for landmark in hand_landmarks.landmark
                    ]
                }
                
                response['detected_hands'].append(hand_result)
        
        # Add letter instructions to response
        response['instructions'] = detector.get_letter_instructions(detector.target_letter)
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'error': f'Error processing request: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/api/toggle_letter', methods=['POST'])
def toggle_letter():
    """Toggle to the next target letter."""
    try:
        new_letter = detector.toggle_target_letter()
        return jsonify({
            'status': 'success',
            'target_letter': new_letter
        })
    except Exception as e:
        return jsonify({
            'error': f'Error toggling letter: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/api/set_letter', methods=['POST'])
def set_letter():
    """Set a specific target letter."""
    try:
        data = request.json
        if 'letter' not in data:
            return jsonify({
                'error': 'No letter provided',
                'status': 'error'
            }), 400
        
        target_letter = data['letter'].lower()
        
        # Validate the letter is in the model's classes
        if target_letter not in detector.model_classes:
            return jsonify({
                'error': f'Letter {target_letter} not in model classes',
                'status': 'error',
                'available_letters': list(detector.model_classes)
            }), 400
        
        detector.target_letter = target_letter
        
        return jsonify({
            'status': 'success',
            'target_letter': detector.target_letter
        })
    except Exception as e:
        return jsonify({
            'error': f'Error setting letter: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/api/available_letters', methods=['GET'])
def get_available_letters():
    """Get the list of letters available in the model."""
    if detector.model is None:
        return jsonify({
            'error': 'No model loaded',
            'status': 'error'
        }), 500
    
    return jsonify({
        'status': 'success',
        'available_letters': list(detector.model_classes)
    })

def parse_args():
    parser = argparse.ArgumentParser(description="ASL Detector API")
    parser.add_argument('--model_dir', type=str, default='./webcam_models',
                       help='Directory containing ASL detection models')
    parser.add_argument('--model_file', type=str, default=None,
                       help='Specific model file to use (if provided, overrides model_dir)')
    parser.add_argument('--port', type=int, default=5050,
                       help='Port to run the API server on')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to run the API server on')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # Initialize detector
    detector = ASLDetector()
    
    # Set model dir if provided
    if args.model_dir:
        detector.models_dir = args.model_dir
    
    # Load model
    if args.model_file and os.path.exists(args.model_file):
        import joblib
        try:
            detector.model = joblib.load(args.model_file)
            if hasattr(detector.model, 'classes_'):
                detector.model_classes = detector.model.classes_
                print(f"Model loaded successfully. Can predict: {', '.join(detector.model_classes)}")
        except Exception as e:
            print(f"Error loading model from {args.model_file}: {e}")
    else:
        success = detector.load_model()
        if not success:
            print("Failed to load model. API will return errors for detection requests.")
    
    # Run the app
    app.run(host=args.host, port=args.port, debug=False) 
# ASL Sign Detector API and Web Client

This project provides a REST API wrapper around the ASL Sign Detector and a web client for easy integration with web applications.

## Components

1. **ASL Detector** (`asl_detector.py`): The core detection engine that uses MediaPipe and machine learning models to detect ASL signs.
2. **ASL API** (`asl_api.py`): A Flask-based REST API that wraps the ASL Detector and provides endpoints for web integration.
3. **Web Client** (`web_client/index.html`): A simple web application that uses the API to provide a user-friendly interface.

## Setup and Installation

1. Make sure you have Python 3.8+ installed
2. Install the required packages:
   ```
   pip install flask flask-cors numpy opencv-python mediapipe scikit-learn joblib
   ```
3. Ensure you have trained models in the `webcam_models` directory

## Running the API

Run the API server with:

```bash
python asl_api.py
```

By default, the API runs on port 5050. You can change this with the `--port` option:

```bash
python asl_api.py --port 8080
```

If your models are in a different directory, specify it with `--model_dir`:

```bash
python asl_api.py --model_dir /path/to/models
```

Or use a specific model file:

```bash
python asl_api.py --model_file /path/to/asl_model.joblib
```

## API Endpoints

The API provides the following endpoints:

- `GET /api/health`: Check if the API is running and if a model is loaded
- `GET /api/available_letters`: Get the list of letters available in the model
- `POST /api/detect`: Process an image and detect ASL signs
- `POST /api/toggle_letter`: Toggle to the next target letter
- `POST /api/set_letter`: Set a specific target letter

## Using the Web Client

1. Start the API server as described above
2. Open the web client in a browser:
   ```
   open web_client/index.html
   ```
   
   Or serve it using a simple HTTP server:
   ```
   python -m http.server
   ```
   
   Then visit http://localhost:8000/web_client/index.html

3. Grant camera access when prompted
4. Click "Start Detection" to begin detecting ASL signs

## Web Client Features

- Real-time sign detection using your webcam
- Visual confidence bar with threshold marker
- Position feedback and instructions for forming signs correctly
- Letter selection buttons for quickly switching between letters
- Toggle button to cycle through available letters

## Troubleshooting

1. **Camera access issues**: Make sure your browser has permission to access your camera
2. **API connection error**: Check that the API server is running and the port matches the one in the web client
3. **No model loaded**: Ensure that you have valid ASL model files in the model directory
4. **CORS errors**: If testing locally, you may need to run the web client on the same port as the API or enable CORS in the API

## Customizing the Web Client

You can customize the web client by editing the HTML, CSS, and JavaScript in `web_client/index.html`. The main configuration options are at the top of the JavaScript section:

```javascript
// Configuration
const API_BASE_URL = 'http://localhost:5050/api';
const DETECTION_INTERVAL = 300; // milliseconds between detection requests
```

Change `API_BASE_URL` if you're running the API on a different host or port. 
# Sign Language Learning App

An interactive application that helps users learn sign language through real-time feedback on their gestures.

## Features

- Real-time hand tracking and gesture recognition
- Instant feedback on sign accuracy
- Clean and intuitive user interface
- Support for learning the ASL alphabet

## Setup

1. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run src/app.py
```

## Project Structure

- `src/app.py`: Main application file with the Streamlit interface
- `src/gesture_recognition.py`: Handles gesture detection and recognition
- `models/`: Directory for trained gesture recognition models (to be added)

## Development

To contribute to the project:

1. Fork the repository
2. Create a new branch for your feature
3. Make your changes
4. Submit a pull request

## License

[Add your license here] 
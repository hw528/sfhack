# SignLearn

An interactive American Sign Language (ASL) learning platform that helps users learn and practice ASL through real-time feedback using computer vision.

## Features

- **Learning Module**: Practice ASL letters with real-time webcam feedback
- **Quiz Game**: Test your knowledge by signing letters in a timed challenge
- **Test Section**: Multiple-choice tests to evaluate your ASL knowledge
- **Real-time Sign Detection**: Get immediate feedback on your hand signs

## Getting Started

### Prerequisites

- Node.js (v14 or higher)
- Python 3.8+ (for the sign detection API)
- Webcam access

### Installation

1. Clone this repository
```bash
git clone https://github.com/hw528/SignLearn.git
cd SignLearn
```

2. Install frontend dependencies
```bash
npm install
```

3. Install API dependencies (in a separate terminal)
```bash
cd api  # Navigate to the API directory
pip install -r requirements.txt
```

## Running the Application

### Step 1: Start the API server

```bash
python asl_api.py  # This will start the server on port 5050
```

### Step 2: Start the frontend

In a new terminal window:

```bash
npm run dev
```

The application should now be running at `http://localhost:5173` (or the port specified by Vite).

## Usage

1. Navigate to the Learning section to practice individual letters
2. Use the Quiz game to test your skills with timed challenges
3. Take the Test to evaluate your knowledge of ASL signs

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
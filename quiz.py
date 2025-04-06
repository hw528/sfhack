import cv2
import numpy as np
import pygame
import time
import os
from flask import Flask, render_template_string, Response, redirect, url_for
from flask_cors import CORS  # Add CORS support
from src.asl_detector import ASLDetector  # Fixed import path

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ============================
# クイズ設定
# ============================
QUIZ_QUESTION = "Question! What's the last name of the American president?"
CORRECT_ANSWER = "TRUMP"  # 内部で管理、表示しない
HOLD_TIME_REQUIRED = 3    # 3秒間連続でホールドで確定
GAME_DURATION_AFTER_COMPLETE = 10  # クイズ完了後10秒でゲーム終了
POINT_PER_LETTER = 20     # 1文字あたり20点

# クイズ進行状態のグローバル変数
current_letter_index = 0  # 0～4
letter_hold_start = None  # 現在ホールド開始時刻
quiz_complete = False     # 全文字入力完了フラグ
finish_time = None        # 完了時刻
user_answers = []         # ユーザーがホールドで入力した文字
answers_correct = True    # 一文字でも間違っていれば False

def reset_quiz():
    global current_letter_index, letter_hold_start, quiz_complete, finish_time, user_answers, answers_correct
    current_letter_index = 0
    letter_hold_start = None
    quiz_complete = False
    finish_time = None
    user_answers = []
    answers_correct = True

reset_quiz()

# ============================
# ASLDetector のインスタンス生成
# ============================
detector = ASLDetector()

# ============================
# フレーム処理関数
# ============================
# グローバル変数を追加（ファイルの先頭付近で）
last_detected_letter = None

def process_frame(frame):
    global current_letter_index, letter_hold_start, quiz_complete, finish_time, user_answers, answers_correct, last_detected_letter
    # ミラー表示
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.hands.process(image_rgb)
    
    detected_letter = None
    detected_conf = 0
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            is_right = detector.get_hand_type(hand_landmarks, results)
            features = detector.extract_features(hand_landmarks.landmark, is_right)
            try:
                pred = detector.model.predict([features])[0]
                probs = detector.model.predict_proba([features])[0]
                if hasattr(detector.model, 'classes_'):
                    classes = detector.model.classes_
                    idx = np.where(classes == pred)[0][0]
                    conf = probs[idx]
                else:
                    conf = 0
                if conf > detected_conf:
                    detected_conf = conf
                    detected_letter = pred.upper()
            except Exception as e:
                print("Detection error:", e)
        
        # 検出結果表示（左上）
        if detected_letter:
            cv2.putText(frame, f"Detected: {detected_letter} ({detected_conf:.2f})", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No sign detected", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 期待文字は内部管理（表示しない）
        if current_letter_index < len(CORRECT_ANSWER):
            expected_letter = CORRECT_ANSWER[current_letter_index]
        else:
            expected_letter = None
        
        # ここで、検出結果が前回と同じかチェック
        if detected_letter is not None:
            if last_detected_letter != detected_letter:
                # 文字が変わったらタイマーをリセット
                letter_hold_start = time.time()
                last_detected_letter = detected_letter
            # もし同じならホールド時間を計算
            hold_duration = time.time() - letter_hold_start
            cv2.putText(frame, f"Hold for {hold_duration:.1f}s", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if expected_letter and hold_duration >= HOLD_TIME_REQUIRED:
                user_letter = detected_letter
                user_answers.append(user_letter)
                if user_letter != expected_letter:
                    answers_correct = False
                current_letter_index += 1
                letter_hold_start = None
                last_detected_letter = None
                if current_letter_index >= len(CORRECT_ANSWER):
                    quiz_complete = True
                    finish_time = time.time()
        else:
            letter_hold_start = None
            last_detected_letter = None
    else:
        cv2.putText(frame, "No hand detected", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return frame

# ============================
# 回答パネル生成関数（独立ストリーム用）
# ============================
def create_answer_panel():
    panel_height = 480
    panel_width = 400  # 幅を拡大して5つのボックスを余裕持って表示
    panel = 255 * np.ones((panel_height, panel_width, 3), dtype=np.uint8)
    
    # 得点表示：正解した文字数×POINT_PER_LETTER
    correct_count = sum(1 for i, letter in enumerate(user_answers)
                        if i < len(CORRECT_ANSWER) and letter == CORRECT_ANSWER[i])
    score = correct_count * POINT_PER_LETTER
    cv2.putText(panel, f"Score: {score}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)
    
    total_boxes = len(CORRECT_ANSWER)  # 5
    box_width = 60
    box_height = 60
    gap = 10
    start_x = 20
    start_y = 80
    for i in range(total_boxes):
        rect_x = start_x + i * (box_width + gap)
        if i < len(user_answers):
            letter = user_answers[i]
            color = (0, 255, 0) if letter == CORRECT_ANSWER[i] else (0, 0, 255)
            cv2.rectangle(panel, (rect_x, start_y), (rect_x+box_width, start_y+box_height), color, -1)
            cv2.putText(panel, letter, (rect_x + 5, start_y + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            cv2.rectangle(panel, (rect_x, start_y), (rect_x+box_width, start_y+box_height), (200, 200, 200), 2)
    
    # 結果は下部の空白スペースに表示
    if quiz_complete:
        if answers_correct:
            line1 = "Amazing!"
            line2 = ""
        else:
            line1 = "Incorrect!"
            line2 = f"The answer is: {CORRECT_ANSWER}"
        cv2.putText(panel, line1, (10, start_y + box_height + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0) if answers_correct else (0, 0, 255), 2)
        if line2:
            cv2.putText(panel, line2, (10, start_y + box_height + 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    return panel

# ============================
# カメラ映像ストリーム生成（独立ストリーム）
# ============================
def gen_camera_stream():
    cap = detector.setup_camera()
    if cap is None:
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed = process_frame(frame)
        
        # クイズ完了後、10秒経過したら強制終了
        if quiz_complete and finish_time is not None:
            if time.time() - finish_time >= GAME_DURATION_AFTER_COMPLETE:
                cap.release()
                os._exit(0)
        
        ret, buffer = cv2.imencode('.jpg', processed)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

# ============================
# 回答パネルストリーム生成（独立ストリーム）
# ============================
def gen_panel_stream():
    while True:
        panel = create_answer_panel()
        
        if quiz_complete and finish_time is not None:
            if time.time() - finish_time >= GAME_DURATION_AFTER_COMPLETE:
                os._exit(0)
        
        ret, buffer = cv2.imencode('.jpg', panel)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        time.sleep(0.2)

# ============================
# Flask ルーティング
# ============================
@app.route('/')
def index():
    reset_quiz()
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>ASL Quiz Game</title>
        <style>
            body {
                font-family: sans-serif;
                background-color: #f0f0f0;
                margin: 0; padding: 20px;
            }
            h1 { color: #333; }
            .container { display: flex; flex-direction: row; align-items: flex-start; }
            .video-container, .panel-container {
                border: 4px solid #333;
                box-shadow: 2px 2px 8px rgba(0,0,0,0.3);
                margin-right: 20px;
            }
            .info { color: #555; margin-top: 20px; }
        </style>
    </head>
    <body>
        <h1>ASL Quiz Game</h1>
        <p>{{ question }}</p>
        <div class="container">
            <div class="video-container">
                <img src="{{ url_for('video_feed') }}" style="max-width: 640px;">
            </div>
            <div class="panel-container">
                <img src="{{ url_for('panel_feed') }}">
            </div>
        </div>
        <div class="info">
            <p>Sign the answer letter by letter. Hold your sign for {{ hold_time }} seconds to confirm each letter.</p>
            <p>After 5 letters, the result is shown and the game will end in 10 seconds.</p>
        </div>
    </body>
    </html>
    ''', question=QUIZ_QUESTION, hold_time=HOLD_TIME_REQUIRED)

@app.route('/video_feed')
def video_feed():
    return Response(gen_camera_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/panel_feed')
def panel_feed():
    return Response(gen_panel_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/result')
def result():
    if not quiz_complete:
        return "Incomplete."
    if answers_correct:
        return "Amazing!"
    else:
        return f"Incorrect! The answer is: {CORRECT_ANSWER}"

if __name__ == '__main__':
    pygame.init()
    app.run(debug=True, host='0.0.0.0', port=5000)

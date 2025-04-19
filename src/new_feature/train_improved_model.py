import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from datetime import datetime
from collections import Counter
import mediapipe as mp
import cv2
import time
from tqdm import tqdm

from improved_feature_extraction import extract_optimized_features, get_feature_names

def collect_training_data(data_dir, output_file=None, samples_per_class=100, camera_index=0):
    """
    Collect training data from the webcam for ASL signs.
    
    Args:
        data_dir: Directory to save the collected data
        output_file: Name of the output CSV file (default: asl_improved_features.csv)
        samples_per_class: Number of samples to collect per class
        camera_index: Camera index to use
        
    Returns:
        DataFrame of collected data
    """
    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    if output_file is None:
        timestamp = datetime.now().strftime('%m%d_%H%M%S')
        output_file = f"asl_improved_features_{timestamp}.csv"
    
    output_path = os.path.join(data_dir, output_file)
    
    # Initialize MediaPipe hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    # Initialize camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return None
    
    # Letters to collect (ASL alphabet excluding J and Z which require motion)
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 
               'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']
    
    # DataFrame to store collected data
    all_data = []
    feature_names = get_feature_names()
    
    # Collect data for each letter
    for letter in letters:
        print(f"\nCollecting data for letter '{letter.upper()}'")
        print(f"Press SPACE to start collecting {samples_per_class} samples")
        
        samples_collected = 0
        collecting = False
        
        while samples_collected < samples_per_class:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame")
                break
            
            # Flip frame for selfie view
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = hands.process(rgb_frame)
            
            # Draw instructions on frame
            cv2.putText(frame, f"Letter: {letter.upper()}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {samples_collected}/{samples_per_class}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if collecting:
                cv2.putText(frame, "COLLECTING...", (10, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Press SPACE to start", (10, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Draw hand landmarks if detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        mp.solutions.drawing_styles.get_default_hand_connections_style())
                    
                    # Collect samples if we're in collection mode
                    if collecting:
                        # Get hand type
                        handedness = results.multi_handedness[0]
                        is_right_hand = handedness.classification[0].label == "Right"
                        
                        # Extract features
                        features = extract_optimized_features(hand_landmarks.landmark, is_right_hand)
                        
                        # Create a row with all features and the label
                        row = {'label': letter, 'hand_type': 'right' if is_right_hand else 'left'}
                        
                        # Add all features to the row
                        for i, feature_name in enumerate(feature_names):
                            row[feature_name] = features[i]
                        
                        # Add row to data
                        all_data.append(row)
                        samples_collected += 1
                        
                        # Small delay between samples
                        time.sleep(0.1)
                        
                        # Stop collecting if we have enough samples
                        if samples_collected >= samples_per_class:
                            collecting = False
                            print(f"Collected {samples_per_class} samples for letter '{letter.upper()}'")
                            break
            
            # Show the frame
            cv2.imshow('Data Collection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("Data collection interrupted")
                break
            elif key == 32:  # SPACE
                collecting = True
                print(f"Collecting samples for letter '{letter.upper()}'...")
        
        if key == 27:  # If ESC was pressed, exit the entire collection
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    # Convert data to DataFrame
    if all_data:
        df = pd.DataFrame(all_data)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"\nData saved to {output_path}")
        
        # Show summary
        print(f"\nCollected {len(df)} samples")
        print(f"Classes: {df['label'].nunique()}")
        print(f"Labels: {', '.join(sorted(df['label'].unique()))}")
        
        return df
    else:
        print("No data collected")
        return None


def process_existing_dataset(dataset_path, output_path=None):
    """
    Process an existing dataset with raw landmarks to extract improved features.
    
    Args:
        dataset_path: Path to the dataset with raw landmarks
        output_path: Path to save processed dataset (default: derived from input)
        
    Returns:
        DataFrame of processed data
    """
    # Read the dataset
    try:
        df = pd.read_csv(dataset_path)
        print(f"Dataset loaded from {dataset_path}")
        print(f"Shape: {df.shape}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    if output_path is None:
        timestamp = datetime.now().strftime('%m%d_%H%M%S')
        file_name = os.path.splitext(os.path.basename(dataset_path))[0]
        output_path = f"{file_name}_improved_{timestamp}.csv"
    
    # Check if dataset has landmarks or features
    if 'landmark_0_x' in df.columns:
        print("Found landmark columns, extracting features...")
        
        # Create a map of landmark columns
        landmark_cols = {}
        for i in range(21):
            for coord in ['x', 'y', 'z']:
                col_name = f'landmark_{i}_{coord}'
                if col_name in df.columns:
                    landmark_cols[(i, coord)] = col_name
        
        if len(landmark_cols) < 21 * 3:
            print(f"Warning: Expected 63 landmark coordinates, found {len(landmark_cols)}")
        
        # Check if we have hand type
        has_hand_type = 'hand_type' in df.columns
        
        # Process each row
        feature_names = get_feature_names()
        all_data = []
        
        print("Processing landmarks...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            # Create landmark objects
            landmarks = []
            for i in range(21):
                x = row[landmark_cols.get((i, 'x'), 0)]
                y = row[landmark_cols.get((i, 'y'), 0)]
                z = row[landmark_cols.get((i, 'z'), 0)]
                landmarks.append(type('Landmark', (), {'x': x, 'y': y, 'z': z}))
            
            # Get hand type
            is_right_hand = True  # Default
            if has_hand_type:
                is_right_hand = row['hand_type'] == 'right'
            
            # Extract features
            features = extract_optimized_features(landmarks, is_right_hand)
            
            # Create a row with all features and the label
            new_row = {'label': row['label'], 'hand_type': 'right' if is_right_hand else 'left'}
            
            # Add all features to the row
            for i, feature_name in enumerate(feature_names):
                new_row[feature_name] = features[i]
            
            # Add row to data
            all_data.append(new_row)
        
        # Convert to DataFrame
        new_df = pd.DataFrame(all_data)
        
        # Save to CSV
        new_df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")
        
        return new_df
    else:
        print("Dataset doesn't contain landmark columns. Please provide raw landmark data.")
        return None


def train_and_evaluate_models(dataset_path=None):
    """
    Train and evaluate models using the improved features.
    
    Args:
        dataset_path: Path to the dataset with improved features
        
    Returns:
        Dictionary of trained models and results
    """
    # Create timestamp for file names
    timestamp = datetime.now().strftime('%m%d_%H%M%S')
    
    # Create directory for this run
    run_dir = f'/Users/wuhaodong/SFhack/models/improved_run_{timestamp}'
    os.makedirs(run_dir, exist_ok=True)
    
    # Load data
    if dataset_path is None:
        # Look for most recent dataset
        data_dir = '/Users/wuhaodong/SFhack'
        csv_files = [f for f in os.listdir(data_dir) if f.startswith('asl_improved_features_') and f.endswith('.csv')]
        
        if csv_files:
            # Get the most recent file
            latest_file = max(csv_files, key=lambda x: os.path.getmtime(os.path.join(data_dir, x)))
            dataset_path = os.path.join(data_dir, latest_file)
            print(f"Using most recent dataset: {dataset_path}")
        else:
            print("No dataset found. Please provide a dataset path.")
            return None
    
    try:
        df = pd.read_csv(dataset_path)
        print(f"Dataset loaded with shape: {df.shape}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    # Feature and target columns
    feature_columns = [col for col in df.columns if col not in ['label', 'hand_type']]
    X = df[feature_columns]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_columns)
    
    # Save the scaler
    scaler_path = f'{run_dir}/feature_scaler_{timestamp}.joblib'
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['label'])
    num_classes = len(np.unique(y))
    
    # Analyze class distribution
    class_distribution = Counter(y)
    print("\nClass Distribution:")
    for class_idx, count in class_distribution.items():
        print(f"Class {label_encoder.classes_[class_idx]}: {count} samples ({count/len(y)*100:.2f}%)")
    
    # Save the label encoder
    encoder_path = f'{run_dir}/label_encoder_{timestamp}.joblib'
    joblib.dump(label_encoder, encoder_path)
    print(f"Label encoder saved to: {encoder_path}")
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y  # Ensure balanced class distribution in splits
    )
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, 
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            bootstrap=True,
            class_weight='balanced'
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100, 
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            objective='multi:softprob',
            num_class=num_classes
        ),
        'SVM': SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            probability=True,
            random_state=42,
            class_weight='balanced'
        ),
        'Neural Network': MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            max_iter=2000,
            random_state=42,
            alpha=0.0001,
            learning_rate_init=0.001,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        )
    }
    
    # Train and evaluate each model
    results = []
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Save trained model
        trained_models[name] = model
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Stratified cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_scaled, y, cv=cv)
        
        # Store results
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'CV Mean Score': cv_scores.mean(),
            'CV Std': cv_scores.std()
        })
        
        # Save the model
        model_path = f'{run_dir}/{name.lower().replace(" ", "_")}_model_{timestamp}.joblib'
        joblib.dump(model, model_path)
        print(f"Model saved to: {model_path}")
        
        # Print model metrics
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test F1-Score: {f1:.4f}")
        print(f"CV Mean Score: {cv_scores.mean():.4f}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_path = f'{run_dir}/model_comparison_results_{timestamp}.csv'
    results_df.to_csv(results_path, index=False)
    
    # Print results
    print("\nModel Comparison Results:")
    print("-" * 80)
    print(results_df.to_string())
    
    # Find best model
    best_model_idx = results_df['Accuracy'].idxmax()
    best_model = results_df.loc[best_model_idx]
    best_model_name = best_model['Model']
    
    print(f"\nBest performing model: {best_model_name}")
    print(f"Accuracy: {best_model['Accuracy']:.4f}")
    print(f"F1-Score: {best_model['F1-Score']:.4f}")
    print(f"CV Mean Score: {best_model['CV Mean Score']:.4f}")
    
    # Save a copy of the best model with a simplified name for easier loading
    best_model_simple_path = f'{run_dir}/improved_model.joblib'
    best_encoder_simple_path = f'{run_dir}/improved_encoder.joblib'
    
    joblib.dump(trained_models[best_model_name], best_model_simple_path)
    joblib.dump(label_encoder, best_encoder_simple_path)
    
    print(f"\nBest model saved to: {best_model_simple_path}")
    print(f"Encoder saved to: {best_encoder_simple_path}")
    
    # Create a run info file with details
    run_info_path = f'{run_dir}/run_info_{timestamp}.txt'
    with open(run_info_path, 'w') as f:
        f.write(f"ASL Improved Model Training Run - {timestamp}\n")
        f.write("-" * 50 + "\n\n")
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Dataset shape: {df.shape}\n")
        f.write(f"Number of features: {len(feature_columns)}\n")
        f.write(f"Number of classes: {num_classes}\n\n")
        
        f.write("Class Distribution:\n")
        for class_idx, count in class_distribution.items():
            f.write(f"  Class {label_encoder.classes_[class_idx]}: {count} samples ({count/len(y)*100:.2f}%)\n")
        
        f.write("\nModel Results:\n")
        for _, row in results_df.iterrows():
            f.write(f"  {row['Model']}:\n")
            f.write(f"    Accuracy: {row['Accuracy']:.4f}\n")
            f.write(f"    F1-Score: {row['F1-Score']:.4f}\n")
            f.write(f"    CV Mean Score: {row['CV Mean Score']:.4f}\n")
            f.write(f"    CV Std: {row['CV Std']:.4f}\n")
        
        f.write(f"\nBest Model: {best_model_name}\n")
        f.write(f"  Test Accuracy: {best_model['Accuracy']:.4f}\n")
        f.write(f"  Test F1-Score: {best_model['F1-Score']:.4f}\n")
        f.write(f"  CV Mean Score: {best_model['CV Mean Score']:.4f}\n")
        f.write(f"  CV Std: {best_model['CV Std']:.4f}\n")
    
    print(f"Run details saved to: {run_info_path}")
    
    return {
        'models': trained_models,
        'results': results_df,
        'best_model': best_model_name,
        'label_encoder': label_encoder,
        'scaler': scaler,
        'run_dir': run_dir
    }


if __name__ == "__main__":
    print("\nASL Improved Model Training")
    print("=" * 30)
    print("Choose an option:")
    print("1. Collect new training data")
    print("2. Process existing dataset with raw landmarks")
    print("3. Train and evaluate models")
    print("4. Run full pipeline (collect, process, train)")
    
    option = input("\nEnter your choice (1-4): ")
    
    if option == '1':
        print("\nCollecting new training data...")
        data_dir = '/Users/wuhaodong/SFhack'
        collect_training_data(data_dir, samples_per_class=50)
    
    elif option == '2':
        print("\nProcessing existing dataset...")
        dataset_path = input("Enter the path to the dataset with raw landmarks: ")
        process_existing_dataset(dataset_path)
    
    elif option == '3':
        print("\nTraining and evaluating models...")
        dataset_path = input("Enter the path to the dataset with improved features (or press Enter to use most recent): ")
        if not dataset_path:
            dataset_path = None
        train_and_evaluate_models(dataset_path)
    
    elif option == '4':
        print("\nRunning full pipeline...")
        # Collect data
        print("\nStep 1: Collecting training data...")
        data_dir = '/Users/wuhaodong/SFhack'
        df = collect_training_data(data_dir, samples_per_class=50)
        
        if df is not None:
            # Train models directly on collected data
            print("\nStep 2: Training and evaluating models...")
            train_and_evaluate_models(None)  # Uses most recent dataset
    
    else:
        print("Invalid option. Please choose 1-4.") 
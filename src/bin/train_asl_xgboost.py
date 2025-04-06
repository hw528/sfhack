import os
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import argparse
import glob

def parse_args():
    parser = argparse.ArgumentParser(description='Train XGBoost model for ASL sign detection')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing training data')
    parser.add_argument('--output_dir', type=str, default='./webcam_models',
                        help='Directory to save the trained model')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Fraction of data to use for testing')
    parser.add_argument('--letters', type=str, default='all',
                        help='Comma-separated list of letters to include (e.g., "a,b,c" or "all" for all letters)')
    parser.add_argument('--tune_hyperparams', action='store_true',
                        help='Perform hyperparameter tuning')
    return parser.parse_args()

def load_data(data_dir, letters='all'):
    """
    Load MediaPipe landmark data from files.
    Expected format: Each file contains landmark features for one sample.
    """
    print(f"Loading data from {data_dir}...")
    
    # Determine which letters to include
    if letters == 'all':
        classes = [chr(i) for i in range(ord('a'), ord('z')+1)] + [str(i) for i in range(10)]
    else:
        classes = letters.split(',')
    
    print(f"Including classes: {classes}")
    
    # Check if the data is in CSV format
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    if csv_files:
        # Load the main CSV file containing all landmarks
        print("Found CSV files, loading data...")
        main_csv = os.path.join(data_dir, "mediapipe_features.csv")
        if os.path.exists(main_csv):
            # Load the dataframe
            df = pd.read_csv(main_csv)
            print(f"Loaded dataframe with shape: {df.shape}")
            
            # Filter to keep only requested classes
            if letters != 'all':
                df = df[df['label'].isin(classes)]
                print(f"Filtered to {len(df)} rows for requested classes")
            
            # Extract labels and confidence
            labels = df['label'].values
            confidence = None
            if 'confidence' in df.columns:
                confidence = df['confidence'].values
                print(f"Found confidence scores, ranging from {confidence.min():.2f} to {confidence.max():.2f}")
            
            # Check if all feature columns have the same pattern (f0, f1, ...)
            feature_cols = [col for col in df.columns if col.startswith('f')]
            print(f"Found {len(feature_cols)} feature columns")
            
            # Extract features array, excluding non-feature columns
            non_feature_cols = ['label', 'file_path', 'confidence']
            X = df.drop(columns=[col for col in non_feature_cols if col in df.columns]).values
            y = labels
            
            print(f"Final dataset shape: X={X.shape}, y={len(y)}")
            
            # Print class distribution
            class_counts = df['label'].value_counts()
            print("\nClass distribution:")
            for cls, count in class_counts.items():
                print(f"  {cls}: {count}")
            
            return X, y
        else:
            raise ValueError(f"Expected to find {main_csv} but it doesn't exist")
    else:
        # Try to load data from directories named by letter
        X_list = []
        y_list = []
        
        for cls in classes:
            class_dir = os.path.join(data_dir, cls)
            if os.path.exists(class_dir):
                files = glob.glob(os.path.join(class_dir, '*.npy'))
                print(f"Found {len(files)} samples for class {cls}")
                for file in files:
                    try:
                        # Load features from numpy file
                        features = np.load(file)
                        X_list.append(features)
                        y_list.append(cls)
                    except Exception as e:
                        print(f"Error loading {file}: {e}")
            else:
                print(f"No directory found for class {cls}")
    
        if not X_list:
            raise ValueError(f"No data found in {data_dir}")
        
        # Check for inconsistent feature lengths
        feature_lengths = [len(x) for x in X_list]
        if len(set(feature_lengths)) > 1:
            print(f"Warning: Inconsistent feature lengths detected - {set(feature_lengths)}")
            # Find the most common length to standardize on
            from collections import Counter
            most_common_length = Counter(feature_lengths).most_common(1)[0][0]
            print(f"Standardizing on {most_common_length} features")
            
            # Filter to keep only samples with the most common feature length
            valid_samples = []
            valid_labels = []
            for features, label in zip(X_list, y_list):
                if len(features) == most_common_length:
                    valid_samples.append(features)
                    valid_labels.append(label)
            
            X_list = valid_samples
            y_list = valid_labels
            print(f"Kept {len(X_list)} samples after filtering for consistent feature length")
        
        # Convert to numpy arrays
        X = np.array(X_list)
        y = np.array(y_list)
    
    print(f"Loaded {len(X)} samples with {X.shape[1]} features")
    unique_classes = np.unique(y)
    print(f"Classes represented: {sorted(unique_classes)}")
    
    return X, y

def train_xgboost_model(X_train, y_train, tune_hyperparams=False):
    """Train an XGBoost model with optional hyperparameter tuning."""
    
    # If letters are in strings, encode them as integers
    if isinstance(y_train[0], str):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(y_train)
        y_train_encoded = le.transform(y_train)
    else:
        le = None
        y_train_encoded = y_train
    
    if tune_hyperparams:
        print("Performing hyperparameter tuning...")
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.1, 0.01, 0.05],
            'n_estimators': [100, 200, 300],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        model = xgb.XGBClassifier(objective='multi:softprob', random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', verbose=1)
        grid_search.fit(X_train, y_train_encoded)
        
        print(f"Best parameters: {grid_search.best_params_}")
        model = grid_search.best_estimator_
    else:
        print("Training XGBoost model with default parameters...")
        # Default parameters
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X_train, y_train_encoded)
    
    return model, le

def evaluate_model(model, X_test, y_test, le=None):
    """Evaluate the trained model and generate performance metrics."""
    if le is not None:
        y_test_encoded = le.transform(y_test)
    else:
        y_test_encoded = y_test
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test_encoded, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Print detailed classification report
    print("\nClassification Report:")
    class_names = le.classes_ if le is not None else None
    print(classification_report(y_test_encoded, y_pred, target_names=class_names))
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test_encoded, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save confusion matrix plot
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to confusion_matrix.png")
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    xgb.plot_importance(model, max_num_features=20)
    plt.title('Feature Importance')
    plt.savefig('feature_importance.png')
    print("Feature importance plot saved to feature_importance.png")
    
    return accuracy

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load the data
    X, y = load_data(args.data_dir, args.letters)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    # Train the model
    model, le = train_xgboost_model(X_train, y_train, args.tune_hyperparams)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test, le)
    
    # Save the model
    model_file = os.path.join(args.output_dir, "asl_combined_model.joblib")
    joblib.dump(model, model_file)
    print(f"Model saved to {model_file}")
    
    # Save label encoder if it exists
    if le is not None:
        le_file = os.path.join(args.output_dir, "label_encoder.joblib")
        joblib.dump(le, le_file)
        print(f"Label encoder saved to {le_file}")
    
    print("Training and evaluation complete!")

if __name__ == "__main__":
    main() 
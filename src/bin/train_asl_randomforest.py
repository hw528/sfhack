#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import argparse
import glob
import json
from scipy.stats import randint

def parse_args():
    parser = argparse.ArgumentParser(description='Train Random Forest model for ASL sign detection')
    parser.add_argument('--data_dir', type=str, default='./data/processed',
                        help='Directory containing training data')
    parser.add_argument('--output_dir', type=str, default='./improved_models',
                        help='Directory to save the trained model')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Fraction of data to use for testing')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--tune', action='store_true',
                        help='Perform hyperparameter tuning')
    parser.add_argument('--n_iter', type=int, default=20,
                        help='Number of parameter settings for random search')
    return parser.parse_args()

def load_data(data_dir):
    """Load features from mediapipe_features.csv file."""
    print(f"Loading data from {data_dir}...")
    
    # Load the main CSV file containing all landmarks
    main_csv = os.path.join(data_dir, "mediapipe_features.csv")
    
    if not os.path.exists(main_csv):
        raise FileNotFoundError(f"Expected to find {main_csv} but it doesn't exist")
        
    # Load the dataframe
    df = pd.read_csv(main_csv)
    print(f"Loaded dataframe with shape: {df.shape}")
    
    # Extract labels and confidence
    labels = df['label'].values
    if 'confidence' in df.columns:
        confidence = df['confidence'].values
        print(f"Found confidence scores, ranging from {confidence.min():.2f} to {confidence.max():.2f}")
    
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

def train_random_forest(X_train, y_train, random_state=42, tune=False, n_iter=20):
    """Train a Random Forest model with optional hyperparameter tuning."""
    # Encode labels
    le = LabelEncoder()
    le.fit(y_train)
    y_train_encoded = le.transform(y_train)
    
    best_params = None
    
    if tune:
        print("Performing hyperparameter tuning...")
        
        # Define parameter space
        param_dist = {
            'n_estimators': randint(100, 500),
            'max_depth': randint(10, 40),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False],
            'class_weight': ['balanced', 'balanced_subsample', None]
        }
        
        # Create base model
        rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
        
        # Set up randomized search
        random_search = RandomizedSearchCV(
            rf, 
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring='accuracy',
            cv=3,
            verbose=2,
            random_state=random_state,
            n_jobs=-1
        )
        
        # Perform search
        print(f"Searching {n_iter} combinations with 3-fold cross-validation...")
        random_search.fit(X_train, y_train_encoded)
        
        # Get best parameters and model
        best_params = random_search.best_params_
        print(f"Best parameters: {best_params}")
        print(f"Best cross-validation accuracy: {random_search.best_score_:.4f}")
        
        # Train final model with best parameters
        print("Training final model with best parameters...")
        model = RandomForestClassifier(
            random_state=random_state,
            n_jobs=-1,
            **best_params
        )
    else:
        print("Training Random Forest with default parameters...")
        # Use good default parameters
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )
        best_params = {
            'n_estimators': 300,
            'max_depth': 30,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'class_weight': 'balanced'
        }
    
    # Train the model
    model.fit(X_train, y_train_encoded)
    
    return model, le, best_params

def evaluate_model(model, X_test, y_test, le, output_dir):
    """Evaluate the model and save detailed performance metrics."""
    y_test_encoded = le.transform(y_test)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test_encoded, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Generate classification report
    class_names = le.classes_
    report = classification_report(y_test_encoded, y_pred, 
                                  target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    # Save classification report
    report_file = os.path.join(output_dir, "rf_classification_report.csv")
    report_df.to_csv(report_file)
    print(f"Classification report saved to {report_file}")
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test_encoded, y_pred, target_names=class_names))
    
    # Plot and save confusion matrix
    plt.figure(figsize=(16, 14))
    cm = confusion_matrix(y_test_encoded, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Random Forest')
    cm_file = os.path.join(output_dir, "rf_confusion_matrix.png")
    plt.savefig(cm_file, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {cm_file}")
    
    # Plot and save feature importance
    plt.figure(figsize=(12, 10))
    feature_importances = model.feature_importances_
    indices = np.argsort(feature_importances)[-30:]  # Top 30 features
    
    plt.barh(range(len(indices)), feature_importances[indices])
    plt.yticks(range(len(indices)), [f'Feature {i}' for i in indices])
    plt.title('Random Forest Feature Importance')
    importance_file = os.path.join(output_dir, "rf_feature_importance.png")
    plt.savefig(importance_file, dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved to {importance_file}")
    
    # Calculate per-class accuracy
    correct_per_class = {}
    total_per_class = {}
    
    for true, pred in zip(y_test_encoded, y_pred):
        true_label = le.inverse_transform([true])[0]
        if true_label not in total_per_class:
            total_per_class[true_label] = 0
            correct_per_class[true_label] = 0
        
        total_per_class[true_label] += 1
        if true == pred:
            correct_per_class[true_label] += 1
    
    # Create per-class accuracy dataframe
    class_accuracy = {}
    for cls in total_per_class:
        class_accuracy[cls] = correct_per_class[cls] / total_per_class[cls]
    
    accuracy_df = pd.DataFrame({
        'class': list(class_accuracy.keys()),
        'accuracy': list(class_accuracy.values()),
        'samples': [total_per_class[cls] for cls in class_accuracy]
    })
    accuracy_df = accuracy_df.sort_values('accuracy', ascending=False)
    
    # Save per-class accuracy
    accuracy_file = os.path.join(output_dir, "rf_class_accuracy.csv")
    accuracy_df.to_csv(accuracy_file, index=False)
    print(f"Per-class accuracy saved to {accuracy_file}")
    
    # Plot per-class accuracy
    plt.figure(figsize=(14, 10))
    sns.barplot(x='class', y='accuracy', data=accuracy_df)
    plt.title('Accuracy by Class - Random Forest')
    plt.xticks(rotation=90)
    plt.tight_layout()
    accuracy_plot_file = os.path.join(output_dir, "rf_class_accuracy.png")
    plt.savefig(accuracy_plot_file, dpi=300, bbox_inches='tight')
    print(f"Per-class accuracy plot saved to {accuracy_plot_file}")
    
    return accuracy, report

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the data
    X, y = load_data(args.data_dir)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    # Train model
    model, le, best_params = train_random_forest(
        X_train, y_train, 
        random_state=args.random_state,
        tune=args.tune,
        n_iter=args.n_iter
    )
    
    # Evaluate model
    accuracy, report = evaluate_model(model, X_test, y_test, le, args.output_dir)
    
    # Save best parameters
    params_file = os.path.join(args.output_dir, "rf_best_params.json")
    with open(params_file, 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"Best parameters saved to {params_file}")
    
    # Save the model
    model_file = os.path.join(args.output_dir, "asl_randomforest.joblib")
    joblib.dump(model, model_file)
    print(f"Model saved to {model_file}")
    
    # Save label encoder
    le_file = os.path.join(args.output_dir, "rf_label_encoder.joblib")
    joblib.dump(le, le_file)
    print(f"Label encoder saved to {le_file}")
    
    # Save model metadata
    metadata = {
        'accuracy': float(accuracy),
        'n_classes': len(le.classes_),
        'classes': le.classes_.tolist(),
        'n_features': X.shape[1],
        'n_samples': len(X),
        'best_params': best_params
    }
    
    metadata_file = os.path.join(args.output_dir, "rf_model_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Model metadata saved to {metadata_file}")
    
    print("\nRandom Forest training and evaluation complete!")

if __name__ == "__main__":
    main() 
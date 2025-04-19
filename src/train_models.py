import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, learning_curve
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib
from datetime import datetime
import os
from collections import Counter
import json

def train_and_evaluate_models():
    # Create timestamp for file names (without year)
    timestamp = datetime.now().strftime('%m%d_%H%M%S')
    
    # Create directory for this run
    run_dir = f'/Users/wuhaodong/SFhack/models/new_run_{timestamp}'
    os.makedirs(run_dir, exist_ok=True)
    
    # Load data
    file_path = '/Users/wuhaodong/SFhack/asl_optimized_features_0418_145734.csv'
    df = pd.read_csv(file_path)
    
    # Filter to keep only positive samples
    if 'sample_type' in df.columns:
        positive_samples = df[df['sample_type'] == 'positive']
        print(f"Filtered dataset to use only positive samples: {len(positive_samples)} samples out of {len(df)} total")
        df = positive_samples
    else:
        print("Warning: No 'sample_type' column found in dataset. Using all samples.")
    
    # Prepare features and target
    feature_columns = [col for col in df.columns if col not in ['hand_type', 'label', 'sample_type']]
    X = df[feature_columns]
    
    # Add debugging information about the dataset
    print(f"Loading data from: {file_path}")
    print(f"Data shape: {df.shape}")
    print(f"Feature statistics:\n{X.describe()}")
        
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
    joblib.dump(label_encoder, f'{run_dir}/label_encoder_{timestamp}.joblib')
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y  # Ensure balanced class distribution in splits
    )
    
    # Define models with more aggressive anti-overfitting parameters
    models = {
        # 'Random Forest': RandomForestClassifier(
        #     n_estimators=100,  # Reduced from 200
        #     max_depth=8,  # Reduced from 10
        #     min_samples_split=15,  # Increased from 10
        #     min_samples_leaf=8,  # Increased from 5
        #     max_features='sqrt',
        #     random_state=42,
        #     bootstrap=True,
        #     class_weight='balanced'
        # ),
        'XGBoost': XGBClassifier(
            n_estimators=150,       # Increase from 100 (more trees for better performance)
            max_depth=5,            # Slight increase from 4 (allow more complex patterns)
            learning_rate=0.03,     # Lower from 0.05 (slower learning, better generalization)
            subsample=0.8,          # Increase from 0.7 (use more data per tree)
            colsample_bytree=0.8,   # Increase from 0.7 (use more features per tree)
            min_child_weight=6,     # Decrease from 8 (allow slightly more specific rules)
            gamma=0.1,              # Decrease from 0.2 (allow more tree splits)
            reg_alpha=0.3,          # Increase from 0.2 (stronger L1 regularization)
            reg_lambda=1.2,         # Decrease from 1.5 (reduce L2 regularization)
            random_state=42,
            objective='multi:softprob',
            eval_metric='mlogloss',
            num_class=num_classes
        )
        # 'Neural Network': MLPClassifier(
        #     hidden_layer_sizes=(32, 16, 8),  # Reduced from (64, 32, 16)
        #     max_iter=2000,
        #     random_state=42,
        #     alpha=0.02,  # Increased from 0.01
        #     learning_rate_init=0.0005,  # Reduced from 0.001
        #     early_stopping=True,
        #     validation_fraction=0.2,  # Increased from 0.1
        #     n_iter_no_change=15,  # Increased from 10
        #     batch_size=64  # Increased from 32
        # )
    }
    
    # Train and evaluate each model
    results = []
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Print training vs test accuracy
        print(f"\nTraining accuracy: {train_accuracy:.4f}")
        print(f"Test accuracy: {accuracy:.4f}")
        print(f"Gap (train-test): {train_accuracy - accuracy:.4f}")
        
        # Stratified cross-validation - make sure all classes are represented in each fold
        cv_scores = []
        if name == 'XGBoost':
            # For XGBoost, manually perform stratified CV to ensure all classes present in each fold
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for train_idx, test_idx in skf.split(X, y):
                # Make sure all classes are in the training set
                X_cv_train, X_cv_test = X.iloc[train_idx], X.iloc[test_idx]
                y_cv_train, y_cv_test = y[train_idx], y[test_idx]
                
                # Verify all classes are present
                if len(np.unique(y_cv_train)) == num_classes:
                    # Train and evaluate
                    model_cv = XGBClassifier(**{k: v for k, v in models['XGBoost'].get_params().items() 
                                              if k != 'xgb_model'})
                    model_cv.fit(X_cv_train, y_cv_train)
                    cv_pred = model_cv.predict(X_cv_test)
                    cv_scores.append(accuracy_score(y_cv_test, cv_pred))
            
            cv_mean = np.mean(cv_scores) if cv_scores else np.nan
            cv_std = np.std(cv_scores) if cv_scores else np.nan
            
            # For XGBoost, do a manual learning curve analysis that ensures class balance
            print("\nXGBoost Learning Curve Analysis:")
            train_sizes = np.linspace(0.1, 1.0, 10)
            train_sizes_abs = [int(train_size * len(X_train)) for train_size in train_sizes]
            train_scores = []
            test_scores = []
            
            for size in train_sizes_abs:
                # Sample with stratification to maintain class balance
                from sklearn.model_selection import StratifiedShuffleSplit
                # Make sure the train_size is a fraction less than 1.0
                sample_fraction = min(0.99, size/len(X_train))  # Cap at 0.99 to avoid ValueError
                sss = StratifiedShuffleSplit(n_splits=1, train_size=sample_fraction, random_state=42)
                for train_idx, _ in sss.split(X_train, y_train):
                    # Create a subsample with balanced classes
                    X_subsample = X_train.iloc[train_idx]
                    y_subsample = y_train[train_idx]
                    
                    # Check that all classes are present
                    if len(np.unique(y_subsample)) == num_classes:
                        # Train a new model on this subsample
                        subsample_model = XGBClassifier(**{k: v for k, v in models['XGBoost'].get_params().items() 
                                                      if k != 'xgb_model'})
                        subsample_model.fit(X_subsample, y_subsample)
                        
                        # Evaluate on both the subsample (train) and the test set
                        train_pred = subsample_model.predict(X_subsample)
                        test_pred = subsample_model.predict(X_test)
                        
                        train_score = accuracy_score(y_subsample, train_pred)
                        test_score = accuracy_score(y_test, test_pred)
                        
                        train_scores.append(train_score)
                        test_scores.append(test_score)
            
            if train_scores:  # Only print if we have scores
                print(f"Train sizes: {train_sizes_abs}")
                print(f"Train scores: {train_scores}")
                print(f"Test scores: {test_scores}")
                
                # Analyze overfitting pattern
                initial_gap = train_scores[0] - test_scores[0] if train_scores and test_scores else float('nan')
                final_gap = train_scores[-1] - test_scores[-1] if train_scores and test_scores else float('nan')
                
                print("\nXGBoost Overfitting Analysis:")
                print(f"Initial gap (small dataset): {initial_gap:.4f}")
                print(f"Final gap (full dataset): {final_gap:.4f}")
                
                if initial_gap > 0.3 and final_gap > 0.1:
                    print("Pattern indicates overfitting: Train accuracy >> Test accuracy")
                    print("Recommendations:")
                    print("- Increase regularization (increase gamma, reg_alpha, reg_lambda)")
                    print("- Decrease model complexity (reduce depth, increase min_child_weight)")
                    print("- Try feature selection to focus on most important features")
                elif test_scores[-1] - test_scores[0] > 0.2:
                    print("Pattern shows good learning: Test accuracy steadily improving with more data")
                    print("Recommendations:")
                    print("- Collect more training data if possible")
                    print("- Current parameters seem well-tuned")
                else:
                    print("Pattern suggests underfitting: Both train and test accuracy are low")
                    print("Recommendations:")
                    print("- Increase model complexity (increase depth, decrease min_child_weight)")
        elif name == 'Neural Network':
            print("\nNeural Network Training Details:")
            print(f"Number of layers: {len(model.hidden_layer_sizes) + 2}")
            print(f"Layer sizes: {model.hidden_layer_sizes}")
            print(f"L2 regularization (alpha): {model.alpha}")
            print(f"Learning rate: {model.learning_rate_init}")
            print(f"Batch size: {model.batch_size}")
            print(f"Early stopping: {model.early_stopping}")
            print(f"Validation fraction: {model.validation_fraction}")
            # MLPClassifier does not have feature_importances_
            print("Note: MLPClassifier doesn't provide feature importance scores")
            train_sizes, train_scores, test_scores = learning_curve(
                model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10))
            print("\nLearning Curve Results:")
            print(f"Train sizes: {train_sizes}")
            print(f"Train scores mean: {np.mean(train_scores, axis=1)}")
            print(f"Test scores mean: {np.mean(test_scores, axis=1)}")
        else:
            # For other models, use standard cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X, y, cv=cv)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        
        # Store results
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'CV Mean Score': cv_mean,
            'CV Std': cv_std
        })
        
        # Save the model with timestamp
        model_path = f'{run_dir}/{name.lower().replace(" ", "_")}_model_{timestamp}.joblib'
        joblib.dump(model, model_path)
        print(f"Model saved to: {model_path}")
        
        # Print model-specific details
        if name == 'Random Forest':
            print("\nRandom Forest Details:")
            print(f"Number of trees: {model.n_estimators}")
            print(f"Max depth: {model.max_depth}")
            print(f"Min samples split: {model.min_samples_split}")
            print(f"Min samples leaf: {model.min_samples_leaf}")
            print(f"Max features: {model.max_features}")
            # Add after training Random Forest
            feature_importances = pd.DataFrame({
                'feature': feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            print("Top 10 most important features:")
            print(feature_importances.head(10))
            train_sizes, train_scores, test_scores = learning_curve(
                model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10))
            print("\nLearning Curve Results:")
            print(f"Train sizes: {train_sizes}")
            print(f"Train scores mean: {np.mean(train_scores, axis=1)}")
            print(f"Test scores mean: {np.mean(test_scores, axis=1)}")
            # After learning curve for Random Forest
            final_gap = np.mean(train_scores, axis=1)[-1] - np.mean(test_scores, axis=1)[-1]
            initial_gap = np.mean(train_scores, axis=1)[0] - np.mean(test_scores, axis=1)[0]
            
            print("\nRandom Forest Overfitting Analysis:")
            print(f"Initial gap (small dataset): {initial_gap:.4f}")
            print(f"Final gap (full dataset): {final_gap:.4f}")
            
        elif name == 'XGBoost':
            print("\nXGBoost Details:")
            print(f"Number of trees: {model.n_estimators}")
            print(f"Max depth: {model.max_depth}")
            print(f"Learning rate: {model.learning_rate}")
            print(f"Subsample: {model.subsample}")
            print(f"Colsample bytree: {model.colsample_bytree}")
            print(f"Min child weight: {model.min_child_weight}")
            print(f"Objective: {model.objective}")
            print(f"Number of classes: {num_classes}")
            feature_importances = pd.DataFrame({
                'feature': feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            print("Top 10 most important features:")
            print(feature_importances.head(10))
            
        elif name == 'Neural Network':
            print("\nNeural Network Training Details:")
            print(f"Number of layers: {len(model.hidden_layer_sizes) + 2}")
            print(f"Layer sizes: {model.hidden_layer_sizes}")
            print(f"L2 regularization (alpha): {model.alpha}")
            print(f"Learning rate: {model.learning_rate_init}")
            print(f"Batch size: {model.batch_size}")
            print(f"Early stopping: {model.early_stopping}")
            print(f"Validation fraction: {model.validation_fraction}")
            # MLPClassifier does not have feature_importances_
            print("Note: MLPClassifier doesn't provide feature importance scores")
            train_sizes, train_scores, test_scores = learning_curve(
                model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10))
            print("\nLearning Curve Results:")
            print(f"Train sizes: {train_sizes}")
            print(f"Train scores mean: {np.mean(train_scores, axis=1)}")
            print(f"Test scores mean: {np.mean(test_scores, axis=1)}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results with timestamp
    results_path = f'{run_dir}/model_comparison_results_{timestamp}.csv'
    results_df.to_csv(results_path, index=False)
    
    # Print results
    print("\nModel Comparison Results:")
    print("-" * 80)
    print(results_df.to_string())
    
    # Find best model
    best_model = results_df.loc[results_df['Accuracy'].idxmax()]
    print(f"\nBest performing model: {best_model['Model']}")
    print(f"Accuracy: {best_model['Accuracy']:.4f}")
    print(f"F1-Score: {best_model['F1-Score']:.4f}")
    
    # Print label mapping
    print("\nLabel Mapping:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"{i}: {label}")
    
    # Save run information with actual model parameters
    run_info = {
        'timestamp': timestamp,
        'best_model': best_model['Model'],
        'best_accuracy': best_model['Accuracy'],
        'best_f1': best_model['F1-Score'],
        'input_file': file_path,
        'total_samples': len(df),
        'feature_count': len(feature_columns),
        'class_count': num_classes,
        'train_test_split': '80/20',
        'random_state': 42,
        'cross_validation_folds': 5,
        'class_distribution': dict(class_distribution),
        'training_data': 'Only positive samples',
        'models': {
            # 'Random Forest': {
            #     'n_estimators': models['Random Forest'].n_estimators,
            #     'max_depth': models['Random Forest'].max_depth,
            #     'min_samples_split': models['Random Forest'].min_samples_split,
            #     'min_samples_leaf': models['Random Forest'].min_samples_leaf,
            #     'max_features': models['Random Forest'].max_features,
            #     'bootstrap': models['Random Forest'].bootstrap,
            #     'class_weight': models['Random Forest'].class_weight
            # },
            'XGBoost': {
                'n_estimators': models['XGBoost'].n_estimators,
                'max_depth': models['XGBoost'].max_depth,
                'learning_rate': models['XGBoost'].learning_rate,
                'subsample': models['XGBoost'].subsample,
                'colsample_bytree': models['XGBoost'].colsample_bytree,
                'min_child_weight': models['XGBoost'].min_child_weight,
                'gamma': models['XGBoost'].gamma,
                'reg_alpha': models['XGBoost'].reg_alpha,
                'reg_lambda': models['XGBoost'].reg_lambda,
                'objective': models['XGBoost'].objective,
                'num_class': num_classes
            }
            # 'Neural Network': {
            #     'hidden_layer_sizes': models['Neural Network'].hidden_layer_sizes,
            #     'max_iter': models['Neural Network'].max_iter,
            #     'alpha': models['Neural Network'].alpha,
            #     'learning_rate_init': models['Neural Network'].learning_rate_init,
            #     'early_stopping': models['Neural Network'].early_stopping,
            #     'validation_fraction': models['Neural Network'].validation_fraction,
            #     'n_iter_no_change': models['Neural Network'].n_iter_no_change,
            #     'batch_size': models['Neural Network'].batch_size
            # }
        }
    }
    
    # Save run information with pretty formatting
    with open(f'{run_dir}/run_info_{timestamp}.txt', 'w') as f:
        f.write("=== Run Information ===\n")
        f.write(f"Timestamp: {run_info['timestamp']}\n")
        f.write(f"Input File: {run_info['input_file']}\n")
        f.write(f"Training Data: {run_info['training_data']}\n")
        f.write(f"Best Model: {run_info['best_model']}\n")
        f.write(f"Best Accuracy: {run_info['best_accuracy']:.4f}\n")
        f.write(f"Best F1-Score: {run_info['best_f1']:.4f}\n")
        f.write(f"Total Samples: {run_info['total_samples']}\n")
        f.write(f"Feature Count: {run_info['feature_count']}\n")
        f.write(f"Class Count: {run_info['class_count']}\n")
        f.write(f"Train/Test Split: {run_info['train_test_split']}\n")
        f.write(f"Random State: {run_info['random_state']}\n")
        f.write(f"Cross-Validation Folds: {run_info['cross_validation_folds']}\n\n")
        
        f.write("=== Class Distribution ===\n")
        for class_idx, count in run_info['class_distribution'].items():
            f.write(f"Class {label_encoder.classes_[class_idx]}: {count} samples ({count/run_info['total_samples']*100:.2f}%)\n")
        f.write("\n")
        
        f.write("=== Model Hyperparameters ===\n")
        for model_name, params in run_info['models'].items():
            f.write(f"\n{model_name}:\n")
            for param, value in params.items():
                f.write(f"  {param}: {value}\n")
    
    print(f"\nAll files saved in: {run_dir}")

# Create a function to evaluate on multiple datasets, which is not used right now
def evaluate_on_datasets(model, label_encoder, feature_columns, datasets):
    results = {}
    for name, path in datasets.items():
        print(f"Evaluating on dataset: {name} ({path})")
        test_df = pd.read_csv(path)
        test_X = test_df[[col for col in test_df.columns if col in feature_columns]]
        if len(test_X.columns) != len(feature_columns):
            print(f"Warning: Feature mismatch. Expected {len(feature_columns)}, got {len(test_X.columns)}")
            missing = set(feature_columns) - set(test_X.columns)
            extra = set(test_X.columns) - set(feature_columns)
            if missing:
                print(f"Missing features: {missing}")
            if extra:
                print(f"Extra features: {extra}")
            continue
        
        try:
            test_y = label_encoder.transform(test_df['label'])
            accuracy = accuracy_score(test_y, model.predict(test_X))
            results[name] = accuracy
            print(f"Accuracy on {name}: {accuracy:.4f}")
        except Exception as e:
            print(f"Error evaluating on {name}: {e}")
    return results

if __name__ == "__main__":
    train_and_evaluate_models() 
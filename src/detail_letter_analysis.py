import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import joblib
from datetime import datetime
import json

def evaluate_with_detailed_letter_analysis(model, label_encoder, feature_columns, test_dataset_path):
    """
    Analyze model performance on each letter with detailed statistics on positive and negative samples.
    
    Args:
        model: Trained model (e.g., XGBoost classifier)
        label_encoder: Label encoder used during training
        feature_columns: List of feature column names used by the model
        test_dataset_path: Path to the test dataset CSV containing positive and negative samples
    
    Returns:
        letter_stats: DataFrame with detailed statistics for each letter
        confusion_matrix: DataFrame showing confusion between letters
    """
    print(f"Loading test dataset from: {test_dataset_path}")
    test_df = pd.read_csv(test_dataset_path)
    
    # Check if sample_type column exists
    if 'sample_type' not in test_df.columns:
        print("Error: Dataset doesn't contain 'sample_type' column to distinguish positive/negative samples")
        return None
    
    # Extract features for prediction
    test_X = test_df[[col for col in test_df.columns if col in feature_columns]]
    if len(test_X.columns) != len(feature_columns):
        print(f"Warning: Feature mismatch. Expected {len(feature_columns)}, got {len(test_X.columns)}")
        missing = set(feature_columns) - set(test_X.columns)
        if missing:
            print(f"Missing features: {missing}")
        return None
    
    # Get predictions and confidence scores
    print("Making predictions...")
    y_pred_idx = model.predict(test_X)
    y_pred_proba = model.predict_proba(test_X)
    
    # Add predictions and confidence to the dataframe
    test_df['predicted_idx'] = y_pred_idx
    test_df['predicted_label'] = label_encoder.inverse_transform(y_pred_idx)
    test_df['confidence'] = [y_pred_proba[i, pred] for i, pred in enumerate(y_pred_idx)]
    
    # Get unique letters in the dataset
    letters = sorted(test_df['label'].unique())
    
    # Initialize results storage
    letter_stats = []
    thresholds_to_try = np.arange(0.1, 1.0, 0.05)
    
    # Analyze each letter
    for letter in letters:
        print(f"\nAnalyzing letter: {letter}")
        
        # Get positive and negative samples for this letter
        letter_pos = test_df[(test_df['label'] == letter) & (test_df['sample_type'] == 'positive')]
        letter_neg = test_df[(test_df['label'] == letter) & (test_df['sample_type'] == 'negative')]
        
        # Skip if no samples
        if len(letter_pos) == 0 and len(letter_neg) == 0:
            print(f"  No samples found for letter {letter}")
            continue
            
        # Calculate statistics
        pos_count = len(letter_pos)
        neg_count = len(letter_neg)
        
        # Positive sample accuracy (true positives)
        pos_correct = (letter_pos['label'] == letter_pos['predicted_label']).sum()
        pos_accuracy = pos_correct / pos_count if pos_count > 0 else np.nan
        
        # Negative sample rejection (true negatives)
        neg_rejected = (letter_neg['predicted_label'] != letter).sum()
        neg_rejection_rate = neg_rejected / neg_count if neg_count > 0 else np.nan
        
        # Confidence statistics
        pos_confidence_mean = letter_pos['confidence'].mean() if pos_count > 0 else np.nan
        pos_confidence_std = letter_pos['confidence'].std() if pos_count > 0 else np.nan
        pos_confidence_min = letter_pos['confidence'].min() if pos_count > 0 else np.nan
        pos_confidence_max = letter_pos['confidence'].max() if pos_count > 0 else np.nan
        
        neg_confidence_mean = letter_neg['confidence'].mean() if neg_count > 0 else np.nan
        neg_confidence_std = letter_neg['confidence'].std() if neg_count > 0 else np.nan
        neg_confidence_min = letter_neg['confidence'].min() if neg_count > 0 else np.nan
        neg_confidence_max = letter_neg['confidence'].max() if neg_count > 0 else np.nan
        
        # Find the optimal threshold
        optimal_threshold = 0.5
        best_f1 = 0
        threshold_results = []
        
        if pos_count > 0 and neg_count > 0:
            for threshold in thresholds_to_try:
                # Count true positives (positive samples correctly predicted with confidence >= threshold)
                tp = ((letter_pos['predicted_label'] == letter) & (letter_pos['confidence'] >= threshold)).sum()
                
                # Count false negatives (positive samples not predicted as this letter or below threshold)
                fn = pos_count - tp
                
                # Count false positives (negative samples predicted as this letter with confidence >= threshold)
                fp = ((letter_neg['predicted_label'] == letter) & (letter_neg['confidence'] >= threshold)).sum()
                
                # Count true negatives (negative samples correctly not predicted as this letter or below threshold)
                tn = neg_count - fp
                
                # Calculate metrics
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                accuracy = (tp + tn) / (pos_count + neg_count)
                
                threshold_results.append({
                    'threshold': threshold,
                    'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'accuracy': accuracy
                })
                
                if f1 > best_f1:
                    best_f1 = f1
                    optimal_threshold = threshold
        
        # Store the statistics
        letter_stats.append({
            'letter': letter,
            'pos_samples': pos_count,
            'neg_samples': neg_count,
            'pos_accuracy': pos_accuracy,
            'neg_rejection_rate': neg_rejection_rate,
            'pos_conf_mean': pos_confidence_mean,
            'pos_conf_std': pos_confidence_std,
            'pos_conf_min': pos_confidence_min,
            'pos_conf_max': pos_confidence_max,
            'neg_conf_mean': neg_confidence_mean,
            'neg_conf_std': neg_confidence_std,
            'neg_conf_min': neg_confidence_min,
            'neg_conf_max': neg_confidence_max,
            'optimal_threshold': optimal_threshold,
            'best_f1': best_f1,
            'threshold_results': threshold_results
        })
        
        # Print summary statistics for this letter
        print(f"  Positive samples: {pos_count}, Accuracy: {pos_accuracy:.4f}")
        print(f"  Negative samples: {neg_count}, Rejection rate: {neg_rejection_rate:.4f}")
        print(f"  Positive confidence: {pos_confidence_mean:.4f} ± {pos_confidence_std:.4f} (range: {pos_confidence_min:.4f}-{pos_confidence_max:.4f})")
        if neg_count > 0:
            print(f"  Negative confidence: {neg_confidence_mean:.4f} ± {neg_confidence_std:.4f} (range: {neg_confidence_min:.4f}-{neg_confidence_max:.4f})")
        print(f"  Optimal threshold: {optimal_threshold:.2f} (F1: {best_f1:.4f})")
    
    # Create a DataFrame with the statistics
    stats_df = pd.DataFrame([{k: v for k, v in stat.items() if k != 'threshold_results'} for stat in letter_stats])
    
    # Generate confusion matrix for positive samples
    pos_samples = test_df[test_df['sample_type'] == 'positive']
    conf_matrix = pd.crosstab(
        pos_samples['label'], 
        pos_samples['predicted_label'],
        rownames=['Actual'],
        colnames=['Predicted'],
        normalize='index'
    )
    
    # Calculate overall statistics
    all_pos = test_df[test_df['sample_type'] == 'positive']
    all_neg = test_df[test_df['sample_type'] == 'negative']
    
    pos_accuracy = (all_pos['label'] == all_pos['predicted_label']).mean()
    neg_rejection = (all_neg['label'] != all_neg['predicted_label']).mean()
    
    print("\nOverall Statistics:")
    print(f"Positive samples: {len(all_pos)}, Accuracy: {pos_accuracy:.4f}")
    print(f"Negative samples: {len(all_neg)}, Rejection rate: {neg_rejection:.4f}")
    
    # Write detailed statistics to CSV
    stats_df.to_csv(f"letter_statistics_{datetime.now().strftime('%m%d_%H%M%S')}.csv", index=False)
    
    # Save threshold curves for each letter
    threshold_curves = {}
    for stat in letter_stats:
        if 'threshold_results' in stat and stat['threshold_results']:
            threshold_curves[stat['letter']] = pd.DataFrame(stat['threshold_results'])
    
    # Create a combined DataFrame with threshold data for all letters
    if threshold_curves:
        all_thresholds = pd.DataFrame()
        for letter, df in threshold_curves.items():
            df['letter'] = letter
            all_thresholds = pd.concat([all_thresholds, df])
        
        all_thresholds.to_csv(f"threshold_curves_{datetime.now().strftime('%m%d_%H%M%S')}.csv", index=False)
    
    return stats_df, conf_matrix

def find_optimal_thresholds_by_letter(model_path, label_encoder_path, test_dataset_path):
    """
    Find optimal confidence thresholds for each letter and save them for real-time use.
    """
    # Load model and label encoder
    model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)
    
    # Get feature columns (exclude metadata columns)
    metadata_cols = ['hand_type', 'label', 'sample_type', 'timestamp']
    test_df = pd.read_csv(test_dataset_path)
    feature_columns = [col for col in test_df.columns if col not in metadata_cols]
    
    # Run detailed analysis
    stats_df, _ = evaluate_with_detailed_letter_analysis(model, label_encoder, feature_columns, test_dataset_path)
    
    # Extract optimal thresholds
    thresholds = {row['letter']: row['optimal_threshold'] for _, row in stats_df.iterrows()}
    
    # Save thresholds to file
    threshold_path = f"letter_thresholds_{datetime.now().strftime('%m%d_%H%M%S')}.json"
    with open(threshold_path, 'w') as f:
        json.dump(thresholds, f, indent=2)
    
    print(f"Optimal thresholds saved to: {threshold_path}")
    print("\nLetter Thresholds:")
    for letter, threshold in thresholds.items():
        print(f"  {letter}: {threshold:.2f}")
    
    return thresholds

if __name__ == "__main__":
    model_path = '/Users/wuhaodong/SFhack/models/new_run_0418_150227/xgboost_model_0418_150227.joblib'
    label_encoder_path = '/Users/wuhaodong/SFhack/models/new_run_0418_150227/label_encoder_0418_150227.joblib'
    test_dataset_path = '/Users/wuhaodong/SFhack/asl_optimized_features_0418_145734.csv'
    thresholds = find_optimal_thresholds_by_letter(model_path, label_encoder_path, test_dataset_path)
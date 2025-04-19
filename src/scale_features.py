import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def scale_features():
    # Read the original data
    df = pd.read_csv('/Users/wuhaodong/SFhack/asl_features.csv')
    
    # Separate features and labels
    feature_columns = [col for col in df.columns if col not in ['hand_type', 'label', 'sample_type']]
    X = df[feature_columns]
    y = df[['label', 'sample_type', 'hand_type']]
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Fit and transform the features
    X_scaled = scaler.fit_transform(X)
    
    # Convert back to DataFrame
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns)
    
    # Combine with labels
    scaled_df = pd.concat([X_scaled_df, y], axis=1)
    
    # Save scaled data
    scaled_df.to_csv('/Users/wuhaodong/SFhack/asl_features_scaled.csv', index=False)
    
    # Print some statistics to verify scaling
    print("\nFeature scaling statistics:")
    print("-" * 50)
    print("Original data statistics:")
    print(X.describe().loc[['mean', 'std', 'min', 'max']])
    print("\nScaled data statistics:")
    print(X_scaled_df.describe().loc[['mean', 'std', 'min', 'max']])
    
    # Save the scaler for later use
    import joblib
    joblib.dump(scaler, '/Users/wuhaodong/SFhack/feature_scaler.joblib')
    print("\nScaler saved to: /Users/wuhaodong/SFhack/feature_scaler.joblib")

if __name__ == "__main__":
    scale_features() 
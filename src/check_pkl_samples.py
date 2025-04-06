#!/usr/bin/env python3
import os
import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Check sample counts in ASL feature pkl files')
    parser.add_argument('--data_dir', type=str, default='./webcam_data',
                        help='Directory containing the feature pkl files')
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Directory '{args.data_dir}' does not exist")
        return
    
    all_characters = [chr(i) for i in range(ord('a'), ord('z')+1)]
    total_samples = 0
    
    print(f"Checking sample counts in {args.data_dir}:")
    print("-" * 60)
    print(f"{'Character':<10} | {'Right Hand':<12} | {'Left Hand':<12} | {'Negative':<12} | {'Total':<12}")
    print("-" * 60)
    
    for char in all_characters:
        data_file = os.path.join(args.data_dir, f"features_{char}.pkl")
        
        if not os.path.exists(data_file):
            continue
            
        with open(data_file, 'rb') as f:
            try:
                data = pickle.load(f)
                
                right_count = len(data.get('X_positive_right', [])) 
                left_count = len(data.get('X_positive_left', []))
                neg_count = len(data.get('X_negative', []))
                
                # Handle old format
                if 'X_positive' in data and 'X_positive_right' not in data:
                    right_count = len(data['X_positive'])
                    left_count = 0
                
                char_total = right_count + left_count + neg_count
                total_samples += char_total
                
                print(f"{char.upper():<10} | {right_count:<12} | {left_count:<12} | {neg_count:<12} | {char_total:<12}")
                
            except Exception as e:
                print(f"{char.upper():<10} | Error: {str(e)}")
    
    print("-" * 60)
    print(f"Total samples across all characters: {total_samples}")
    
    # Check feature vector dimensions
    print("\nFeature vector dimensions:")
    for char in all_characters:
        data_file = os.path.join(args.data_dir, f"features_{char}.pkl")
        if os.path.exists(data_file):
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
                
                # Try to get the dimension from any available sample
                if data.get('X_positive_right') and len(data['X_positive_right']) > 0:
                    print(f"Feature vector length: {len(data['X_positive_right'][0])}")
                    break
                elif data.get('X_positive_left') and len(data['X_positive_left']) > 0:
                    print(f"Feature vector length: {len(data['X_positive_left'][0])}")
                    break
                elif data.get('X_positive') and len(data['X_positive']) > 0:
                    print(f"Feature vector length: {len(data['X_positive'][0])}")
                    break
                elif data.get('X_negative') and len(data['X_negative']) > 0:
                    print(f"Feature vector length: {len(data['X_negative'][0])}")
                    break

if __name__ == "__main__":
    main() 
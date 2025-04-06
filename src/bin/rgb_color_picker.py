#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import os

# Global variables
points = []
image = None
window_name = "RGB Color Picker"

def click_event(event, x, y, flags, param):
    """Handle mouse clicks to get RGB values"""
    global points, image
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get BGR values (OpenCV uses BGR format)
        b, g, r = image[y, x]
        
        # Print RGB values (converting from BGR to RGB)
        print(f"RGB at position ({x},{y}): R={r}, G={g}, B={b}")
        print(f"BGR (OpenCV format): {b}, {g}, {r}")
        print(f"Hex Color: #{r:02x}{g:02x}{b:02x}")
        print("-" * 40)
        
        # Draw a circle at clicked point for reference
        cv2.circle(image, (x, y), 3, (0, 255, 255), -1)
        
        # Add to points list
        points.append({"position": (x, y), "rgb": (r, g, b), "bgr": (b, g, r)})
        
        # Display the image with the clicked points
        cv2.imshow(window_name, image)

def main():
    global image, window_name
    
    parser = argparse.ArgumentParser(description='RGB Color Picker Tool')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--output', help='Optional output directory for annotated images')
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.isfile(args.image):
        print(f"Error: Image file '{args.image}' not found.")
        return
    
    # Read the image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not read image '{args.image}'.")
        return
    
    # Create a window and set the callback function
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_event)
    
    # Display instructions
    print(f"\nINSTRUCTIONS:")
    print(f"- Click on pixels to get their exact RGB values")
    print(f"- Press 's' to save the annotated image (if output directory specified)")
    print(f"- Press 'r' to reset the annotations")
    print(f"- Press 'q' or ESC to quit\n")
    print(f"Starting color picker for: {args.image}")
    print("-" * 40)
    
    # Show the image
    cv2.imshow(window_name, image)
    
    # Keep the window open until the user presses 'q' or ESC
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        # Quit if 'q' or ESC is pressed
        if key == ord('q') or key == 27:  # 27 is the ASCII code for ESC
            break
        
        # Save the image with annotations
        elif key == ord('s') and args.output:
            # Create output directory if it doesn't exist
            os.makedirs(args.output, exist_ok=True)
            
            # Generate output filename
            base_filename = os.path.basename(args.image)
            output_path = os.path.join(args.output, f"annotated_{base_filename}")
            
            # Save the image
            cv2.imwrite(output_path, image)
            print(f"Saved annotated image to: {output_path}")
            
            # Also save text file with color information
            txt_output_path = os.path.join(args.output, f"colors_{os.path.splitext(base_filename)[0]}.txt")
            with open(txt_output_path, 'w') as f:
                f.write(f"Colors from {base_filename}:\n")
                f.write("-" * 40 + "\n")
                for i, point in enumerate(points):
                    pos = point["position"]
                    rgb = point["rgb"]
                    f.write(f"Point {i+1} at ({pos[0]},{pos[1]}): RGB={rgb}, Hex=#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}\n")
            print(f"Saved color information to: {txt_output_path}")
        
        # Reset annotations
        elif key == ord('r'):
            image = cv2.imread(args.image)
            points = []
            cv2.imshow(window_name, image)
            print("Reset annotations")
    
    # Close all windows
    cv2.destroyAllWindows()
    
    # Print summary of all collected points
    if points:
        print("\nSummary of collected color points:")
        print("-" * 40)
        for i, point in enumerate(points):
            pos = point["position"]
            rgb = point["rgb"]
            print(f"Point {i+1} at ({pos[0]},{pos[1]}): RGB={rgb}, Hex=#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}")

if __name__ == "__main__":
    main() 
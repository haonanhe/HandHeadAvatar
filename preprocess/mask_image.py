import os
import cv2
import numpy as np

import argparse

# Define argument parser
parser = argparse.ArgumentParser(description="Apply masks to RGB images and save the results with a white background.")
parser.add_argument('--parent_dir', type=str, required=True, help="parent_dir.")
args = parser.parse_args()

# Define directories
# parent_dir = '/home/haonan/Codes/IMavatar/data/datasets/Jan_17/Jan_17/rgb'
parent_dir = args.parent_dir
image_dir = 'image'
mask_dir = 'mask'
output_dir = 'masked_rgb_cropped'
image_dir = os.path.join(parent_dir, image_dir)
mask_dir = os.path.join(parent_dir, mask_dir)
output_dir = os.path.join(parent_dir, output_dir)

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get list of image and mask files
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

# Ensure the number of images and masks match
if len(image_files) != len(mask_files):
    raise ValueError("The number of images and masks do not match.")

# Process each image and mask
for img_file, mask_file in zip(image_files, mask_files):
    # Read image and mask
    img_path = os.path.join(image_dir, img_file)
    mask_path = os.path.join(mask_dir, mask_file)
    
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Read mask as grayscale

    # Ensure mask is binary (0 and 255)
    # _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    # Resize mask to match image dimensions if necessary
    if img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    # Convert mask to 3 channels to match image
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Apply mask to image
    masked_img = cv2.bitwise_and(img, mask)

    # Create a white background
    white_background = np.ones_like(img) * 255  # Create a white image of the same size

    # Invert the mask to get the background region
    inverted_mask = cv2.bitwise_not(mask)

    # Combine the masked image and white background
    final_img = cv2.bitwise_or(masked_img, cv2.bitwise_and(white_background, inverted_mask))

    # Save the final image
    output_path = os.path.join(output_dir, img_file)
    cv2.imwrite(output_path, final_img)

    print(f"Processed and saved: {output_path}")

print("All images have been processed and saved.")
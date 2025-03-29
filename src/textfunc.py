import cv2
import os
from pathlib import Path

def clean_dir(path):
    """Recursively delete files in a directory."""
    os.makedirs(path, exist_ok=True)
    directory = Path(path)
    for item in directory.iterdir():
        if item.is_dir():
            clean_dir(item)
        else:
            item.unlink()

def pre_process_image(image):
    """Convert image to binary while removing noise."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to create a binary image since findContours only works with binary
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    return binary

def process_letter(contour, image, letters_size, target_dir, index_counter):
    """Extract, resize, and save a letter image."""
    x, y, w, h = cv2.boundingRect(contour)

    # Ignore small contours (filtering noise)
    if w > image.shape[1] * 0.05 and h > image.shape[0] * 0.1:
        pad_w, pad_h = int(w * 0.1), int(h * 0.1)

        # Define new bounding box with padding
        x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
        x2, y2 = min(image.shape[1], x + w + pad_w), min(image.shape[0], y + h + pad_h)

        # Extract letter
        letter = image[y1:y2, x1:x2]

        # Resize letter
        resized_letter = cv2.resize(letter, letters_size, interpolation=cv2.INTER_AREA)

        # Save image
        output_path = os.path.join(target_dir, f"letter_{index_counter[0]:04d}.png")
        if cv2.imwrite(output_path, resized_letter):
            index_counter[0] += 1

def generate_letters(target_dir, image_path, letters_size):
    """Detect letters, extract them, and save resized versions."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"The file \"{image_path}\" does not exist")

    binary = pre_process_image(image)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours from left to right
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    # Create output directory
    os.makedirs(target_dir, exist_ok=True)

    index_counter = [0]

    # Process each detected letter in parallel
    for i, contour in enumerate(contours):
        process_letter(contour, image, letters_size, target_dir, index_counter)

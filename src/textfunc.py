import cv2
import os
from pathlib import Path

def clean_dir(path):
    directory = Path(path)
    for item in directory.iterdir():
        if item.is_dir():
            clean_dir(item)
        else:
            item.unlink()

def generate_letters(target_dir, image_path, letters_size):
    image = cv2.imread(image_path)
    if (image is None):
        raise FileNotFoundError(f"The file \"{image_path}\" does not exist")

    binary = pre_process_image(image_path)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours from left to right
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    # Create output directory
    os.makedirs(target_dir, exist_ok=True)

    # Process each detected letter
    n = 0
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)

        # Ignore small contours (filtering noise)
        if w > 5 and h > 10:
            # Calculate padding (10% of width and height)
            pad_w = int(w * 0.1)
            pad_h = int(h * 0.1)

            # Define new bounding box with padding
            x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
            x2, y2 = min(image.shape[1], x + w + pad_w), min(image.shape[0], y + h + pad_h)

            # Extract letter with padding (from original image, without green rectangles)
            letter = image[y1:y2, x1:x2]

            # Resize
            resized_letter = cv2.resize(letter, letters_size, interpolation=cv2.INTER_AREA)

            # Save resized letter
            output_path = os.path.join(target_dir, f"letter_{n}.png")
            cv2.imwrite(output_path, resized_letter)
            n += 1

def pre_process_image(image_path):
    image = cv2.imread(image_path)
    if (image is None):
        raise FileNotFoundError(f"The file \"{image_path}\" does not exist")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create a binary image since findContours only works with binary
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    return binary

import cv2
import os
from pathlib import Path

def clean_dir(path):
    """Recursively delete files in a directory."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
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

def process_letter(contour, image, target_dir, index_counter):
    """Extract, resize, and save a letter image."""
    x, y, w, h = cv2.boundingRect(contour)

    # Ignore small contours (filtering noise)
    if w > image.shape[1] * 0.025 and h > image.shape[0] * 0.05:
        pad_w, pad_h = int(w * 0.1), int(h * 0.1)

        # Define new bounding box with padding
        x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
        x2, y2 = min(image.shape[1], x + w + pad_w), min(image.shape[0], y + h + pad_h)

        # Extract letter
        letter = image[y1:y2, x1:x2]

        # Save image
        output_path = os.path.join(target_dir, f"letter_{index_counter[0]:04d}.png")
        if cv2.imwrite(output_path, letter):
            index_counter[0] += 1

def generate_letters(target_dir, image_path):
    """Detect letters, extract them, and save resized versions."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"The file \"{image_path}\" does not exist")

    binary = pre_process_image(image)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Define minimum contour size
    MIN_WIDTH, MIN_HEIGHT = 5, 10

    # Filter and sort contours from left to right
    contours = sorted(
        [c for c in contours if cv2.boundingRect(c)[2] > MIN_WIDTH and cv2.boundingRect(c)[3] > MIN_HEIGHT],
        key=lambda c: cv2.boundingRect(c)[0]
    )

    # Create output directory
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    index_counter = [0]

    # Process each detected letter
    for i, contour in enumerate(contours):
        process_letter(contour, image, target_dir, index_counter)

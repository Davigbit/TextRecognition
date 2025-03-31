import os
import string
import random
import pickle
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Generate list of characters (A-Z, a-z, 0-9)
string_list = list(string.ascii_uppercase + string.ascii_lowercase + string.digits)

# Base directory for NIST dataset
base_dir = "data/NIST"

def process_char(char):
    """Processes a character directory, returning a list of (char, img_path) tuples."""
    hex_char = format(ord(char), 'x')  # Convert character to hexadecimal
    char_dir = os.path.join(base_dir, hex_char)
    all_files = []

    if not os.path.exists(char_dir):
        return []

    for folder in os.listdir(char_dir):
        folder_path = os.path.join(char_dir, folder)
        if os.path.isdir(folder_path) and folder.startswith("hsf_"):
            # Convert file listing to a set for O(1) lookups
            folder_files = set(os.listdir(folder_path))
            all_files.extend([(char, os.path.join(folder_path, f)) for f in folder_files])

    return all_files

# Use multiprocessing to process characters in parallel
with Pool(cpu_count()) as p:
    results = list(tqdm(p.imap(process_char, string_list), total=len(string_list), desc="Processing Characters"))

# Flatten the results list
all_samples = [item for sublist in results for item in sublist]

# Shuffle dataset randomly
random.shuffle(all_samples)

# Save dataset to a file
with open("src/model/NIST.pkl", "wb") as f:
    pickle.dump(all_samples, f)

print(f"Dataset saved: {len(all_samples)} samples")

import os
import string
import random
import pickle

# Generate list of characters (A-Z, a-z, 0-9)
string_list = list(string.ascii_uppercase + string.ascii_lowercase + string.digits)

data_map = {}

# Base directory for NIST dataset
base_dir = "data/NIST"

for char in string_list:
    hex_char = format(ord(char), 'x')  # Convert character to hexadecimal
    char_dir = os.path.join(base_dir, hex_char)  # Path to character directory
    
    data_map[char] = []  # Initialize list for character files
    
    # Iterate over all hsf_* folders dynamically
    for folder in os.listdir(char_dir):
        folder_path = os.path.join(char_dir, folder)
        if os.path.isdir(folder_path) and folder.startswith("hsf_"):
            data_map[char].extend(os.listdir(folder_path))

# Create a list of (character, image path) tuples
all_samples = []
for char, file_list in data_map.items():
    hex_char = format(ord(char), 'x')  # Convert character to hexadecimal
    char_dir = os.path.join(base_dir, hex_char)  # Character directory
    
    for filename in file_list:
        # Find the correct hsf_* folder for each file
        for folder in os.listdir(char_dir):
            folder_path = os.path.join(char_dir, folder)
            if os.path.isdir(folder_path) and folder.startswith("hsf_") and filename in os.listdir(folder_path):
                img_path = os.path.join(folder_path, filename)
                all_samples.append((char, img_path))
                break  # Stop searching once the file is found

# Shuffle dataset randomly
random.shuffle(all_samples)

# Save all_samples to a file
with open("NIST.pkl", "wb") as f:
    pickle.dump(all_samples, f)

def load_data(per_train, per_val, per_test):
    assert(per_test + per_train + per_val == 100)

    with open("NIST.pkl", "rb") as f:
        NIST_dataset = pickle.load(f)

    total = len(NIST_dataset)
    train_end = int(per_train * total / 100)
    val_end = int((per_train + per_val) * total / 100)

    train_data = NIST_dataset[:train_end]
    val_data = NIST_dataset[train_end:val_end]
    test_data = NIST_dataset[val_end:]

    return (train_data, val_data, test_data)

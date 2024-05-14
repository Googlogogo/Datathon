import pandas as pd
from PIL import Image
import numpy as np
import os

# Function to load and process images
def load_and_process_images(df, base_dir, size=(32, 32)):
    images = []
    labels = []

    for _, row in df.iterrows():
        filepath = os.path.join(base_dir, row['Path'])
        try:
            # Open the image file
            with Image.open(filepath) as img:
                # Resize and normalize the image
                img = img.resize(size)
                img_array = np.array(img) / 255.0  # Normalize to [0, 1]
                
                images.append(img_array)
                labels.append(row['ClassId'])
        except IOError as e:
            print(f"Error in reading {filepath}: {e}. Skipping.")

    return np.array(images), np.array(labels)

# Paths to CSV files
meta_csv_path = 'path/to/Meta.csv'
train_csv_path = 'path/to/traffic_train.csv'
test_csv_path = 'path/to/traffic_test.csv'
sample_submission_csv_path = 'path/to/traffic_sample_submission.csv'

# Read the CSV files
meta_df = pd.read_csv(meta_csv_path)
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)
sample_submission_df = pd.read_csv(sample_submission_csv_path)

# Base directory for images (adjust as necessary)
base_dir = 'path/to/your/images'

# Load and process training images
train_images, train_labels = load_and_process_images(train_df, base_dir)

# Load and process test images (if needed)
test_images, _ = load_and_process_images(test_df, base_dir)

# Further steps would include splitting the train data for validation,
# building and training a machine learning model,
# and then using the model to make predictions on the test set.

# Note: Replace 'path/to/your/images' and 'path/to' with the actual paths to your image directories and CSV files.

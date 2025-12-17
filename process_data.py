import os
import cv2 # Computer Vision Library
import numpy as np
from skimage.feature import hog  # Histogram of Oriented Gradients (feature extraction method, recognizes edges and their directions)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# loads images from a specified database path, applies data augmentation
# extracts HOG and color histogram features, splits the data into training and testing sets,
# scales the features, and saves the preprocessed data to a pickle file.

Database_path = 'dataset'

CLASSES = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]
img_size = (128, 64)  # Standard image size for resizing

# Data Augmentation function
# creates 3 images from 1 image (original, flipped, rotated), 200% increase in dataset size
def augment_image(image): 
    augmented_images = [image]

    # Horizontal Flip
    # This Teaches model to ignore directionality of objects
    flipped = cv2.flip(image, 1)
    augmented_images.append(flipped)

    # Rotation
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 15, 1) # Rotate by 15 degrees
    rotated = cv2.warpAffine(
        image, M, (cols, rows),
        borderMode=cv2.BORDER_REFLECT
    )
    augmented_images.append(rotated)
    
    return augmented_images

# function that converts an image into a 1D numerical vector
def extract_features(image):
    # 1. Resize (Standard size)
    img_resized = cv2.resize(image, img_size)   # All images must have the same size
    
    # 2. LIGHTWEIGHT HOG FEATURES
    # This reduces feature count by ~85% (from ~3700 to ~500 features)
    # It removes "noise" and focuses on the main shape (cylinder vs box)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        transform_sqrt=True
    )
    
    # 3. COLOR FEATURES
    # Split image vertically into Top, Middle, Bottom to find "Bottle Caps" or "Labels"
    h, _, _ = img_resized.shape
    third = h // 3
    parts = [img_resized[:third, :], img_resized[third:2*third, :], img_resized[2*third:, :]]
    
    color_features = []
    for part in parts:
        hsv_part = cv2.cvtColor(part, cv2.COLOR_BGR2HSV)
        # Histograms for Hue (Color) and Saturation
        hist_h = cv2.calcHist([hsv_part], [0], None, [32], [0, 180]) 
        hist_s = cv2.calcHist([hsv_part], [1], None, [32], [0, 256])
        cv2.normalize(hist_h, hist_h) # makes features scale-independent
        cv2.normalize(hist_s, hist_s)
        color_features.extend(hist_h.flatten())
        color_features.extend(hist_s.flatten())
        
    return np.hstack([hog_features, color_features])

# Load Data
TARGET_COUNT = 500  # target images per class

def load_and_preprocess_data():
    features = []
    labels = []

    for class_id, class_name in enumerate(CLASSES):
        class_path = os.path.join(Database_path, class_name)
        file_list = os.listdir(class_path)
        print(f"Processing {class_name} ({len(file_list)} images)")

        class_features = []
        class_labels = []

        # Load original images
        for file_name in file_list:
            img_path = os.path.join(class_path, file_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            feat = extract_features(img)
            class_features.append(feat)
            class_labels.append(class_id)

        # Use augmentation to reach TARGET_COUNT
        current_count = len(class_features)
        idx = 0
        while current_count < TARGET_COUNT:
            img_path = os.path.join(class_path, file_list[idx % len(file_list)])
            img = cv2.imread(img_path)
            if img is None:
                idx += 1
                continue

            # augment image
            aug_images = augment_image(img)
            for aug_img in aug_images:
                feat = extract_features(aug_img)
                class_features.append(feat)
                class_labels.append(class_id)
                current_count += 1
                if current_count >= TARGET_COUNT:
                    break
            idx += 1

        # Trim if somehow exceeded
        features.extend(class_features[:TARGET_COUNT])
        labels.extend(class_labels[:TARGET_COUNT])

    X = np.array(features)
    y = np.array(labels)
    return X, y

X, y = load_and_preprocess_data()

# Split Data
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Feature Scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Save preprocessed data
with open('data_features.pkl', 'wb') as f:
    pickle.dump((x_train, x_test, y_train, y_test, scaler), f)


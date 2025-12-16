import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

Database_path = 'dataset'

CLASSES = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]
img_size = (128, 64) 

# Data Augmentation function
def augment_image(image): 
    # the increase 200% (3 times the original)
    # list of images: original, flipped, rotated
    augmented_images = [image]

    # Horizontal Flip
    # This Teaches model to ignore directionality of objects
    flipped = cv2.flip(image, 1)
    augmented_images.append(flipped)

    # Rotation
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 15, 1) # Rotate by 15 degrees
    rotated = cv2.warpAffine(image, M, (cols, rows)) 
    augmented_images.append(rotated)
    
    return augmented_images

import cv2
import numpy as np
from skimage.feature import hog

# --- OPTIMIZED FEATURE EXTRACTION ---
def extract_features(image):
    # 1. Resize (Standard size)
    img_resized = cv2.resize(image, img_size)
    
    # 2. LIGHTWEIGHT HOG (The Speed Fix)
    # Changed pixels_per_cell from (8,8) to (16,16)
    # This reduces feature count by ~85% (from ~3700 to ~500 features)
    # It removes "noise" and focuses on the main shape (cylinder vs box)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    hog_features = hog(gray, orientations=9, pixels_per_cell=(16, 16), 
                       cells_per_block=(2, 2), block_norm='L2-Hys', transform_sqrt=True)
    
    # 3. SPATIAL COLOR (The Accuracy Fix)
    # Split image into Top, Middle, Bottom to find "Bottle Caps" or "Labels"
    h, w, _ = img_resized.shape
    third = h // 3
    parts = [img_resized[:third, :], img_resized[third:2*third, :], img_resized[2*third:, :]]
    
    color_features = []
    for part in parts:
        hsv_part = cv2.cvtColor(part, cv2.COLOR_BGR2HSV)
        # Histograms for Hue (Color) and Saturation
        hist_h = cv2.calcHist([hsv_part], [0], None, [32], [0, 180]) 
        hist_s = cv2.calcHist([hsv_part], [1], None, [32], [0, 256])
        cv2.normalize(hist_h, hist_h)
        cv2.normalize(hist_s, hist_s)
        color_features.extend(hist_h.flatten())
        color_features.extend(hist_s.flatten())
        
    return np.hstack([hog_features, color_features])

# Load Data
def load_and_preprocess_data():
    features = []
    labels = []

    for class_id,class_name in enumerate(CLASSES):
        class_path = os.path.join(Database_path, class_name)

        print(f"Processing class: {class_name}")
        for file_name in os.listdir(class_path):
            img_path = os.path.join(class_path, file_name)

            # Read image
            img = cv2.imread(img_path)
            if img is None:
                continue
            # Apply data augmentation
            augmented_images = augment_image(img)
            for aug_img in augmented_images:
                feature = extract_features(aug_img)
                features.append(feature)
                labels.append(class_id) # 0 to 5
        
    X = np.array(features)
    y = np.array(labels)
    return X, y

X, y = load_and_preprocess_data()

# Split Data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Save preprocessed data
with open('data_features.pkl', 'wb') as f:
    pickle.dump((x_train, x_test, y_train, y_test, scaler), f)
# The script loads images from a specified database path, applies data augmentation (flipping and rotation),
# extracts HOG and color histogram features, splits the data into training and testing sets,
# scales the features, and saves the preprocessed data to a pickle file.

import os
import cv2
import numpy as np
from skimage.feature import hog  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

#configuration
Database_path = 'dataset'

CLASSES = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]
img_size = (128, 64)  # Standard image size for resizing

#data Augmentation function
def augment_image(image): 
    augmented_images = [image]
    # Horizontal Flip
    flipped = cv2.flip(image, 1)
    augmented_images.append(flipped)
    # Rotation
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 15, 1) #15 degrees Rotation
    rotated = cv2.warpAffine(image, M, (cols, rows),borderMode=cv2.BORDER_REFLECT)
    augmented_images.append(rotated)
    return augmented_images

#converts an image into a 1D numerical vector
def extract_features(image):
    # 1. Resize 
    img_resized = cv2.resize(image, img_size)   
    # 2. LIGHTWEIGHT HOG FEATURES
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2,2),
        block_norm='L2-Hys',
        transform_sqrt=True
    )
    
    # 3.cOLOR FEATURES
    h, _, _ = img_resized.shape
    third = h // 3
    parts = [img_resized[:third, :], img_resized[third:2*third, :], img_resized[2*third:, :]]
    
    color_features = []
    for part in parts:
        hsv_part = cv2.cvtColor(part, cv2.COLOR_BGR2HSV)
        #histograms for Hue (Color) and Saturation
        hist_h = cv2.calcHist([hsv_part], [0], None, [32], [0, 180]) 
        hist_s = cv2.calcHist([hsv_part], [1], None, [32], [0, 256])
        cv2.normalize(hist_h, hist_h) # makes features scale-independent
        cv2.normalize(hist_s, hist_s)
        color_features.extend(hist_h.flatten())
        color_features.extend(hist_s.flatten())
    return np.hstack([hog_features, color_features])

#load Data
TARGET_COUNT = 500  # target images per class

def load_and_preprocess_data():
    features = []
    labels = []

    for class_id, class_name in enumerate(CLASSES):
        class_path = os.path.join(Database_path, class_name)
        file_list = os.listdir(class_path)
        print(f"Processing {class_name}")

        class_features = []
        class_labels = []

        # Load original images
        for file_name in file_list:
            img_path = os.path.join(class_path, file_name)
            img = cv2.imread(img_path)
            if img is None:
                continue              # skip unreadable images
            feat = extract_features(img)
            class_features.append(feat)
            class_labels.append(class_id)

        #use augmentation to reach TARGET_COUNT
        current_count = len(class_features)
        idx = 0
        while current_count < TARGET_COUNT:
            img_path = os.path.join(class_path, file_list[idx % len(file_list)])
            img = cv2.imread(img_path)
            if img is None:
                idx += 1
                continue

            aug_images = augment_image(img)   #augment image
            for aug_img in aug_images:
                feat = extract_features(aug_img)
                class_features.append(feat)
                class_labels.append(class_id)
                current_count += 1
                if current_count >= TARGET_COUNT:
                    break
            idx += 1

        features.extend(class_features[:TARGET_COUNT]) #Trim if excess
        labels.extend(class_labels[:TARGET_COUNT])

    X = np.array(features)
    y = np.array(labels)
    return X, y

X, y = load_and_preprocess_data()

#split Data to Train and Test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Feature Scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Save preprocessed data to a pickle file
with open('data_features.pkl', 'wb') as f:
    pickle.dump((x_train, x_test, y_train, y_test, scaler), f)


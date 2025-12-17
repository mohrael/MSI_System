import os
import cv2
import pickle
import numpy as np
from process_data import extract_features

# CONFIGURATION
IMAGE_FOLDER = "test_images"   
MODEL_PATH = "best_model.pkl"
CLASSES = ["glass", "paper", "cardboard", "plastic", "metal", "trash", "Unknown"]


# Load trained model
with open(MODEL_PATH, "rb") as f:
    model, scaler, model_type = pickle.load(f)

print(f"Loaded model: {model_type}")

# Loop over images
i=1
for file_name in os.listdir(IMAGE_FOLDER):

    img_path = os.path.join(IMAGE_FOLDER, file_name)

    # Skip non-image files
    if not file_name.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read: {file_name}")
        continue

    # Feature extraction
    feature = extract_features(img)
    feature = scaler.transform([feature])[0]

    # Prediction 
    if model_type == "SVM":
        probs = model.predict_proba(feature.reshape(1, -1))[0]
        confidence = np.max(probs)
        if confidence < 0.4:
            pred = 6
        else:
            pred = np.argmax(probs)

    else:  # KNN
        distances, _ = model.kneighbors(feature.reshape(1, -1))
        mean_dist = np.mean(distances)
        if mean_dist >  1.2:
            pred = 6
            confidence = 0.0
        else:
            pred = model.predict(feature.reshape(1, -1))[0]
            confidence = 1.0 / (1.0 + mean_dist)

    label = CLASSES[pred]

    # Print Results
    print(f"{i}- {file_name:25s} : {label:10s} | Confidence: {confidence:.2f}")
    i+=1
    # VISUALIZATION
    cv2.putText(
        img, f"{label} ({confidence:.2f})",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0) if pred != 6 else (0, 0, 255),
        2
    )

    cv2.imshow("Prediction", img)
    cv2.waitKey(0)   # Press any key for next image

cv2.destroyAllWindows()

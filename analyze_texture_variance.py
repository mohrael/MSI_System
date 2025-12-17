import os
import cv2
import numpy as np

# same function you use in testing/live
def texture_variance(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.var(gray)

CLASSES = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]
DATASET_PATH = "dataset"

for cls in CLASSES:
    path = os.path.join(DATASET_PATH, cls)
    vals = []

    for f in os.listdir(path)[:50]:  # first 50 images only
        img_path = os.path.join(path, f)
        img = cv2.imread(img_path)
        if img is not None:
            vals.append(texture_variance(img))

    print(f"{cls:10s} | mean={np.mean(vals):.1f} | min={np.min(vals):.1f} | max={np.max(vals):.1f}")

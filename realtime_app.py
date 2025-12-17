import cv2
import pickle
import numpy as np
from process_data import extract_features

CLASSES = ["Glass", "Paper", "Cardboard", "Plastic", "Metal", "Trash", "Unknown"]

# Load model
with open('best_model.pkl', 'rb') as f:
    model, scaler, model_type = pickle.load(f)

cap = cv2.VideoCapture(0) # open camera

while True:
    ret, frame = cap.read() # read frame
    if not ret: # stops if frame not read
        break

    h, w, _ = frame.shape
    y1 = int(0.12 * h)
    y2 = int(0.85 * h)
    x1 = int(0.12 * w)
    x2 = int(0.85 * w)
    crop = frame[  # crop center of the frame
        int(y1):int(y2),
        int(x1):int(x2)
    ]

    feature = extract_features(crop)   # HOG features + color histograms: 1D vector
    feature = scaler.transform([feature])[0] 

    if model_type == 'SVM':
        probs = model.predict_proba(feature.reshape(1, -1))[0]
        conf = np.max(probs)
    
        if np.max(probs) < 0.45:
            pred = 6
        else:
            pred = np.argmax(probs)
    else:
        distances, _ = model.kneighbors(feature.reshape(1, -1))
        if np.mean(distances) > 1.1:
            pred = 6
        else:
            pred = model.predict(feature.reshape(1, -1))[0]

    # Draw rectangle around the crop area
    if pred == 6:          # Unknown
        color = (0, 0, 255)  # Red
    else:
        color = (0, 255, 0)  # Green

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = CLASSES[pred]
    
    if model_type == 'SVM':
        cv2.putText(frame, f"{label} ({conf:.2f})",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)
    else:
        cv2.putText(frame, label, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("MSI System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

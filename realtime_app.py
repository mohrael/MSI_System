import cv2
import pickle
import numpy as np
from process_data import extract_features

CLASSES = ["Glass", "Paper", "Cardboard", "Plastic", "Metal", "Trash", "Unknown"]

# Load model
with open('best_model.pkl', 'rb') as f:
    model, scaler, model_type = pickle.load(f)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    crop = frame[
        int(0.2*h):int(0.8*h),
        int(0.2*w):int(0.8*w)
    ]

    feature = extract_features(crop)

    feature = scaler.transform([feature])[0]

    if model_type == 'SVM':
        probs = model.predict_proba(feature.reshape(1, -1))[0]
        conf = np.max(probs)
        if np.max(probs) < 0.4:
            pred = 6
        else:
            pred = np.argmax(probs)
    else:
        distances, _ = model.kneighbors(feature.reshape(1, -1))
        if np.mean(distances) > 1.2:
            pred = 6
        else:
            pred = model.predict(feature.reshape(1, -1))[0]

    label = CLASSES[pred]
    cv2.putText(frame, f"{label} ({conf:.2f})",
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,0), 2)

    cv2.putText(frame, label, (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("MSI System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import numpy as np

UNKNOWN_ID = 6

def predict_with_unknown_svm(model, feature, threshold=0.6):
    probs = model.predict_proba(feature.reshape(1, -1))[0]
    if np.max(probs) < threshold:
        return UNKNOWN_ID
    return np.argmax(probs)

def predict_with_unknown_knn(model, feature, dist_threshold=0.8):
    distances, _ = model.kneighbors(feature.reshape(1, -1))
    if np.mean(distances) > dist_threshold:
        return UNKNOWN_ID
    return model.predict(feature.reshape(1, -1))[0]

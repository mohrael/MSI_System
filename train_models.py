import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC # Support Vector Machine classifier
from sklearn.neighbors import KNeighborsClassifier # K-Nearest Neighbors classifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # evaluate model performance
import seaborn as sns
import pickle # load and save trained models
from sklearn.model_selection import GridSearchCV


# Load the preprocessed data
with open('data_features.pkl','rb') as f:
    x_train, x_test, y_train, y_test, scaler = pickle.load(f)

CLASSES = ["Glass", "Paper", "Cardboard", "Plastic", "Metal", "Trash"]


# Train SVM Model
print("\n" + "="*30)
print("TRAINING SVM")
print("="*30)

# Create the model
# probability=True is REQUIRED for the 'Unknown' class logic later
svm_model = SVC(
    kernel='rbf',
    C=10,
    gamma=0.001,
    class_weight='balanced',
    probability=True,
    random_state=42
)
svm_model.fit(x_train, y_train)

# Evaluate SVM Model
svm_preds = svm_model.predict(x_test)
svm_acc = accuracy_score(y_test, svm_preds)
print(f"SVM Accuracy: {svm_acc*100:.2f}%")

print(classification_report(y_test, svm_preds, target_names=CLASSES))

# Train KNN Model
print("\n" + "="*30)
print("TRAINING KNN")
print("="*30)

# n_neighbors=5 is standard. weights='distance' gives more vote to closer neighbors
knn_model = KNeighborsClassifier(n_neighbors=4, weights='distance')
knn_model.fit(x_train, y_train)

# Evaluate KNN Model
knn_preds = knn_model.predict(x_test)
knn_acc = accuracy_score(y_test, knn_preds)
print(f"KNN Accuracy: {knn_acc*100:.2f}%")
# k-NN often struggles with high-dimensional data compared to SVM


# Compare and save the best model

print("\n" + "="*30)
print("Final Results")
print("="*30)

if svm_acc > knn_acc:
    print(f"SVM won with {svm_acc:.2%} accuracy.")
    best_model = svm_model
    another_model = knn_model
    model_type = 'SVM'
    another_type = 'KNN'
else:
    print(f"KNN won with {knn_acc:.2%} accuracy.")
    best_model = knn_model
    another_model = svm_model
    model_type = 'KNN'
    another_type = 'SVM'

with open('another_model.pkl', 'wb') as f:
    pickle.dump((another_model, scaler, another_type), f)

with open('best_model.pkl', 'wb') as f:
    pickle.dump((best_model, scaler, model_type), f)




print("Starting federated learning script...")
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("Loading breast cancer dataset")
data = load_breast_cancer()
X = data.data
print("Dataset loaded")
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Data scaled")

num_nodes = 4
X_splits = np.array_split(X_train, num_nodes)
y_splits = np.array_split(y_train, num_nodes)

def federated_averaging(models):
    coef = np.mean([model.coef_ for model in models], axis=0)
    intercept = np.mean([model.intercept_ for model in models], axis=0)
    global_model = LogisticRegression(max_iter=1000)
    global_model.coef_ = coef
    global_model.intercept_ = intercept
    global_model.classes_ = models[0].classes_
    return global_model
communication_rounds = 20
global_model = LogisticRegression(max_iter=1000)

for round in range(communication_rounds):
    print(f"Communication round {round+1}/{communication_rounds}")
    local_models = []
    
    for i in range(num_nodes):
        local_model = LogisticRegression(max_iter=1000)
        local_model.fit(X_splits[i], y_splits[i])
        local_models.append(local_model)
    
    global_model = federated_averaging(local_models)

try:
    y_pred = global_model.predict(X_test)
except Exception as e:
    print("Error during prediction:", e)
    raise

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Federated Learning Results")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
central_model = LogisticRegression(max_iter=1000)
central_model.fit(X_train, y_train)
y_pred_central = central_model.predict(X_test)

print("\nCentralized Model Accuracy:", accuracy_score(y_test, y_pred_central))

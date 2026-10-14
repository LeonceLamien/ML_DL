# from sklearn.metrics import balanced_accuracy_score
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import StandardScaler
# import pandas as pd

# # Load the training data
# x_train = pd.read_pickle('A5_2024_xtrain.gz')
# y_train = pd.read_pickle('A5_2024_ytrain.gz')

# # Split the training data into training and validation sets
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train)

# # Preprocess the data (standardization)
# scaler = StandardScaler()
# x_train_scaled = scaler.fit_transform(x_train)
# x_val_scaled = scaler.transform(x_val)

# # Train the classifier (K-NN classifier in this example)
# knn_classifier = KNeighborsClassifier(n_neighbors=3,weights='distance',algorithm="auto",metric="minkowski")
# knn_classifier.fit(x_train_scaled, y_train)

# # Make predictions on the validation set
# y_pred_val = knn_classifier.predict(x_val_scaled)

# # Calculate the BCR on the validation set
# validation_bcr = balanced_accuracy_score(y_val, y_pred_val)
# print("Validation BCR:", validation_bcr)

# # Based on insights gained during model development, estimate the expected BCR for the test set
# # For example, you could use the validation BCR as an estimate
# expected_bcr = validation_bcr
# print("Expected BCR for the test set:", expected_bcr)


from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import csv
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import csv
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

x_train = pd.read_pickle('./A5_2024_xtrain.gz')
y_train = pd.read_pickle('./A5_2024_ytrain.gz')
x_test = pd.read_pickle('./A5_2024_xtest.gz')
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=1)

# scaler = StandardScaler()
# x_train_scaled = scaler.fit_transform(x_train)
# x_test_scaled = scaler.transform(x_test)
# rand_clf = RandomForestClassifier(criterion = 'entropy', max_depth = 10, max_features = 0.5, min_samples_leaf = 3, min_samples_split = 3, n_estimators = 150)
# rand_clf.fit(x_train_scaled, y_train)
# y_pred = rand_clf.predict(x_test_scaled)
# rand_clf_acc = accuracy_score(y_test, y_pred)
# bcr = balanced_accuracy_score(y_test, y_pred)
# print("Balanced Classification Rate (BCR):", bcr)
# print(rand_clf_acc)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
model = SVC()


# Tunning Params
random_grid = {
    'kernel':['linear'],
    'C':[0.01],
    'gamma':[0.0001]
# 'criterion':'entropy',
#  'max_depth': [10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100], # 10, 20, 30, 60, 70, 100,
#  'max_features': [0.2,0.4,0.6],
#  'min_samples_leaf': [2,3,4], # , 4
#  'min_samples_split': [2,3,4], # , 10
#  'n_estimators': [100,130, 150],
#  'criterion':['entropy']
} # , 600, 800, 1000, 1200, 1400, 1600, 1800, 2000

# Implement GridSearchCV
gsc = GridSearchCV(model, random_grid, cv=20) # 10 Cross Validation

# Model Fitting
gsc.fit(x_train_scaled, y_train)

print("\n Best Score is ")
print(gsc.best_score_)

print("\n Best Estinator is ")
print(gsc.best_estimator_)

print("\n Best Parametes are")
print(gsc.best_params_) 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


train_df = pd.read_csv("wine_train.csv")
X_train = train_df.drop("quality", axis=1)
y_train = train_df["quality"]
dt_classifier = DecisionTreeClassifier(random_state=0)
dt_classifier.fit(X_train, y_train)
y_train_pred = dt_classifier.predict(X_train)
# print("Predictions for the training set:")
# print(y_train_pred[:5])
num_nodes = dt_classifier.tree_.node_count
print("Number of nodes of the learned tree:", num_nodes)
# train_accuracy = accuracy_score(y_train, y_train_pred)
# print("Classification accuracy on the training data:", train_accuracy)
train_accuracy = dt_classifier.score(X_train, y_train)
print("Classification accuracy on the training data:", train_accuracy)

print("------------------------------------------------------------")
test_df = pd.read_csv("wine_test.csv")
X_test = test_df.drop("quality", axis=1)
y_test = test_df["quality"]
# dt_classifier = DecisionTreeClassifier(random_state=0)
# dt_classifier.fit(X_test, y_test)
y_test_pred = dt_classifier.predict(X_test)
# print("Predictions for the tes set:")
# print(y_test_pred[0:])
num_nodes = dt_classifier.tree_.node_count
print("Number of nodes of the learned tree:", num_nodes)
test_accuracy = dt_classifier.score(X_test, y_test)
print("Classification accuracy on the test data:", test_accuracy)

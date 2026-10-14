import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import BaggingClassifier
train_df = pd.read_csv("wine_train.csv")
# X_train = train_df.drop("quality", axis=1)
# y_train = train_df["quality"]
test_df = pd.read_csv("wine_test.csv")
X_test = test_df.drop("quality", axis=1)
y_test = test_df["quality"]

def bagging_fit(train_df, nb_trees, seeds):
    tree_list = []
    for i in seeds:
        bootstrap_sample = train_df.sample(n=len(train_df), replace=True, random_state=i)
        X_train = bootstrap_sample.drop(columns=['quality'])
        y_train = bootstrap_sample['quality']
        tree = DecisionTreeClassifier(random_state=0)
        tree.fit(X_train, y_train)
        tree_list.append(tree)
    return tree_list
    #return []  # TODO: replace by your own python code

def bagging_predict(tree_list, x_test):
    predictions = []
    for tree in tree_list:
        predictions.append(tree.predict(X_test))
    label_encoder = LabelEncoder()
    predictions = [label_encoder.fit_transform(pred) for pred in predictions]
    predictions = np.array(predictions)
    majority_votes = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions.astype(int))
    return majority_votes

# Function to perform a single run
def single_run(train_df, test_df, nb_trees):
    seeds = np.random.randint(1000, size=nb_trees)
    tree_list = bagging_fit(train_df, nb_trees, seeds)
    predictions = bagging_predict(tree_list, test_df.drop(columns=['quality']))
    true_labels = test_df['quality']
    accuracy = accuracy_score(true_labels, predictions)
    return accuracy

# Function to perform multiple runs and return the average accuracy
def multiple_runs(train_df, test_df, nb_trees, num_runs=100):
    accuracies = []
    for _ in range(num_runs):
        accuracy = single_run(train_df, test_df, nb_trees)
        accuracies.append(accuracy)
    avg_accuracy = sum(accuracies) / len(accuracies)
    return avg_accuracy

# Values of nb_trees to test
nb_trees_values = [1, 5, 10, 20, 30, 40, 50]

# Perform multiple runs for each value of nb_trees
avg_accuracies = []
for nb_trees in nb_trees_values:
    avg_accuracy = multiple_runs(train_df, test_df, nb_trees)
    avg_accuracies.append(avg_accuracy)

# Plot the curve
plt.plot(nb_trees_values, avg_accuracies, marker='o')
plt.xlabel('Number of Trees')
plt.ylabel('Average Test Set Accuracy')
plt.title('Evolution of Test Set Accuracy with Number of Trees in Bagging')
plt.grid(True)
plt.show()

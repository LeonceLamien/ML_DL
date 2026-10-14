from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import BaggingClassifier
import pandas as pd

train_df = pd.read_csv("wine_train.csv")
X_train = train_df.drop("quality", axis=1)
y_train = train_df["quality"]
test_df = pd.read_csv("wine_test.csv")
X_test = test_df.drop("quality", axis=1)
y_test = test_df["quality"]

def bagging_fit(train_df, nb_trees, seeds):
    tree_list = []
    for i in range(nb_trees):
        bootstrap_sample = train_df.sample(
            n=len(train_df), replace=True, random_state=seeds[i])
        X_train = bootstrap_sample.drop(columns=['quality'])
        y_train = bootstrap_sample['quality']
        tree = DecisionTreeClassifier(
            random_state=0, min_impurity_decrease=0.0045, max_depth=10, min_samples_split=30)
        tree.fit(X_train, y_train)
        train_accuracy = tree.score(X_test, y_test)
        print("hggh", train_accuracy)
    return tree_list


# Set the number of trees for bagging
nb_trees = 4

# Set seeds for reproducibility
seeds = [42, 100, 123, 150]  # example seeds
bagging_fit(train_df, nb_trees, seeds)
# Tune meta-parameters for Random Forest
# Note: Tune parameters such as n_estimators, max_depth, min_samples_split, etc.
my_best_forest = RandomForestClassifier(random_state=0, max_depth=10, min_samples_split=30,min_impurity_decrease=0.0045)
my_best_forest.fit(X_train, y_train)
y_train_pred = my_best_forest.predict(X_train)
train_accuracy = my_best_forest.score(X_test, y_test)
print("my_best_forest", train_accuracy)

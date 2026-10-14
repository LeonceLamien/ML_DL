import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# Load the training and test data
train_df = pd.read_csv("wine_train.csv")
test_df = pd.read_csv("wine_test.csv")

# Extract features and target variable
X_train = train_df.drop("quality", axis=1)
y_train = train_df["quality"]
X_test = test_df.drop("quality", axis=1)
y_test = test_df["quality"]

# Define a range of values for min_impurity_decrease
min_impurity_decrease_values = [1,0,0.1,0.01,0.001,0.15,0.19,0.0001,0.00001,0.05,0.0005,0.005,0.0045]

# Initialize a DataFrame to store results
frame = pd.DataFrame(columns=["min_impurity_decrease", "NodeCount", "TrainAcc", "TestAcc"])

for min_impurity_decrease in min_impurity_decrease_values:
    # Initialize decision tree classifier with min_impurity_decrease value
    # dt_classifier = DecisionTreeClassifier(min_impurity_decrease=min_impurity_decrease, random_state=0)
    # dt_classifier = DecisionTreeClassifier(random_state=0,min_impurity_decrease=0.0045 )  # TODO: replace by your own python code
    # Train the decision tree classifier
     #min_samples_leaf=20,
    # dt_classifier = DecisionTreeClassifier(random_state=0,criterion='gini', max_depth=15,min_impurity_decrease=0.0045,max_features=11, min_samples_split=9   )  # TODO: replace by your own python code
    dt_classifier = DecisionTreeClassifier(random_state=0)  # TODO: replace by your own python code
    dt_classifier.fit(X_train, y_train)
    
    # dt_classifier = DecisionTreeClassifier(random_state=0,max_features=5,criterion='gini', max_depth=15, min_samples_leaf=20, min_samples_split=30,min_impurity_decrease=0.0045  )  # TODO: replace by your own python code

    # Predict labels for training and test sets
    y_train_pred = dt_classifier.predict(X_train)
    y_test_pred = dt_classifier.predict(X_test)

    # Calculate number of nodes in the learned tree
    num_nodes = dt_classifier.tree_.node_count

    # Calculate classification accuracy on training and test data
    train_accuracy = dt_classifier.score(X_train, y_train)
    test_accuracy = dt_classifier.score(X_test, y_test)

    # Store results in the DataFrame
    frame = frame._append({"min_impurity_decrease": min_impurity_decrease,
                          "NodeCount": num_nodes,
                          "TrainAcc": train_accuracy,
                          "TestAcc": test_accuracy}, ignore_index=True)

# Print the DataFrame
print(frame)

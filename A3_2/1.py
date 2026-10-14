import pandas as pd

train_df = pd.read_csv("college_train.csv")
test_df = pd.read_csv("college_test.csv")
from sklearn.tree import DecisionTreeClassifier

random_state = 0
train_sample = train_df.sample(frac=0.05, random_state=random_state)
test_sample = test_df.sample(n=100, random_state=random_state)
X_train = train_sample.drop(columns=['Outcome'])
y_train = train_sample['Outcome']
X_test = test_sample.drop(columns=['Outcome'])
y_test = test_sample['Outcome']
clf = DecisionTreeClassifier(random_state=random_state)
clf.fit(X_train, y_train)
test_acc = clf.score(X_test, y_test)

print("Test Accuracy:", test_acc)

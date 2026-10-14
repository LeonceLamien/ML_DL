import pandas as pd

train_df = pd.read_csv("college_train.csv")
test_df = pd.read_csv("college_test.csv")
from sklearn.tree import DecisionTreeClassifier

test_accs = []

random_state_train = 0  
random_state_test = 0   

num_test_folds = 100

train_sample = train_df.sample(frac=0.05, random_state=random_state_train)

X_train = train_sample.drop(columns=['Outcome'])
y_train = train_sample['Outcome']

clf = DecisionTreeClassifier(random_state=random_state_train)

clf.fit(X_train, y_train)

for i in range(num_test_folds):
    test_fold = test_df.sample(n=100, random_state=random_state_test)
    
    X_test_fold = test_fold.drop(columns=['Outcome'])
    y_test_fold = test_fold['Outcome']
    
    test_acc = clf.score(X_test_fold, y_test_fold)
    
    test_accs.append(test_acc)
    
    random_state_test += 1

mean_test_acc = sum(test_accs) / num_test_folds

print("Mean Test Accuracy:", mean_test_acc)

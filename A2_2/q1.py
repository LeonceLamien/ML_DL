from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import accuracy_score

train_df = pd.read_csv("diabetes_train.csv")
test_df = pd.read_csv("diabetes_test.csv")

X_train = train_df.drop("Diabetes", axis=1)
y_train = train_df["Diabetes"]
X_test = test_df.drop("Diabetes", axis=1)
y_test = test_df["Diabetes"]

cv_acc = 0.0

num_folds = 10
kf = KFold(n_splits=num_folds, shuffle=False)
validation_accuracies = []
for train_index, val_index in kf.split(train_df):
    X_train_fold, X_val_fold = train_df.iloc[train_index].drop(columns=['Diabetes']), train_df.iloc[val_index].drop(columns=['Diabetes'])
    y_train_fold, y_val_fold = train_df.iloc[train_index]['Diabetes'], train_df.iloc[val_index]['Diabetes']
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train_fold, y_train_fold)
    val_accuracy = svm_model.score(X_val_fold, y_val_fold)
    validation_accuracies.append(val_accuracy)

cv_acc = np.mean(validation_accuracies)


# '''
# q2
# '''

optimal_degree = 4
num_folds = 10
kf = KFold(n_splits=num_folds, shuffle=False)
validation_accuracies = []
for train_index, val_index in kf.split(train_df):
    X_train_fold, X_val_fold = train_df.iloc[train_index].drop(columns=['Diabetes']), train_df.iloc[val_index].drop(columns=['Diabetes'])
    y_train_fold, y_val_fold = train_df.iloc[train_index]['Diabetes'], train_df.iloc[val_index]['Diabetes']
    svm_model = SVC(kernel='poly', degree=optimal_degree)
    svm_model.fit(X_train_fold, y_train_fold)
    val_accuracy = svm_model.score(X_val_fold, y_val_fold)
    validation_accuracies.append(val_accuracy)

cv_acc = np.mean(validation_accuracies)


'''
q3
'''


optimal_degree = 4  # Replace with the optimal degree you found

# Define the number of folds for cross-validation
num_folds = 10

# Initialize the KFold object
kf = KFold(n_splits=num_folds, shuffle=False)

# Initialize an array to store the validation accuracies for each fold
validation_accuracies = []

# Iterate over the folds
for train_index, val_index in kf.split(train_df):
    # Split the data into training and validation sets for this fold
    X_train_fold, X_val_fold = train_df.iloc[train_index].drop(columns=['Diabetes']), train_df.iloc[val_index].drop(columns=['Diabetes'])
    y_train_fold, y_val_fold = train_df.iloc[train_index]['Diabetes'], train_df.iloc[val_index]['Diabetes']
    
    # Initialize the SVM model with polynomial kernel and optimal degree
    svm_model = SVC(kernel='sigmoid',C=20)
    # svm_model = SVC(kernel='rbf',C=10)
    # svm_model = SVC(kernel='poly', degree=optimal_degree,coef0=9)
    # svm_model = SVC(kernel='poly', degree=optimal_degree,coef0=1,shrinking = False, probability=True,break_ties=True)
    
    # Train the SVM model on the training data for this fold
    svm_model.fit(X_train_fold, y_train_fold)
    
    # Evaluate the SVM model on the validation data for this fold
    val_accuracy = svm_model.score(X_val_fold, y_val_fold)
    
    # Store the validation accuracy for this fold
    validation_accuracies.append(val_accuracy)

# Compute the mean cross-validation accuracy
cv_acc = np.mean(validation_accuracies)

# Print the cross-validation accuracy
print("Cross-validation accuracy:", cv_acc)

'''
q5
'''
linear_svm = SVC(kernel="linear")
linear_svm.fit(X_train, y_train)
linear_test_accuracy = accuracy_score(y_test, linear_svm.predict(X_test))

# Polynomial SVM model (from Question 3)
poly_svm = SVC(kernel="poly",degree=optimal_degree,coef0=9)
poly_svm.fit(X_train, y_train)
poly_test_accuracy = accuracy_score(y_test, poly_svm.predict(X_test))

# RBF SVM model (from Question 4)
rbf_svm = SVC(kernel="rbf", C=10)
rbf_svm.fit(X_train, y_train)
rbf_test_accuracy = accuracy_score(y_test, rbf_svm.predict(X_test))

# Sigmoid SVM model (from Question 4)
sigm_svm = SVC(kernel="sigmoid",C=20)
sigm_svm.fit(X_train, y_train)
sigm_test_accuracy = accuracy_score(y_test, sigm_svm.predict(X_test))

# Determine the best performing model
test_accuracies = {
    "linear": linear_test_accuracy,
    "poly": poly_test_accuracy,
    "rbf": rbf_test_accuracy,
    "sigmoid": sigm_test_accuracy
}

best_kernel = max(test_accuracies, key=test_accuracies.get)
kernel_choice = best_kernel


def perceptron(x, y, b, w_init, eta, epoch, decay=False):

    y = 2 * y - 1
    w = np.array(w_init)  
    x = np.hstack((np.ones((x.shape[0], 1)), x))
    
    k = 0
    
    while k < epoch:
        k += 1
        
        i = (k - 1) % x.shape[0]
        
        margin = np.dot(w.T, x[i]) * y[i]
        
        if margin <= b:
            eta_k = eta / k if decay else eta
            w += eta_k * y[i] * x[i]
    
    return w

def simple_decay(k):
    return 0.1 / (k + 1)
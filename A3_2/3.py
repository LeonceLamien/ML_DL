from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from scipy.stats import t
import pandas as pd

train_df = pd.read_csv("college_train.csv")
test_df = pd.read_csv("college_test.csv")
frame = pd.DataFrame(columns=["indiv_test_acc", "CI_lower_bound", "CI_upper_bound", "mean_test_acc", "observed_lower_bound", "observed_upper_bound"])

num_experiments = 20

random_state_experiment = 0

for i in range(num_experiments):
    random_state_train_test = i
    
    train_sample = train_df.sample(frac=0.05, random_state=random_state_train_test)
    
    X_train = train_sample.drop(columns=['Outcome'])
    y_train = train_sample['Outcome']
    
    clf = DecisionTreeClassifier(random_state=random_state_train_test)
    
    clf.fit(X_train, y_train)
    
    single_test_sample = test_df.sample(n=100, random_state=random_state_train_test)
    
    X_single_test = single_test_sample.drop(columns=['Outcome'])
    y_single_test = single_test_sample['Outcome']
    
    indiv_test_acc = clf.score(X_single_test, y_single_test)
    
    std_error = np.sqrt((indiv_test_acc * (1 - indiv_test_acc)) / len(y_single_test))
    
    t_statistic = t.ppf(0.975, df=len(y_single_test)-1)
    
    margin_error = t_statistic * std_error
    
    CI_lower_bound = indiv_test_acc - margin_error
    CI_upper_bound = indiv_test_acc + margin_error
    
    test_accs = []
    
    random_state_test_folds = [(i + 1) * j for j in range(100)]
    
    for random_state_test_fold in random_state_test_folds:
        test_fold = test_df.sample(n=100, random_state=random_state_test_fold)
        
        X_test_fold = test_fold.drop(columns=['Outcome'])
        y_test_fold = test_fold['Outcome']
        
        test_acc = clf.score(X_test_fold, y_test_fold)
        
        test_accs.append(test_acc)
    
    mean_test_acc = np.mean(test_accs)
    
    observed_std_dev = np.std(test_accs, ddof=1)
    
    observed_t_statistic = t.ppf(0.975, df=len(test_accs)-1)
    
    observed_margin_error = observed_t_statistic * (observed_std_dev / np.sqrt(len(test_accs)))
    
    observed_lower_bound = mean_test_acc - observed_margin_error
    observed_upper_bound = mean_test_acc + observed_margin_error
    
    frame.loc[i] = [indiv_test_acc, CI_lower_bound, CI_upper_bound, mean_test_acc, observed_lower_bound, observed_upper_bound]

print(frame)

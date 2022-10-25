# spam-classification
spam-classification using simple perceptron classifier

# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data'
names = ['word_freq_make',
'word_freq_address',
'word_freq_all',
'word_freq_3d',
'word_freq_our',
'word_freq_over',
'word_freq_remove',
'word_freq_internet',
'word_freq_order',
'word_freq_mail',
'word_freq_receive',
'word_freq_will',
'word_freq_people',
'word_freq_report',
'word_freq_addresses',
'word_freq_free',
'word_freq_business',
'word_freq_email',
'word_freq_you',
'word_freq_credit',
'word_freq_your',
'word_freq_font',
'word_freq_000',
'word_freq_money',
'word_freq_hp',
'word_freq_hpl',
'word_freq_george',
'word_freq_650',
'word_freq_lab',
'word_freq_labs',
'word_freq_telnet',
'word_freq_857',
'word_freq_data',
'word_freq_415',
'word_freq_85',
'word_freq_technology',
'word_freq_1999',
'word_freq_parts',
'word_freq_pm',
'word_freq_direct',
'word_freq_cs',
'word_freq_meeting',
'word_freq_original',
'word_freq_project',
'word_freq_re',
'word_freq_edu',
'word_freq_table',
'word_freq_conference',
'char_freq_;',
'char_freq_(',
'char_freq_[',
'char_freq_!',
'char_freq_$',
'char_freq_#',
'capital_run_length_average',
'capital_run_length_longest',
'capital_run_length_total',
'spam']

# loading data
df = pd.read_csv(url, names = names)
df
df.describe()
df.isnull().sum()

# assigning target and features
X = df.drop('spam', axis = True)
y = df["spam"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# create the perceptron instance
ppn = Perceptron(max_iter = 100, eta0 = 0.1, random_state = 0)

#fit the model for standardized data
ppn.fit(X_train_std, y_train)

# make prediction
y_pred = ppn.predict(X_test_std)

# we can measure the performance using accuracy score
print(accuracy_score(y_test,y_pred))

Hyperparameter tuning
# grid search learning rate for the perceptron

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=0)

#define grid
grid = dict()
grid['eta0'] = [0.0001, 0.001, 0.01, 0.1, 1.0]

# define search
search = GridSearchCV(ppn, grid, scoring='accuracy', cv=cv, n_jobs=-1)

# perform the search
results = search.fit(X_train_std, y_train)
print('Mean Accuracy: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)
means = results.cv_results_['mean_test_score']
params = results.cv_results_['params']
for mean, param in zip(means, params):
    print(">%.3f with: %r" % (mean, param))
    
# grid search total epochs for the perceptron

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=0)

# define grid
grid = dict()
grid['max_iter'] = [1, 10, 100, 1000, 10000]

# define search
search = GridSearchCV(ppn, grid, scoring='accuracy', cv=cv, n_jobs=-1)

# perform the search
results = search.fit(X_train_std, y_train)
print('Mean Accuracy: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)
means = results.cv_results_['mean_test_score']
params = results.cv_results_['params']
for mean, param in zip(means, params):
    print(">%.3f with: %r" % (mean, param))

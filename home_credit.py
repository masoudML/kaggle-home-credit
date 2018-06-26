
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

#train = pd.read_csv('application_train.csv')
#bureau_balance = pd.read_csv('bureau_balance.csv')
#bureau = pd.read_csv('bureau.csv')
#data = pd.merge(train, bureau, how='left', on=['SK_ID_CURR'])

fields = ['SK_ID_CURR', 'TARGET', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL',
          'AMT_CREDIT', \
          'AMT_REQ_CREDIT_BUREAU_QRT']
train = pd.read_csv('application_train.csv', usecols=fields)
train_y = train['TARGET']
train_x = train.drop(['SK_ID_CURR', 'TARGET'], axis=1)
train_x['CODE_GENDER'].fillna('F' if np.random.rand() > 0.5 else 'M' , inplace=True)
train_x['CODE_GENDER'] = train_x['CODE_GENDER'].map({'F': 1, 'M': 0, 'XNA': 1 if np.random.rand() > 0.5 else 0})
train_x['FLAG_OWN_CAR'] = train_x['FLAG_OWN_CAR'].map({'N': 0, 'Y': 1})
train_x['FLAG_OWN_REALTY'] = train_x['FLAG_OWN_REALTY'].map({'N': 0, 'Y': 1})
train_x['AMT_REQ_CREDIT_BUREAU_QRT'].fillna(0, inplace=True)

test = pd.read_csv('application_train.csv', usecols=fields)
test_y = train['TARGET']
test_x = train.drop(['SK_ID_CURR', 'TARGET'], axis=1)
test_x['CODE_GENDER'].fillna('F' if np.random.rand() > 0.5 else 'M' , inplace=True)
test_x['CODE_GENDER'] = test_x['CODE_GENDER'].map({'F': 1, 'M': 0, 'XNA': 1 if np.random.rand() > 0.5 else 0})
test_x['FLAG_OWN_CAR'] = test_x['FLAG_OWN_CAR'].map({'N': 0, 'Y': 1})
test_x['FLAG_OWN_REALTY'] = test_x['FLAG_OWN_REALTY'].map({'N': 0, 'Y': 1})
test_x['AMT_REQ_CREDIT_BUREAU_QRT'].fillna(0, inplace=True)


logR = LogisticRegression(C=1e5)
logR.fit(train_x,train_y)

score = logR.score(test_x, test_y)
print(score)




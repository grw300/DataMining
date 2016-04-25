import multiprocessing

import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV


def xgb_apply(estimators, X):
    X = estimators_[0, 0]._validate_X_predict(X, check_input=True)

    # n_classes will be equal to 1 in the binary classification or the
    # regression case.
    n_estimators, n_classes = self.estimators_.shape
    leaves = np.zeros((X.shape[0], n_estimators, n_classes))

    for i in range(n_estimators):
        for j in range(n_classes):
            estimator = self.estimators_[i, j]
            leaves[:, i, j] = estimator.apply(X, check_input=False)

    return leaves

kagTrainDat = pd.read_csv('~/Documents/git/DataMining/train.csv')
kagTestDat = pd.read_csv('~/Documents/git/DataMining/test.csv')

y_train = kagTrainDat['ACTION']
X_train = kagTrainDat.ix[:, kagTrainDat.columns != 'ACTION']

X_test = kagTestDat.ix[:, kagTestDat.columns != 'id']

D_train = xgb.DMatrix(X_train.values, label=y_train.values)
D_test = xgb.DMatrix(X_test.values)


# specify parameters via map, definition are same as c++ version
param = {'max_depth': 7, 'eta': 0.1, 'silent': 1, 'min_child_weight': 1, 'n_estimators': 280,
         'objective': 'binary:logistic', 'eval_metric': 'auc'}

# specify validations set to watch performance
watchlist = [(D_train, 'train')]
num_round = 10000
evals_result = {}
bst = xgb.train(param, D_train, num_round, watchlist, early_stopping_rounds=500,
                evals_result=evals_result, verbose_eval=False)

y_test = bst.predict(D_test)

param_skl = {'max_delta_step': 0, 'learning_rate': 0.1, 'colsample_bytree': 0.5, 'silent': True, 'min_child_weight': 1,
             'scale_pos_weight': 1, 'max_depth': 7, 'seed': 0, 'subsample': 0.7, 'objective': 'binary:logistic',
             'nthread': multiprocessing.cpu_count(), 'reg_lambda': 1, 'reg_alpha': 0, 'missing': None,
             'base_score': 0.5, 'n_estimators': 280, 'colsample_bylevel': 1, 'gamma': 0.2}

xgb = xgb.XGBClassifier(**param_skl)

xgb_enc = OneHotEncoder(handle_unknown='ignore')
xgb_lm = LogisticRegression()
xgb.fit(X_train, y_train)
xgb_enc.fit(X_train)
xgb_lm.fit(xgb_enc.transform(X_train), y_train)

y_test_enc = xgb_lm.predict_proba(xgb_enc.transform(X_test))[:, 1]

y_sub = (y_test + y_test_enc) / 2.0

submission = pd.Series(data=y_sub, name='Action', index=kagTestDat['id'])

submission.to_csv("~/Documents/git/DataMining/submission_xgboost.csv",
                  index=True,
                  sep=',',
                  header=True)


#print(bst.get_dump())


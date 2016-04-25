import multiprocessing

import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

if __name__ == "__main__":

    kagTrainDat = pd.read_csv('~/Documents/git/DataMining/train.csv')
    kagTestDat = pd.read_csv('~/Documents/git/DataMining/test.csv')

    y_train = kagTrainDat['ACTION']
    X_train = kagTrainDat.ix[:, kagTrainDat.columns != 'ACTION']

    X_test = kagTestDat.ix[:, kagTestDat.columns != 'id']

    D_train = xgb.DMatrix(X_train.values, label=y_train.values)
    D_test = xgb.DMatrix(X_test.values)

    # specify parameters via map, definition are same as c++ version
    param = {'max_depth': 7, 'eta': 0.14, 'silent': 1, 'min_child_weight': 1, 'n_estimators': 358,
             'objective': 'binary:logistic', 'seed': 42, 'eval_metric': 'auc'}

    # specify validations set to watch performance
    watchlist = [(D_train, 'train')]
    num_round = 10000
    evals_result = {}
    bst = xgb.train(param, D_train, num_round, watchlist, early_stopping_rounds=500,
                    evals_result=evals_result, verbose_eval=False)

    y_test = bst.predict(D_test)

    param_skl = {'max_delta_step': 0, 'learning_rate': 0.14, 'colsample_bytree': 0.5, 'silent': True,
                 'min_child_weight': 1, 'scale_pos_weight': 1, 'max_depth': 7, 'seed': 42, 'subsample': 0.7,
                 'objective': 'binary:logistic', 'nthread': multiprocessing.cpu_count(), 'reg_lambda': 1,
                 'reg_alpha': 0, 'missing': None, 'base_score': 0.5, 'n_estimators': 358, 'colsample_bylevel': 1,
                 'gamma': 0.2}

    gbf = xgb.XGBClassifier(**param_skl)

    gbf_enc = OneHotEncoder(handle_unknown='ignore')
    gbf_lm = LogisticRegression()
    gbf.fit(X_train, y_train)
    gbf_enc.fit(X_train)
    gbf_lm.fit(gbf_enc.transform(X_train), y_train)

    y_test_enc = gbf_lm.predict_proba(gbf_enc.transform(X_test))[:, 1]

    y_sub = 0.45*y_test + 0.55*y_test_enc

    submission = pd.Series(data=y_sub, name='Action', index=kagTestDat['id'])

    submission.to_csv("~/Documents/git/DataMining/submission_xgboost.csv",
                      index=True,
                      sep=',',
                      header=True)

    # cv_result = xgb.cv(param, D_train, num_round, nfold=5, early_stopping_rounds=100,
    #                    metrics="auc", verbose_eval=500)

    # param_test = {
    #    'learning_rate': [0.13, 0.14, 0.15, 0.17],
    # }

    # grid_search = GridSearchCV(gbf,
    #                           param_grid=param_test, scoring='roc_auc', n_jobs=multiprocessing.cpu_count(), cv=5)

    # grid_search.fit(X_train, y_train)
    # print(grid_search.best_params_, grid_search.best_score_)

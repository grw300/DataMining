# coding: utf-8
# pylint: disable=unused-import, invalid-name, wrong-import-position
"""For compatibility"""

from __future__ import absolute_import

import sys


PY3 = (sys.version_info[0] == 3)

if PY3:
    # pylint: disable=invalid-name, redefined-builtin
    STRING_TYPES = str,
    py_str = lambda x: x.decode('utf-8')
else:
    # pylint: disable=invalid-name
    STRING_TYPES = basestring,
    py_str = lambda x: x

# pandas
try:
    from pandas import DataFrame
    PANDAS_INSTALLED = True
except ImportError:

    class DataFrame(object):
        """ dummy for pandas.DataFrame """
        pass

    PANDAS_INSTALLED = False

# sklearn
try:
    from sklearn.base import BaseEstimator
    from sklearn.base import RegressorMixin, ClassifierMixin
    from sklearn.preprocessing import LabelEncoder
    from sklearn.cross_validation import KFold, StratifiedKFold
    SKLEARN_INSTALLED = True

    XGBKFold = KFold
    XGBStratifiedKFold = StratifiedKFold
    XGBModelBase = BaseEstimator
    XGBRegressorBase = RegressorMixin
    XGBClassifierBase = ClassifierMixin
except ImportError:
    SKLEARN_INSTALLED = False

    # used for compatiblity without sklearn
    XGBModelBase = object
    XGBClassifierBase = object
    XGBRegressorBase = object
    XGBKFold = None
    XGBStratifiedKFold = None

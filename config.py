from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn import svm
import torch
import os



dir = './'
epochs = 10
# epochs = 1
# testset_size = 2
folds = 5
# folds = 2
num_subs = 50
max_subs = 50
jumps = 10
HIDDEN_DIM = 64
target_len = 2
seq_len = 4


names = [
        # "Nearest Neighbors",
         # "Linear SVM",
         # "RBF SVM",
         # "Gaussian Process",
         # "Decision Tree",
         # "Random Forest",
         "Neural Net"
         # "AdaBoost",
         # "Naive Bayes"
         ]
clfs = [
    # KNeighborsClassifier(3),
    # SVC(kernel="linear", C=0.025, probability=True),
    # SVC(gamma=2, C=1, probability=True),
    # GaussianProcessClassifier(1.0 * RBF(1.0)),
    # DecisionTreeClassifier(max_depth=5),
    # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000)
    # AdaBoostClassifier()
    # GaussianNB()
]
classifiers = list(zip(names, clfs))

regressors = [
    ('SVM', svm.SVR(), svm.SVR())
    # ('SGD Regression', linear_model.SGDRegressor(), linear_model.SGDRegressor()),
    # ('Bayesian Ridge Regression', linear_model.BayesianRidge(), linear_model.BayesianRidge()),
    # ('Lasso Regression', linear_model.LassoLars(), linear_model.LassoLars())
    # ('Passive Aggressive Regression', linear_model.PassiveAggressiveRegressor()),
    # ('Theil-Sen Regression', linear_model.TheilSenRegressor()),
    # ('Linear Regression', linear_model.LinearRegression())
]

groups = {'with': ['222', '223', '224', '225', '226', '227', '228', '242', '243', '244', '246', '247', '248', '101', '104', '106', '112', '114', '116', '117', '12', '121', '122', '124', '131', '132', '134', '136', '137', '14', '142', '144', '146', '15', '152', '156', '162', '164', '17', '2', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '22', '24', '25', '27', '32', '34', '35', '37', '4', '4001', '4002', '4003', '4004', '4005', '4006', '4007', '4008', '4009', '4010', '4011', '4012', '4013', '4014', '4015', '4016', '4017', '4019', '4020', '4021', '4022', '41', '43', '46', '5', '52', '54', '55', '57', '62', '64', '65', '67', '7', '73', '75', '77', '81', '83', '85', '92', '94', '96', '97', 'id', 'group'], 'without': ['202', '203', '204', '205', '206', '207', '208', '212', '213', '214', '215', '216', '217', '218', '232', '233', '234', '235', '236', '237', '238', '103', '105', '11', '111', '113', '115', '123', '125', '13', '133', '135', '138', '143', '145', '147', '151', '153', '155', '16', '161', '171', '21', '23', '26', '3', '3001', '3002', '3003', '3004', '3005', '3006', '3007', '3008', '3009', '3010', '3011', '3012', '3013', '3014', '3015', '3016', '3017', '3018', '3019', '3020', '3021', '3022', '3023', '3024', '3025', '3026', '3027', '3028', '3029', '3030', '3031', '3032', '3033', '3034', '3035', '31', '33', '36', '42', '44', '45', '47', '51', '53', '56', '6', '61', '63', '66', '71', '72', '74', '76', '82', '84', '86', '87', '91', '93', '95', 'id', 'group'], 'ones': ['1001', '1002', '1003', '1004', '1005', '1006', '1007', '1008', '1009', '1010', '1011', '1012', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020', '0021', '0022', '0023', '0024', '0025', '0026']}
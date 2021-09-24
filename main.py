import pickle

import numpy as np

from sklearn.svm import SVC
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from scipy.stats import spearmanr

with open('dblp_dataset.pickle', 'rb') as f:
    data = pickle.load(f)

K = 5 # number of venues
num_seed = 10

data_x = [[] for i in range(K)]
data_ty = [[] for i in range(K)]
for r in data:
    data_x[r[2]].append(r[1])
    data_ty[r[2]].append([r[2], np.log(r[3])])


LR, LRs = [], [[] for i in range(K)]
RF, RFs = [], [[] for i in range(K)]
SV, SVs = [], [[] for i in range(K)]
MLP, MLPs = [], [[] for i in range(K)]
RD, RDs = [], [[] for i in range(K)]
IPW, IPWs = [], [[] for i in range(K)]
IPW_S, IPW_Ss = [], [[] for i in range(K)]

for seed in range(num_seed):
    print(seed, '/', num_seed)
    np.random.seed(seed)
    train_x = []
    train_t = []
    train_y = []
    test_x = []
    test_t = []
    test_y = []
    for i in range(K):
        train_x_i, test_x_i, train_ty_i, test_ty_i = train_test_split(np.array(data_x[i]), np.array(data_ty[i]), test_size=0.3, random_state=seed)
        train_x.append(train_x_i)
        train_t.append(train_ty_i[:, 0].astype(np.int64))
        train_y.append(train_ty_i[:, 1])
        test_x.append(test_x_i)
        test_t.append(test_ty_i[:, 0].astype(np.int64))
        test_y.append(test_ty_i[:, 1])

    print('Random Forest')    
    param = {
     'n_estimators': [10, 100, 1000],
     'max_depth': [2, 3, None],
     'criterion': ['gini', 'entropy'],
     'min_samples_split': [2, 3, 5],
     'min_samples_leaf': [1, 2, 5],
    }
    propensity = RandomForestClassifier(random_state=0)
    model = GridSearchCV(estimator=propensity, param_grid=param, cv=5, n_jobs=-1)
    model.fit(np.vstack(train_x), np.concatenate(train_t))
    propensity = model.best_estimator_
    propensity_hat = propensity.predict_proba(np.vstack(test_x))[np.arange(len(np.vstack(test_x))), np.concatenate(test_t)]
    RF.append(spearmanr(propensity_hat, np.concatenate(test_y))[0])
    for i in range(K):
        RFs[i].append(spearmanr(propensity_hat[np.concatenate(test_t) == i], np.concatenate(test_y)[np.concatenate(test_t) == i])[0])

    print('SVM')
    param = {
     'C': sum([[(10 ** j) * i for i in range(1, 10)] for j in range(-3, 2)], []),
     'kernel': ['poly', 'rbf', 'sigmoid']
    }
    propensity = SVC(probability=True, random_state=0)
    model = GridSearchCV(estimator=propensity, param_grid=param, cv=5, n_jobs=-1)
    model.fit(np.vstack(train_x), np.concatenate(train_t))
    propensity = model.best_estimator_
    propensity_hat = propensity.predict_proba(np.vstack(test_x))[np.arange(len(np.vstack(test_x))), np.concatenate(test_t)]
    SV.append(spearmanr(propensity_hat, np.concatenate(test_y))[0])
    for i in range(K):
        SVs[i].append(spearmanr(propensity_hat[np.concatenate(test_t) == i], np.concatenate(test_y)[np.concatenate(test_t) == i])[0])

    print('MLP')
    param = {
     'hidden_layer_sizes': [(32,), (64,), (128,), (256,)],
     'alpha': [0.01, 0.001, 0.0001, 0.00001],
     'learning_rate_init': [0.001, 0.0001, 0.00001]
    }    
    propensity = MLPClassifier(random_state=0)
    model = GridSearchCV(estimator=propensity, param_grid=param, cv=5, n_jobs=-1)
    model.fit(np.vstack(train_x), np.concatenate(train_t))
    propensity = model.best_estimator_
    propensity_hat = propensity.predict_proba(np.vstack(test_x))[np.arange(len(np.vstack(test_x))), np.concatenate(test_t)]
    MLP.append(spearmanr(propensity_hat, np.concatenate(test_y))[0])
    for i in range(K):
        MLPs[i].append(spearmanr(propensity_hat[np.concatenate(test_t) == i], np.concatenate(test_y)[np.concatenate(test_t) == i])[0])

    print('Logistic regression')
    param = {
     'C': sum([[(10 ** j) * i for i in range(1, 10)] for j in range(-3, 2)], [])
    }
    propensity = LogisticRegression(max_iter=1000, random_state=0)
    model = GridSearchCV(estimator=propensity, param_grid=param, cv=5, n_jobs=-1)
    model.fit(np.vstack(train_x), np.concatenate(train_t))
    propensity = model.best_estimator_
    propensity_hat = propensity.predict_proba(np.vstack(test_x))[np.arange(len(np.vstack(test_x))), np.concatenate(test_t)]
    LR.append(spearmanr(propensity_hat, np.concatenate(test_y))[0])
    for i in range(K):
        LRs[i].append(spearmanr(propensity_hat[np.concatenate(test_t) == i], np.concatenate(test_y)[np.concatenate(test_t) == i])[0])
        
        
    train_p = [[] for i in range(K)]
    for i in range(K):
        for x in train_x[i]:
            train_p[i].append(propensity.predict_proba(x.reshape(1, -1))[0, i])
        train_p[i] = np.array(train_p[i])

    print('Poincare-UW')
    alphas = sum([[(10 ** j) * i for i in range(1, 10)] for j in range(-3, 2)], [])
    models = []
    for i in range(K):
        model = GridSearchCV(estimator=Ridge(random_state=0), param_grid={'alpha': alphas}, cv=5, n_jobs=-1)
        model.fit(train_x[i], train_y[i])
        models.append(model.best_estimator_)
    y_hat = []
    for i in range(K):
        y_hat.append(models[i].predict(test_x[i]))
        RDs[i].append(spearmanr(models[i].predict(test_x[i]), test_y[i])[0])
    RD.append(spearmanr(np.concatenate(y_hat), np.concatenate(test_y))[0])

    print('Poincare')
    alphas = sum([[(10 ** j) * i for i in range(1, 10)] for j in range(-3, 2)], [])
    models_IPW = []
    for i in range(K):
        score = lambda estimator, X, y: estimator.score(X, y, sample_weight=X[:, 0])
        pipe = Pipeline([('ignore', FunctionTransformer(lambda X: X[:, 1:])), ('ridge', Ridge(random_state=0))])
        model = GridSearchCV(estimator=pipe, param_grid={'ridge__alpha': alphas}, cv=5, scoring=score, n_jobs=-1)
        model.fit(np.hstack([1/train_p[i].reshape(-1, 1), train_x[i]]), train_y[i], ridge__sample_weight=1/train_p[i])
        models_IPW.append(model.best_estimator_['ridge'])
    y_hat = []
    for i in range(K):
        y_hat.append(models_IPW[i].predict(test_x[i]))
        IPWs[i].append(spearmanr(models_IPW[i].predict(test_x[i]), test_y[i])[0])
    IPW.append(spearmanr(np.concatenate(y_hat), np.concatenate(test_y))[0])        

    print('Poincare-S')        
    alphas = sum([[(10 ** j) * i for i in range(1, 10)] for j in range(-3, 2)], [])
    score = lambda estimator, X, y: estimator.score(X, y, sample_weight=X[:, 0])
    pipe = Pipeline([('ignore', FunctionTransformer(lambda X: X[:, 1:])), ('ridge', Ridge(random_state=0))])
    model = GridSearchCV(estimator=pipe, param_grid={'ridge__alpha': alphas}, cv=5, scoring=score, n_jobs=-1)
    model.fit(np.hstack([1/train_p[i].reshape(-1, 1), train_t[i].reshape(-1, 1), train_x[i]]), train_y[i], ridge__sample_weight=1/train_p[i])
    model_S_IPW = model.best_estimator_['ridge']
    y_hat = []
    for i in range(K):
        y_hat.append(model_S_IPW.predict(np.hstack([test_t[i].reshape(-1, 1), test_x[i]])))
        IPW_Ss[i].append(spearmanr(model_S_IPW.predict(np.hstack([test_t[i].reshape(-1, 1), test_x[i]])), test_y[i])[0])
    IPW_S.append(spearmanr(np.concatenate(y_hat), np.concatenate(test_y))[0])

def out(name, arr, arrs):
    print(name, format(np.mean(arr), '.3f'), format(np.std(arr), '.3f'))
    for i in range(K):
        print(format(np.mean(arrs[i]), '.3f'), format(np.std(arrs[i]), '.3f'))

out('Linear Regression', LR, LRs)
out('Random Forrest', RF, RFs)
out('Support Vector Machine', SV, SVs)
out('Multi Layer Perceptron', MLP, MLPs)
out('Poincare', IPW, IPWs)
out('Poincare-UW', RD, RDs)
out('Poincare-S', IPW_S, IPW_Ss)

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import argparse
import pyltr
import xgboost as xgb
from xgboost import plot_importance


def train_LR(X,Y,model):
    #delete 16,17,4 is the best score: 0.8237416251503178 ('score:', 0.8625927837128107)
    X = np.asarray(X, dtype=np.float64)
    mask = [True]*X.shape[1]
    mask[16] = False
    mask[17] = False
    mask[14] = False
    #mask[1] = False
    X = X[:, mask]
    Y = np.asarray(Y, dtype=np.int32)
    logreg = LogisticRegression(C=1.0, solver='liblinear', multi_class='ovr')
    clf = logreg.fit(X, Y)
    #joblib.dump(clf, "models/L1_LR_train_model.m")
    print("LR score:",clf.score(X,Y))
    joblib.dump(clf,model)
    scores = clf.predict_proba(X)
    labels = clf.predict(X)
    return scores,labels

def train_RF(X,Y,model):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.int32)
    clf = RandomForestClassifier(n_estimators=500, max_depth=2, random_state=0)
    clf.fit(X, Y)
    print("feature importance,",clf.feature_importances_)
    #joblib.dump(clf, "models/L1_RF_train_model.m")
    joblib.dump(clf, model)
#     ('feature importance,', array([7.82722123e-04, 3.39524151e-02, 9.80291102e-02, 
# 1.46344023e-01,                                                                
#        2.09017260e-04, 5.37960842e-02, 2.04428460e-04, 4.39473404e-03,         
#        5.12405964e-03, 5.58617809e-03, 2.12143594e-02, 4.38502621e-02,         
#        8.26976723e-02, 1.46414209e-01, 1.13165187e-01, 2.41856736e-01,    
#        5.99338024e-04, 1.77946434e-03]))
    # n_estimators=100 max_depth = 2, score: 0.8234266735383382
    print("RF score,",clf.score(X,Y))
    scores = clf.predict_proba(X)
    labels = clf.predict(X)
    return scores,labels

def train_SVM(X,Y,model):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.int32)
    clf = SVC(gamma='auto',probability=True)
    clf.fit(X, Y)
    #joblib.dump(clf, "models/L1_SVM_train_model.m")
    joblib.dump(clf, model)
    #score 0.9969936437038309
    print("SVM score,",clf.score(X,Y))
    scores = clf.predict_proba(X)
    labels = clf.predict(X)
    return scores,labels

def train_LM(X,Y,model):
    Tqids = 0
    metric = pyltr.metrics.NDCG(k=10)
    clf = pyltr.models.LambdaMART(
        metric=metric,
        n_estimators=1000,
        learning_rate=0.02,
        max_features=0.5,
        query_subsample=0.5,
        max_leaf_nodes=10,
        min_samples_leaf=64,
        verbose=1,
    )
    clf.fit(X, Y, Tqids)
    Epred = clf.predict(EX)
    joblib.dump(clf, model)
    print 'Random ranking:', metric.calc_mean_random(Eqids, Y)
    print 'Our model:', metric.calc_mean(Eqids, Y, Epred)

def train_xgboost(X,Y,model):
    params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'num_class': 3,
    'gamma': 0.1,
    'max_depth': 6,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
    }  
    plst = params.items()

    dtrain = xgb.DMatrix(X, Y)
    num_rounds = 500
    clf = xgb.train(plst, dtrain, num_rounds)
    dtrain = xgb.DMatrix(X)
    ans = clf.predict(dtrain)
    cnt1 = 0
    cnt2 = 0
    for i in range(len(Y)):
        if ans[i] == Y[i]:
            cnt1 += 1
        else:
            cnt2 += 1
    print("Score: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))
    joblib.dump(clf, model)


def write_result(scores,labels,outfile):
    outf = open(outfile,"w")
    for i in range(len(ques)):
        # print("scores, ", scores[i])
        # print("predict label, ", clf.predict([X[i],]))
        # print("label,", Y[i])
        for x in [Y[i],"\t",labels[i],"\t",ques[i],"\t",ans[i] \
        ,"\t",dis[i],"\t",scores[i][0],"\t",scores[i][1],'\n']:
            if type(x)!=type(""):
                try:
                    outf.write('{}'.format(x))
                except:
                    outf.write(x.encode('utf-8'))
                    pass
            else:
                outf.write(x)
    outf.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, default="mcql_train_new.json", help='path to json')
    parser.add_argument('--outfile', type=str, default="result.txt", help='path to output result')
    parser.add_argument('--model', type=str, default="models/train_model.m", help='path to model')
    parser.add_argument('--type', type=str, default="SVM", help='model type')
    parser.add_argument('--saveresult',type=bool, default="False",help='save predict result or not')
    args = parser.parse_args()
    inputfile = args.json
    model = args.model
    model_type = args.type
    outfile = args.outfile
    f = open(inputfile,'r')
    data = json.load(f)
    X = []
    Y = []
    ques = []
    ans = []
    dis = []
    for item in data:
        X.append(item[0]) 
        Y.append(item[1])
        ques.append(item[2])
        ans.append(item[3])
        dis.append(item[4])
    if model_type == 'SVM':
        scores,labels = train_SVM(X,Y,model)
    elif model_type == 'LR':
        scores,labels = train_LR(X,Y,model)
    elif model_type == 'RF':
        scores,labels = train_RF(X,Y,model)
    elif model_type == 'LM':
        train_LM(X,Y,model)
    else:
        train_xgboost(X,Y,model)
    if args.saveresult:
        write_result(scores,labels,outfile)
#y = [0,0,1,1,0,] contains wether 0 or 1
#rank contains voc_index with the highest probability
import numpy as np
import json
from math import log
import matplotlib
matplotlib.use('Agg')
import io
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.externals import joblib

def load_data(feature_file):
    X = []
    Y = []
    ques = []
    ans = []
    dis = []
    f = io.open(feature_file,'r',encoding="utf-8")
    res = f.readlines()
    for line in res:
        L = line.strip().split('\t')
        x = L[0].split()
        x = [float(i) for i in x]
        y = int(L[1])
        X.append(x)
        Y.append(y)
        ques.append(L[2])
        ans.append(L[3])
        dis.append(L[4])
    return X,Y,ques,ans,dis

def load_model(model_type,modelpath,filename,outfile,rankfile): #train_model.m, multi_mcql_valid.json, L1_valid_result.txt
    loaded_model = joblib.load(modelpath)
    of = open(outfile,'w')
    rf = open(rankfile,'w')
    X,Y,ques,ans,dis = load_data(filename)
    ques2score = {} #question
    ques2cnt = {} # count number of distractors for each question
    print("load data done!")
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.int32)
    if model_type == 'LR':        
        mask = [True]*X.shape[1]
        mask[16] = False
        mask[17] = False
        mask[14] = False
        #mask[1] = False
        X = X[:, mask]
    scores = loaded_model.predict_proba(X)
    print("score:",loaded_model.score(X,Y))
    print("predict done!")
    for i in range(len(scores)):
        score = scores[i][1]
        question = ques[i]
        if Y[i] == 1:
            if question in ques2cnt.keys():
                ques2cnt[question] += 1
            else:
                ques2cnt[question] = 1
        if question in ques2score.keys():
            ques2score[question].append([i,score])
        else:
            ques2score[question] = [[i,score],]
    for k,t in ques2score.items():
        t = sorted(t, key = lambda x: float(x[1]),reverse= True)
        rank = 0
        for item in t:
            if Y[item[0]] == 1:
               of.write(k.encode('utf-8')+"\t"+dis[item[0]].encode('utf-8')+"\t"+str(rank)+'\n')
            rank += 1
            if rank <= 20:
                # rank y index dis question
                rf.write(str(rank)+'\t'+str(Y[item[0]])+"\t"+str(ques2cnt[k])+'\t'+str(item[0])+'\t'+dis[item[0]].encode('utf-8')+'\t'+k.encode('utf-8')+'\n')
            else:
                break
    of.close()
    rf.close()

if __name__ == '__main__':
    model_name = 'RF'
    load_model(model_name,"models/L2_"+model_name+"_model.m","data/L2_valid_features.txt","models/L2_"+model_name+"_valid_result.txt","models/L2_"+model_name+"_rank_bigram_result.txt")
    model_name = 'LR'
    load_model(model_name,"models/L2_"+model_name+"_model.m","data/L2_valid_features.txt","models/L2_"+model_name+"_valid_result.txt","models/L2_"+model_name+"_rank_bigram_result.txt")

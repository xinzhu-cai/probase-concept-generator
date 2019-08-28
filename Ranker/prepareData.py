import json
import sys 
import pandas as pd
sys.path.append('./')
sys.path.append('.')
import multi_calculate_features
import io
import random
from multiprocessing import Pool
import multiprocessing

def extract_content(s):
    index = s.find("<a>")
    if index-1 > 0:
        return s[:index-1]
    else:
        return s

def prepare_training_data(infile,vocfile,outfile):
    voc_file = io.open(vocfile,'r',encoding='utf-8')
    vocab = []
    for line in voc_file:
        vocab.append(line.strip('\n').lower().replace(' ','_'))
    features = []
    cpu_count = multiprocessing.cpu_count()
    pool = Pool(processes=cpu_count)
    params = []
    print("="*50)
    f = io.open(infile,'r',encoding='utf-8')
    L = f.readlines()
    length = len(L)
    print("#total, ",length)
    lastques = ""
    lastlabel = 1
    dislist = []
    for line in range(length):
        items = L[line].strip().split(' ')
        ques = extract_content(items[1])
        ans = extract_content(items[2])
        dis = extract_content(items[3])
        label = int(items[0])
        dislist.append(dis)
        if ques == lastques and label == 0:
            if lastlabel == 1:
                sslice = random.sample(vocab,10)
                for v in sslice:
                    while v in dislist:
                        v = random.sample(vocab,1)[0]
                    params.append([ques,ans,v,0])
                del dislist[:]
                lastlabel = label
            else:
                continue
        params.append([ques,ans,dis,label])
        lastques = ques
        lastlabel = label
    features = pool.map(multi_calculate_features.cal_26_feature_vec,params)
    with open(outfile,'w') as t:
        for i in range(len(features)):
            for x in features[i][0]:
                t.write(str(x))
                t.write(' ')
            t.write('\t')
            t.write(str(features[i][1]))
            t.write('\t')
            t.write(features[i][2].encode('utf-8'))
            t.write('\t')
            t.write(features[i][3].encode('utf-8'))
            t.write('\t')
            t.write(features[i][4].encode('utf-8'))
            t.write('\n')
    print("finish!")
        
if __name__=="__main__":
    prepare_training_data(\
    './data/train.data',\
    './data/vocab.txt',\
    './data/L2_train_features.txt')
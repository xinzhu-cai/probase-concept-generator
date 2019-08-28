import json
import os
from nltk.parse import stanford

# os.environ['STANFORD_PARSER'] = '/mnt/e/Course/NLP/Toolkits/jars/stanford-parser.jar'
# os.environ['STANFORD_MODELS'] = '/mnt/e/Course/NLP/Toolkits/jars/stanford-parser-3.9.2-models.jar'

# java_path = "/mnt/c/Program Files/Java/jdk1.8.0_111/bin/java.exe"
# os.environ['JAVAHOME'] = java_path

# parser = stanford.StanfordParser(model_path="/mnt/e/Course/NLP/Toolkits/jars/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")

from stanfordcorenlp import StanfordCoreNLP
import logging
import json
from nltk.tree import *

class StanfordNLP:
    def __init__(self, host='http://localhost', port=9000):
        self.nlp = StanfordCoreNLP(host, port=port,
                                   timeout=30000)  # , quiet=False, logging_level=logging.DEBUG)
        self.props = {
            'annotators': 'parse',
            'pipelineLanguage': 'en',
            'outputFormat': 'json'
        }
    def parse(self, sentence):
        return self.nlp.parse(sentence)
    @staticmethod
    def tokens_to_dict(_tokens):
        tokens = defaultdict(dict)
        for token in _tokens:
            tokens[int(token['index'])] = {
                'word': token['word'],
                'lemma': token['lemma'],
                'pos': token['pos'],
                'ner': token['ner']
            }
        return tokens

sNLP = StanfordNLP()

def dump_json(data, outpath):
    print ('Saving to', outpath)
    with open(outpath, 'w') as out:
        json.dump(data, out, indent=4, separators=(',', ': '))

def normalize_string(s):
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            pass
        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass
    
        return False
    s = s.strip('').strip(".")
    if is_number(s):
        s = ''
    return s

def conv_json(filename):
    with open(filename,encoding="utf-8") as f:
        data = json.load(f)
    results = []
    print("*"*30)
    print("start, ",filename)
    print("# of MCQs before",len(data))
    for item in data:

        item['answer'] = normalize_string(item['answer'])

        # delete questions whose answer is number or "all of above"
        if item['answer'] =='' or "all of the above" in item['answer'] or "All of the above" in item['answer']:
            continue
        # delete distractor which is number or "all of above"
        i = 0
        while i < len(item['distractors']):
            item['distractors'][i] = normalize_string(item['distractors'][i])
            if item['distractors'][i]=='' or "all of the above" in item['distractors'][i] or "All of the above" in item['distractors'][i]:
                del item['distractors'][i]
            else:
                i += 1
    
        phases = list(item['distractors'])
        phases.append(item['answer'])
        flag = True

        for p in phases:
            L = p.split()
            #print("length, ",len(L))
            if len(L) == 1:
                continue
            elif len(L)>5:
                flag = False
                break
            else:
                #print ("start parsing, ", p)
                # do stanford parser
                res = sNLP.parse(p)
                res = res[1:].replace(' ','').replace('\n','').split('(')
               # print("parse result, ",res)
                if res[1] != 'VP':
                    flag = False
                    break
        if flag:
            results.append(item)
    # save each file 
    outpath = filename[:-5]+"_filtered.json"
    print("processing done, ",outpath)
    print("# of MCQs after",len(results))
    dump_json(results,outpath)
    return results



if __name__=="__main__":
    filenames = [
    #'MCQ/mcq_total.json',
    #'Gre/gre_total.json',
    #"OpenTriviaQA/trivia_total.json"
    ]
    for i in ['test_neg.json','test.json','train_neg.json','train.json','valid_neg.json','valid.json']:
        path = 'LTR-DG/data/mcql_processed/' + i
        filenames.append(path)
        #path = 'LTR-DG/data/sciq_processed/' + i
        #filenames.append(path)
    results = []
    for file in filenames:
        conv_json(file)
        #results.extend()
    #dump_json(results,'total_filtered.json')
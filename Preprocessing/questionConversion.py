import json
import os
from POSTree import POSTree
from nltk.parse import stanford
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
# convert all the questions to sentences
# the fillin part is represented with '_'*4(____)

def dump_json(data, outpath):
    print ('Saving to', outpath)
    with open(outpath, 'w') as out:
        json.dump(data, out, indent=4, separators=(',', ': '))

def question_to_sentence(sentence):
    parse_res = sNLP.parse(sentence)
    #print(parse_res)
    tree = POSTree(parse_res)
    try:
        res = tree.adjust_order()
        #print(res)
    except:
        #print ("*****************************")
       # print (sentence)
        flag = False
        res = sentence
        if res[-1] == '?':
            for mark in ['what','where','which','when','Where','When','What','Which','how','How']:
                if mark in res:
                    flag = True
                    res = res.replace(mark,' **blank** ')
                    res = res[:-1]
                    break
        if not flag:
            if res[-1] == '.':
                res = res[:-1]
            res += ' **blank** '
    return res

def convert_gre_data(filename):
    with open(filename,encoding="utf-8") as f:
        data = json.load(f)
    print("*"*30)
    print("start, ",filename)
    length = len(data)
    print("# of MCQs before",length)
    for i in range(length):
        #print(data[i]['sentence'].find('  '))
        if data[i]['sentence'].find('________') != -1:
            data[i]['sentence'] = data[i]['sentence'].replace('________',' **blank** ')
        else:
            data[i]['sentence'] = data[i]['sentence'].replace('  ',' **blank** ')
        #print(data[i]['sentence'])
    outpath = filename[:-5]+"_converted.json"
    print("processing done, ",outpath)
    print("# of MCQs after",len(data))
    dump_json(data,outpath)

def convert_mcq_data(filename):
    with open(filename,encoding="utf-8") as f:
        data = json.load(f)
    print("*"*30)
    print("start, ",filename)
    length = len(data)
    print("# of MCQs before",length)
    for i in range(length):
        if data[i]['sentence'].find('________') != -1:
            data[i]['sentence'] = data[i]['sentence'].replace('________',' **blank** ')
        elif data[i]['sentence'][-1] == '?':
            data[i]['sentence'] = question_to_sentence(data[i]['sentence'])
        else:
            data[i]['sentence'] += ' **blank** '
            # flag = False
            # for mark in ['what','where','which','when','Where','When','What','Which']:
            #     if mark in data[i]['sentence']:
            #         flag = True
            #         data[i]['sentence'] = data[i]['sentence'].replace(mark,'**blank**')
            #         break
            # if not flag:
            #     data[i]['sentence'] += ' **blank**'
    outpath = filename[:-5]+"_converted.json"
    print("processing done, ",outpath)
    print("# of MCQs after",len(data))
    dump_json(data,outpath)

def convert_trivia_data(filename):
    with open(filename,encoding="utf-8") as f:
        data = json.load(f)
    print("*"*30)
    print("start, ",filename)
    length = len(data)
    print("# of MCQs before",length)
    for i in range(length):
        if data[i]['answer'] == 'True' or data[i]['answer'] == 'False':
            continue
        if '?' in data[i]['sentence']:
            data[i]['sentence'] = question_to_sentence(data[i]['sentence'])
            # for mark in ['what','where','which','when','Where','When','What','Which']:
            #     if mark in data[i]['sentence']:
            #         flag = True
            #         data[i]['sentence'] = data[i]['sentence'].replace(mark,'**blank**')
            #         break
            # if not flag:
            #     print (data[i]['sentence'])
        elif data[i]['sentence'].find('________') != -1:
            data[i]['sentence'] = data[i]['sentence'].replace('________',' **blank** ')
        else:
            data[i]['sentence'] = data[i]['sentence'][:-1] + ' **blank** '
    outpath = filename[:-5]+"_converted.json"
    print("processing done, ",outpath)
    print("# of MCQs after",len(data))
    dump_json(data,outpath)

def convert_mcql_data(filename):
    with open(filename,encoding="utf-8") as f:
        data = json.load(f)
    print("*"*30)
    print("start, ",filename)
    length = len(data)
    print("# of MCQs before",length)
    for i in range(length):
        data[i]['sentence'] = data[i]['sentence'] + ' **blank** '
        #print(data[i]['sentence'])
    outpath = filename[:-5]+"_converted.json"
    print("processing done, ",outpath)
    print("# of MCQs after",len(data))
    dump_json(data,outpath)

def convert_sciq_data(filename):
    with open(filename,encoding="utf-8") as f:
        data = json.load(f)
    print("*"*30)
    print("start, ",filename)
    length = len(data)
    print("# of MCQs before",length)
    for i in range(length):
        flag = False
        if '?' in data[i]['sentence']:
            data[i]['sentence'] = question_to_sentence(data[i]['sentence'])
            # for mark in ['What','what','which','Which','where','when','Where','When','Who','who','How many','How do','this']:
            #     if mark in data[i]['sentence']:
            #         flag = True
            #         data[i]['sentence'] = data[i]['sentence'].replace(mark,'**blank**')
            #         break
            # if not flag:
            #     data[i]['sentence'] = data[i]['sentence'][:-1] + '**blank**'
    outpath = filename[:-5]+"_converted.json"
    print("processing done, ",outpath)
    print("# of MCQs after",len(data))
    try:
        dump_json(data,outpath)
    except:
        print(data)



if __name__=="__main__":
    filenames = [
    'MCQ/mcq_total_filtered.json',
    'Gre/gre_total_filtered.json',
    "OpenTriviaQA/trivia_total_filtered.json"
    ]
    for x in ['LTR-DG/data/mcql_processed/', 'LTR-DG/data/sciq_processed/']:
        for i in ['test_neg_filtered.json','test_filtered.json','train_neg_filtered.json','train_filtered.json','valid_neg_filtered.json','valid_filtered.json']:
            path = x + i
            filenames.append(path)
    sentence = "If a cat pushes its face against your head, this means what?"
    question_to_sentence(sentence)
    convert_gre_data(filenames[1])
    convert_mcq_data(filenames[0])
    convert_trivia_data(filenames[2])
    for i in range(3,9):
       convert_mcql_data(filenames[i])
    for i in range(9,15):
        convert_sciq_data(filenames[i])

print(question_to_sentence("What is the opportunity cost of purchasing the factory for the first year of operation?"))
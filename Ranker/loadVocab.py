# add neg_samples to total.json, get total_neg.json
import json
import random
import io

def dump_json(data, outpath):
    print ('Saving to', outpath)
    with open(outpath, 'w') as out:
        json.dump(data, out, indent=4, separators=(',', ': '))

if __name__=="__main__":
    voc_file = 'data/vocab.txt'
    of = io.open(voc_file,'w')
    with io.open('data/total.json',encoding="utf-8") as f:
        data = json.load(f)
    vocab = set()
    for item in data:
        vocab.add(item['answer'])
        for d in item['distractors']:
            vocab.add(d)
    for i in vocab:
        of.write(i+'\n')
    of.close()
    print("load vocab done!")
    results = []
    for item in data:
        num = len(item['distractors'])
        item['neg_samples'] = random.sample(vocab,num)
        results.append(item)

    dump_json(results,"data/total_neg.json")
    print("output total_neg.json done!")

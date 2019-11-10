import pandas as pd
import numpy as np
import json
import csv
import re
from seq2seq.util.tokenizer import *


def process_lexnorm2015(sourcefile, targetfile):
    """
    lexnorm2015 the format of the data:
    {
        "tid": "",
        "index": "",
        "output": [],
        "input": []
    }
    :param sourcefile: source filename .json
    :param targetfile: The name of the file to be saved, the saved file format is csv
    :return:
    """
    with open(sourcefile) as f:
        source_file = json.load(f)
        print('the total of source data is ', len(source_file))
        with open(targetfile, 'a') as f:
            csv_write = csv.writer(f)
            for data in source_file:
                output = ' '.join(data['output'])
                input = ' '.join(data['input'])
                output = re.sub(RE_URL, "URL", output)
                input = re.sub(RE_URL, "URL", input)
                tid = data['tid']
                index = data['index']
                csv_write.writerow([input, output])

    print("Lexnorm2015 data, Done!!!")


def process_kaggle(sourcefile, targetfile):
    """
    the format of data is:
    "sentence_id","token_id","class","before","after"
    :param sourcefile: source filename .csv
    :param targetfile: The name of the file to be saved, the saved file format is csv , ['source', 'target']
    :return:
    """
    df = pd.read_csv(sourcefile, header=0, names=["sentence_id", "token_id", "class", "before", "after"])
    # join the word to a sentence
    sentences = df.groupby("sentence_id")
    print("How long are the sentences like?\n", sentences['sentence_id'].count().describe())
    with open(targetfile, 'a') as f:
        csv_write = csv.writer(f)
        for sent in sentences:
            before = ' '.join([str(s) for s in sent[1]["before"]])
            after = ' '.join([str(s) for s in sent[1]["after"]])
            csv_write.writerow([before, after])
    print("kaggle data done!!!")


if __name__ == '__main__':
    # sent = json.load(open("../../data/lexnorm2015/train_data.json"))
    # print(sent[0])
    # input = ' '.join(sent[0]['input'])
    # print(input)
    # process_lexnorm2015('../../data/lexnorm2015/train_data.json', "../../data/train_lexnrom2015.csv")
    process_kaggle("../../data/kaggle/en_train.csv", "../../data/train_kaggle.csv")



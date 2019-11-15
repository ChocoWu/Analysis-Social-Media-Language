import pandas as pd
import numpy as np
import json
import csv
import re
from seq2seq.util.tokenizer import *
from sklearn.model_selection import train_test_split


def replace_symbol(text):
    """
    replace HASHTAG MENTION URL PERCENTAGE EMAIL NUM DATE PHONE TIME MONEY
    :param text:
    :return:
    """
    result = re.sub(RE_HASHTAG, 'HASHTAG', text)
    result = re.sub(RE_MENTION, 'MENTION', result)
    result = re.sub(RE_URL, 'URL', result)
    result = re.sub(RE_PERCENTAGE, 'PERCENTAGE', result)
    result = re.sub(RE_EMAIL, 'EMAIL', result)
    result = re.sub(RE_NUM, 'NUM', result)
    result = re.sub(RE_DATE, 'DATE', result)
    result = re.sub(RE_PHONE, 'PHONE', result)
    result = re.sub(RE_TIME, 'TIME', result)
    result = re.sub(RE_MONEY, 'MONEY', result)
    result = re.sub(r'[\"\'\|\*\/]+', '', result)
    return result


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
                output = ' '.join(data['output']).strip()
                input = ' '.join(data['input']).strip()
                output = replace_symbol(output)
                input = replace_symbol(input)
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
            before = ' '.join([str(s) for s in sent[1]["before"]]).strip()
            after = ' '.join([str(s) for s in sent[1]["after"]]).strip()
            # remove the punctuation
            before = re.sub(RE_PUNCT, '', before)
            after = re.sub(RE_PUNCT, '', after)
            csv_write.writerow([before, after])
    print("kaggle data done!!!")


def remove_0_length(sourcefile, targetfile):
    """
    if the length of one sentence is zero, remove it
    :param sourcefile:
    :param targetfile:
    :return:
    """
    df = pd.read_csv(sourcefile, names=['src', 'tgt'])
    print(df['src'][0])
    print(df['tgt'][0])
    print('the length of the original data: ', len(df))
    data = df[[len(data) > 0 for data in df['src']]]
    print('the length of processed data: ', len(data))
    data.to_csv(targetfile, index=False, header=False)


def process_trac(sourcefile, targetfile):
    """
    process the data from TRAC, the format of the data is csv(id, context, label)
    :param sourcefile:
    :param targetfile:
    :return:
    """
    df = pd.read_csv(sourcefile, names=['id', 'context', 'label'])
    print("the total number of TRAC dataset is ", len(df))
    df['encode_context'] = df.apply(lambda s: replace_symbol(str(s.context)), axis=1)
    print(df['encode_context'][0])
    print(df['label'][0])
    df.to_csv(targetfile, columns=['encode_context', 'label'], index=False, header=False)


def merage_dataset(sf1, sf2, tf1):
    """

    :param sf1: source file 1 'src' 'tgt'
    :param sf2: source file 2 'context' 'label'
    :param tf1: target file 'src' 'tgt' 'context' 'label'
    :return:
    """
    df1 = pd.read_csv(sf1, names=['src', 'tgt'])
    df2 = pd.read_csv(sf2, names=['context', 'label'])
    length = len(df2)
    print('the length of meraged data is ', length)
    data = pd.concat([df1['src'][:length], df1['tgt'][:length], df2['context'], df2['label']], axis=1)
    data.to_csv(tf1, index=False, header=False)


if __name__ == '__main__':
    # sent = json.load(open("../../data/lexnorm2015/train_data.json"))
    # print(sent[0])
    # input = ' '.join(sent[0]['input'])
    # print(input)

    # df = pd.read_csv("../../data/kaggle/en_train.csv")
    # sentences = df.groupby("sentence_id")
    # print("How long are the sentences like?\n", sentences['sentence_id'].count().describe())

    # process_lexnorm2015('../../data/lexnorm2015/train_data.json', "../../data/train_lexnrom2015.csv")
    # process_kaggle("../../data/kaggle/en_train.csv", "../../data/train_kaggle.csv")

    # merage the  two source file
    # with open("../../data/train_lexnrom2015.csv", 'r') as f:
    #     for line in f:
    #         with open('../../data/train_kaggle.csv', 'a') as f_1:
    #             f_1.write(line)

    # split data into train data, dev data, test data
    # df = pd.read_csv('../../data/train_kaggle.csv')
    # train, test = train_test_split(df, shuffle=True, test_size=0.4)
    # dev, test = train_test_split(test, shuffle=True, test_size=0.5)
    # train.to_csv("../../data/train.csv", header=None, index=None)
    # test.to_csv('../../data/test.csv', header=None, index=None)
    # dev.to_csv("../../data/dev.csv", header=None, index=None)

    # remove sentence that length is zero
    # df = pd.DataFrame({"key": ['green', 'red', 'blue'],
    #                    "data1": ['a', 'b', 'c'], "sorce": [33, 61, 99]})
    # data = pd.concat([df, df], ignore_index=True)
    # print(data)
    # print([len(d) > 3 for d in data['key']])
    # print(data[[len(d) > 3 for d in data['key']]])
    # remove_0_length('../../data/train.csv', '../../data/train_1.csv')

    # process TRAC data
    # process_trac('../../data/agr_en_train.csv', '../../data/agr_en_train.csv')

    # merage data
    merage_dataset('../../data/train.csv', '../../data/agr_en_train.csv', '../../data/train_new.csv')
    merage_dataset('../../data/dev.csv', '../../data/agr_en_dev.csv', '../../data/dev_new.csv')
    merage_dataset('../../data/test.csv', '../../data/agr_en_fb_test_p.csv', '../../data/test_1.csv')
    merage_dataset('../../data/test.csv', '../../data/agr_en_tw_test_p.csv', '../../data/test_2.csv')



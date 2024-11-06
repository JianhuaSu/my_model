import json
import os
import shutil
import pandas as pd
import numpy as np
import argparse
import csv
import re

def cal_info(data_frame, info_name):
    
    print('****************************************')
    print('Data information of {} data'.format(info_name))

    labels = np.array(data_frame['label'].tolist())
    label_list = list(np.unique(labels))
    num_of_intents = len(label_list)

    print('{} Intent labels: {}'.format(num_of_intents, label_list))

    label_dict = {}
    for label in label_list:
        num = len(labels[labels == label])
        label_dict[label] = num

    for k in sorted(label_dict, key=label_dict.__getitem__, reverse=True):
        print('label:{}, num:{}'.format(k, label_dict[k]))

    max_length = 0
    sum_length = 0
    texts = np.array(data_frame['text'].tolist())
    cnt = 0
    cnt_ones =  0
    lengths = []
    for text in texts:
        sent_length = len(text.split(' '))
        if sent_length == 1:
            cnt_ones += 1
        sum_length += sent_length
        max_length = max(max_length, sent_length)
        lengths.append(sent_length)

    final_length = int(np.mean(lengths) + 3 * np.std(lengths))

    print('The number of ones is {}'.format(cnt_ones))
    avg_length = sum_length / len(texts)
    print('max_length: %d, avg_length: %.3f, final_length: %.3f' % (max_length, avg_length, final_length))

def raw_data_process(args):

    with open(args.raw_text_path, 'r') as f:

        reader = csv.reader(f, delimiter='\t')
        all_elems = []
        
        for i, line in enumerate(reader):
            if i == 0:
                columns = line
                continue
            print(line)
            
            sent = re.sub('(\s*-\s)|(\"+)|(\[\w+\])|(<i>)|(</i>)', ' ', line[3]).strip()
            if sent[-1] == ',':
                sent = sent[:-1]
            all_elems.append([line[0], line[1], line[2], sent.lower(), line[4]])

    all_data_frame = pd.DataFrame(all_elems, columns=columns)
    all_data_frame.to_csv(args.plain_text_path, sep = '\t', index = False)

def check(args):
    
    text_data_path = os.path.join(args.data_path, 'all.tsv')
    all_data_frame = pd.read_csv(text_data_path, sep = '\t')
    
    cal_info(all_data_frame, 'all')
    
    train_data_path = os.path.join(args.data_path, 'train.tsv')
    train_data_frame = pd.read_csv(train_data_path, sep = '\t')

    cal_info(train_data_frame, 'train')

    dev_data_path = os.path.join(args.data_path, 'dev.tsv')
    dev_data_frame = pd.read_csv(dev_data_path, sep = '\t')

    cal_info(dev_data_frame, 'dev')

    test_data_path = os.path.join(args.data_path, 'test.tsv')
    test_data_frame = pd.read_csv(test_data_path, sep = '\t')

    cal_info(test_data_frame, 'test')    
    
def train_test_splits(args):

    text_data_path = os.path.join(args.data_path, 'all.tsv')
    all_data_frame = pd.read_csv(text_data_path, sep = '\t')

    # from sklearn.utils import shuffle
    # all_data_frame = shuffle(all_data_frame)
    # all_data_frame.to_csv(os.path.join(args.data_path, 'all.tsv'), sep = '\t', index = False)

    from sklearn.model_selection import train_test_split
    seed = 0

    # train: dev: test = 5:1:2
    # train: dev: test = 7:1:2

    train_dev_other, test = train_test_split(all_data_frame, test_size = 450 / 1602, stratify = all_data_frame.label, shuffle = True, random_state = seed)
    print('11111111', len(train_dev_other))
    other, train_dev = train_test_split(train_dev_other, test_size = 150 / 1151, stratify = train_dev_other.label, shuffle = True, random_state = seed)

    train, dev = train_test_split(train_dev, test_size = 1 / 3, stratify = train_dev.label, shuffle = True, random_state = seed)

    train.to_csv(os.path.join(args.data_path, 'train.tsv'), sep = '\t', index = False)
    dev.to_csv(os.path.join(args.data_path, 'dev.tsv'), sep = '\t', index = False)
    test.to_csv(os.path.join(args.data_path, 'test.tsv'), sep = '\t', index = False)

    cal_info(train, 'train')
    cal_info(dev, 'dev')
    cal_info(test, 'test')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--raw_text_path', type=str, default='/home/sharing/Datasets/L-M-8081/L-MIntRec/all.tsv', help="The directory of raw text data.")
    # parser.add_argument('--plain_text_path', type=str, default='/home/sharing/Datasets/L-M-8081/L-MIntRec/all.tsv', help="The directory of raw text data.")

    parser.add_argument('--data_path', type=str, default='/home/sharing/disk1/zhanghanlei/Datasets/public/TED-OOD', help="The directory of the text data.")
    args = parser.parse_args()

    # raw_data_process(args)
    train_test_splits(args)
    # check(args)



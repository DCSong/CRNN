# -*- coding: UTF-8 -*- 
import os

char2idx_path = './Dataset/labels/Char2Idx.txt'

char2idx = {}
with open(char2idx_path, 'r', encoding='utf-8-sig') as fp:
    text = fp.read()
    char2idx = eval(text)

result = ''
with open('./data_generator/data_set/labels.txt', 'r', encoding='utf-8') as fa, open('./data_generator/data_set/labels_trans.txt', 'w', encoding='utf-8') as fp:
    lines = fa.readlines()
    idxLabels = []
    for line in lines:
        line = line.replace('\n', '')
        name, labels = line.split(' ')
        labels = list(labels)
        idx = []
        for label in labels:
            idx.append(str(char2idx[label]))
        idxLabel = name + ' ' + ' '.join(idx) + '\n'
        idxLabels.append(idxLabel)
    fp.writelines(idxLabels)
    

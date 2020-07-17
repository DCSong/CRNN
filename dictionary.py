# -*- coding: UTF-8 -*- 
import os

char_dict = {}
with open(file='./Dataset/labels/char_std_6129.txt', mode='r', encoding='utf-8-sig') as fp:
    lines = fp.readlines()
    num = len(lines)
    for idx, line in zip(range(num), lines):
        line = line.replace('\n', '')
        char_dict.update({line: idx})

inverse_char_dict = dict([val, key] for key, val in char_dict.items())


char2idx_path = './Dataset/labels/Char2Idx.txt'
idx2char_path = './Dataset/labels/Idx2Char.txt'

with open(file=char2idx_path, mode='w', encoding='utf-8-sig') as fp:
    fp.write(str(char_dict))

with open(file=idx2char_path, mode='w', encoding='utf-8-sig') as fp:
    fp.write(str(inverse_char_dict))


# -*- coding: UTF-8 -*- 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from torchvision import transforms

import cv2
import os
import time
import numpy as np

import main
from main import ctc_greedy_decoder
import model_v2

import Config
width = Config.width
height = Config.height
channels = Config.channels
num_classes = Config.num_classes
ctc_blank = Config.ctc_blank

IMG_ROOT = Config.TEST_IMG_ROOT      # 测试图片存放文件夹

# 判断是使用GPU还是CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取字典文件 用来翻译结果
fp = open('./Dataset/labels/Idx2Char.txt', 'r', encoding='utf-8-sig')
dictionary = fp.read()
fp.close()

char_dict = eval(dictionary)


# 根据字典将序号转换为文字
def Idx2Word(labels, dict=char_dict):
    texts = []
    for label in labels[0]:
        texts.append(dict[label])
    return texts


# 读取图像，解决imread不能读取中文路径的问题
def cv_imread(file_path):
    # cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flags=cv2.IMREAD_GRAYSCALE)
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flags=cv2.IMREAD_COLOR)
    return cv_img

# 对结果进行解码/翻译
def decode(outputs):
    output_argmax = outputs.permute(1, 0, 2).argmax(dim=-1)
    output_argmax = output_argmax.cpu().numpy()
    # assert output_argmax.shape = (batch_size, int(sequence_length[0]))
    output_labels = output_argmax.tolist()
    output = ''
    for idx in output_labels[0]:
        output += str(char_dict[idx])
    return output


def infer(image, model):
    model.eval()
    image = image.unsqueeze(0)
    image = image.to(device)
    output = model(image)
    output_log_softmax = F.log_softmax(output, dim=-1)
    # print('predict result:{}\n'.format(decode(output_log_softmax)))
    pred_labels = ctc_greedy_decoder(output_log_softmax)
    pred_texts = ''.join(Idx2Word(pred_labels))
    print('predict result: {}\n'.format(pred_texts))


if __name__ == '__main__':
    transform = transforms.ToTensor()

    # 模型定义
    model = model_v2.DenseNet_BLSTM_CTC_MODEL()
    model = model.to(device)
    print(model)

    if os.path.exists('checkpoint_v2.pth.tar'):
        checkpoint = torch.load('checkpoint_v2.pth.tar', map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print('model has restored')

    # 依次推断文件夹中每张图片
    files = sorted(os.listdir(IMG_ROOT))
    for file in files:
        image_path = os.path.join(IMG_ROOT, file)
        image = cv_imread(image_path)

        # if (image.shape[1] / image.shape[0] < width / height):
        #     padding = int(width / height * image.shape[0]) - image.shape[1]
        #     image = cv2.copyMakeBorder(image, 0, 0, 0, padding, borderType=cv2.BORDER_CONSTANT, value=255)

        # 改变图片大小以适应网络结构要求
        scale = float(height / image.shape[0])
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = transform(image)

        started = time.time()
        print("=============================================")
        print("ocr image is %s\n" % image_path)

        infer(image, model)    # 测试
        finished = time.time()
        print('elapsed time: {0}\n'.format(finished - started))
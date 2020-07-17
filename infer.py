# -*- coding: UTF-8 -*- 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from torchvision import transforms
import cv2            # 图形处理库
import os             # 文件读取库
import time
import numpy as np    # 向量处理

import main
from main import ctc_greedy_decoder
import model
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


# 测试函数
def infer(image, model):
    model.eval()
    image = image.unsqueeze(0)
    image = image.to(device)
    # 将图片数据输入模型
    output = model(image)
    output_log_softmax = F.log_softmax(output, dim=-1)
    # print('predict result:{}\n'.format(decode(output_log_softmax)))
    # 对结果进行解码
    pred_labels = ctc_greedy_decoder(output_log_softmax)
    pred_texts = ''.join(Idx2Word(pred_labels))
    print('predict result: {}\n'.format(pred_texts))


if __name__ == '__main__':
    transform = transforms.ToTensor()

    # 模型定义
    model = model.CRNN(num_classes)
    model = model.to(device)
    print(model)

    # 读取参数
    if os.path.exists('checkpoint.pth.tar'):
        checkpoint = torch.load('checkpoint.pth.tar', map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print('model has restored')

    # 依次推断文件夹中每张图片
    files = sorted(os.listdir(IMG_ROOT))
    # for循环依次读取文件夹中的每张图片
    for file in files:
        image_path = os.path.join(IMG_ROOT, file)
        # 将图片读取进来
        image = cv_imread(image_path)
        
        # 改变图片大小以适应网络结构要求
        scale = float(height / image.shape[0])
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        # 将彩色图片转化为灰度图像
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = transform(image)

        started = time.time()
        print("=============================================")
        print("ocr image is %s\n" % image_path)

        infer(image, model)    # 测试
        finished = time.time()
        print('elapsed time: {0}\n'.format(finished - started))
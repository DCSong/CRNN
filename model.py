# -*- coding: UTF-8 -*- 
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

# from efficientnet_pytorch import EfficientNet

import Config
width = Config.width
height = Config.height
channels = Config.channels
num_classes = Config.num_classes
ctc_blank = Config.ctc_blank


# 卷积层部分定义
class VGG_16(torch.nn.Module):
    def __init__(self):
        super(VGG_16, self).__init__()
        self.convolution1 = torch.nn.Conv2d(1, 64, 3, padding=1)
        self.pooling1 = torch.nn.MaxPool2d(2, stride=2)
        self.convolution2 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.pooling2 = torch.nn.MaxPool2d(2, stride=2)
        self.convolution3 = torch.nn.Conv2d(128, 256, 3, padding=1)
        self.convolution4 = torch.nn.Conv2d(256, 256, 3, padding=1)
        self.pooling3 = torch.nn.MaxPool2d((2, 2), stride=(2, 1), padding=(0, 1))
        self.convolution5 = torch.nn.Conv2d(256, 512, 3, padding=1)
        self.BatchNorm1 = torch.nn.BatchNorm2d(512)
        self.convolution6 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.BatchNorm2 = torch.nn.BatchNorm2d(512)
        self.pooling4 = torch.nn.MaxPool2d((2, 2), stride=(2, 2))
        self.convolution7 = torch.nn.Conv2d(512, 512, 2)

    def forward(self, input):
        x = F.relu(self.convolution1(input), inplace=True)
        x = self.pooling1(x)
        x = F.relu(self.convolution2(x), inplace=True)
        x = self.pooling2(x)
        x = F.relu(self.convolution3(x), inplace=True)
        x = F.relu(self.convolution4(x), inplace=True)
        x = self.pooling3(x)
        x = self.convolution5(x)
        x = F.relu(self.BatchNorm1(x), inplace=True)
        x = self.convolution6(x)
        x = F.relu(self.BatchNorm2(x), inplace=True)
        x = self.pooling4(x)
        x = F.relu(self.convolution7(x), inplace=True)
        return x    # batch_size * 512 * 1 * 16


# 循环层定义
class BLSTM(torch.nn.Module):
    def __init__(self, nIn, nHidden, classes):
        super(BLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=nIn,
                           hidden_size=nHidden,
                           num_layers=1,
                           bidirectional=True)
        
        self.embedding = nn.Linear(nHidden * 2, classes)  


    def forward(self, input):
        recurrent, _ = self.rnn(input)
        time_steps, batch_size, channel = recurrent.size()
        recurrent = recurrent.view(time_steps * batch_size, channel)
        output = self.embedding(recurrent)
        output = output.view(time_steps, batch_size, -1)
        return output     # time_steps:17 * batch_size * num_classes:6129


# CRNN模型
class CRNN(torch.nn.Module):
    def __init__(self, class_num=num_classes, hidden_unit=256):
        super(CRNN, self).__init__()
        self.CNN = torch.nn.Sequential()
        self.CNN.add_module('VGG_16', VGG_16())
        self.RNN = torch.nn.Sequential()
        self.RNN.add_module('BLSTM1', BLSTM(512, hidden_unit, hidden_unit))
        self.RNN.add_module('BLSTM2', BLSTM(hidden_unit, hidden_unit, class_num))

    def forward(self, input):
        x = self.CNN(input)
        b, c, h, w = x.size()
        assert h == 1   # "the height of conv must be 1"
        x = x.squeeze(2)  # remove h dimension, b * 512 * width
        x = x.permute(2, 0, 1)  # [w, b, c] = [time_steps, batch_size, input_size]
        output = self.RNN(x)
        return output



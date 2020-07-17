# -*- coding: UTF-8 -*- 
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

from torchvision.models.densenet import _DenseBlock
from torchvision.models.densenet import _Transition

import Config
width = Config.width
height = Config.height
channels = Config.channels
num_classes = Config.num_classes
ctc_blank = Config.ctc_blank


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
        return output     # time_steps:21 * batch_size * num_classes:5990


# 卷积层部分定义
class DenseNet(nn.Module):
    __constants__ = ['features']

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0.1, memory_efficient=False):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(channels, num_init_features, kernel_size=7, stride=(2, 2),
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=0)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                if i % 2 == 0:
                    trans = Transition22(num_input_features=num_features,
                                        num_output_features=num_features // 2)
                elif i % 2 == 1:
                    trans = Transition21(num_input_features=num_features,
                                        num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # TimeDistributed
        # self.timedistributed = TimeDistributed(nn.Linear(num_features, num_classes, bias=True), batch_first=False)


    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        return out


class Transition22(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(Transition22, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class Transition21(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(Transition21, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=(2, 1)))


# CRNN模型
class DenseNet_BLSTM_CTC_MODEL(torch.nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0.1, num_classes=num_classes, memory_efficient=False):
        super( DenseNet_BLSTM_CTC_MODEL, self).__init__()
        self.densetnet = nn.Sequential(
            DenseNet(growth_rate, block_config, num_init_features, bn_size, 
                     drop_rate, memory_efficient))
        self.RNN = torch.nn.Sequential()
        self.RNN.add_module('BLSTM1', BLSTM(1024, 256, 256))
        self.RNN.add_module('BLSTM2', BLSTM(256, 256, num_classes))

    def forward(self, input):
        output = self.densetnet(input)
        batch_size, channels, height, width = output.size()
        assert height == 1
        output = output.squeeze(2)
        output = output.permute(2, 0, 1)
        output = self.RNN(output)     # 34 * batch * num_classes
        return output

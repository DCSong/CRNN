# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import models
from torchvision import transforms
# from torchvision.transforms import Compose, ToTensor, Resize

import cv2
import os
from tqdm import tqdm
import random
import numpy as np
from collections import OrderedDict

import model_v2
import Config
width = Config.width
height = Config.height
image_shape = Config.image_shape
channels = Config.channels
num_classes = Config.num_classes
ctc_blank = Config.ctc_blank

batch_size = Config.batch_size
epochs = Config.epochs

# 判断使用CPU还是GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.ToTensor()


# 文件读取类
class ImageDataSet(Dataset):
    def __init__(self, info_file, transform=None):
        self.transform = transform
        self.img_names = []
        self.img_labels = []
        self.labels_length = []

        with open(info_file, 'r', encoding='utf-8') as fp:
            lines = fp.readlines()
            for line in lines:
                img_name = line.split(' ')[0]
                img_label = line.replace('\n', '').split(' ')[1: ]
                img_label = map(int, img_label)
                img_label = list(img_label)
                self.img_names.append(img_name)
                self.img_labels.append(img_label)
                self.labels_length.append(len(img_label))


    def __getitem__(self, index):
        path = './Dataset/images/images/'
        image_path = path + self.img_names[index]
        image_label = self.img_labels[index]
        label_length = self.labels_length[index]

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img, image_label, label_length

    def __len__(self):
        return len(self.img_names)


# 重构的文件结构组织函数
def collate_fn(data_list):
    img_data, img_label, label_length = zip(*data_list)
    length = len(label_length)
    img_label = list(img_label)
    img_data = list(img_data)
    image_label = []
    image_width = []
    for i in range(length):
        image_label.extend(img_label[i])
        # img_data[i] = cv2.copyMakeBorder(img_data[i], 0, 0, 0, width - 280, borderType=cv2.BORDER_CONSTANT, value=255)
        # img_data[i] = cv2.resize(img_data[i], (width, height), interpolation=cv2.INTER_CUBIC)
        img_data[i] = transform(img_data[i])
        # image_width.append(img_data[i].shape[1])
    # max_width = max(image_width)
    # for i in range(length):
    #     if label_length[i] != 10:
    #         img_data[i] = cv2.copyMakeBorder(img_data[i], 0, 0, 0, max_width - img_data[i].shape[1], borderType=cv2.BORDER_CONSTANT, value=255)
    #     img_data[i] = transform(img_data[i])
    # image_data = torch.DoubleTensor(img_data)
    image_data = torch.stack(img_data, dim=0)
    # image_label = torch.IntTensor(image_label)
    # label_length = torch.IntTensor(label_length)
    image_label = torch.LongTensor(image_label)
    label_length = torch.LongTensor(label_length)
    return [image_data, image_label, label_length]


# 解码真实标签
def decode_trueLabel(labels, label_lengths):
    labels = labels.cpu().numpy()
    true_labels = []
    sum = 0
    for index in range(label_lengths.shape[0]):
        # target_label[index][0 : (target_lengths[index])] = target[sum : (sum + target_lengths[index])]
        true_labels.append(labels[sum : (sum + label_lengths[index])].tolist())
        sum += label_lengths[index]
        # if target_lengths[index] < cols:
        # pad = [3755] * (cols - target_lengths[index])
        # target_label[index][target_lengths[index] :] = 3755
    return true_labels


# 贪心策略解码函数，解码预测标签
def ctc_greedy_decoder(outputs, ctc_blank=ctc_blank):
    output_argmax = outputs.permute(1, 0, 2).argmax(dim=-1)
    output_argmax = output_argmax.cpu().numpy()
    # assert output_argmax.shape = (batch_size, int(sequence_length[0]))
    output_labels = output_argmax.tolist()
    pred_labels = []

    # 删除ctc_blank
    for label in output_labels:
        pred_label= []
        preNum = label[0]
        for curNum in label[1: ]:
            if preNum == ctc_blank:
                pass
            elif curNum == preNum:
                pass
            else:
                pred_label.append(preNum)
            preNum = curNum
        if preNum != ctc_blank:
            pred_label.append(preNum)
        pred_labels.append(pred_label)

    return pred_labels


# 计算最小编辑距离
def minStringDistance(true_label, pred_label):
    n1 = len(true_label)
    n2 = len(pred_label)
    # dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]
    dp = np.zeros((n1 + 1, n2 + 1), dtype=np.int32)
    # 第一行
    for j in range(1, n2 + 1):
        dp[0][j] = dp[0][j - 1] + 1
    # 第一列
    for i in range(1, n1 + 1):
        dp[i][0] = dp[i - 1][0] + 1

    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            if true_label[i - 1] == pred_label[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1] ) + 1
    return dp[-1][-1]


# 准确率计算函数
def calculate_accuracy(true_labels, pred_labels):
    '''
    CER = (Sub + Ins + Del) / (Ins + Del + N)
    CER = (Sub + Ins + Del) / N = MSD / N
    SER
    '''
    assert len(true_labels) == len(pred_labels)
    sentences_count = len(true_labels)

    error_sentences = 0
    total_CER = 0
    for true, pred in zip(true_labels, pred_labels):
        MSD = minStringDistance(true, pred)
        if (MSD != 0):
            error_sentences += 1

        total_CER += MSD / len(true)

    avg_CER = total_CER / sentences_count
    avg_SER = error_sentences / sentences_count
    return avg_CER, avg_SER


# 训练函数
def train(model, optimizer, epoch, dataloader):
    model.train()
    # loss_mean = 0
    # acc_mean = 0
    with tqdm(dataloader) as pbar:
        loss_sum = 0
        cer_sum = 0
        ser_sum = 0
        # acc_sum = 0
        for batch_index, (img_data, target, target_lengths) in enumerate(pbar):
            # 读取一个batch的数据
            img_data, target, target_lengths = img_data.to(device), target.to(device), target_lengths.to(device)
            # input_lengths = torch.IntTensor([25] * int(target_lengths.size(0))).cuda()

            optimizer.zero_grad()
            output = model(img_data)  # 调用模型得到结果

            output_log_softmax = F.log_softmax(output, dim=-1)

            input_length = int(output_log_softmax.shape[0])
            # input_lengths = torch.LongTensor([input_length] * int(target_lengths.size(0))).cuda()
            input_lengths = torch.IntTensor([input_length] * int(target_lengths.size(0))).to(device)

            # 损失值计算
            loss = ctc_loss(output_log_softmax, target, input_lengths, target_lengths)

            loss.backward()
            optimizer.step()

            batch_loss = loss.item()

            # 标签解码
            true_labels = decode_trueLabel(target, target_lengths)
            pred_labels = ctc_greedy_decoder(output_log_softmax)
            # pred_labels = ctc_beam_search_decoder(output_log_softmax)

            # 准确率计算
            cer, ser = calculate_accuracy(true_labels, pred_labels)

            # 损失值、SER、CER累加和平均
            loss_sum += batch_loss
            cer_sum += cer
            ser_sum += ser
            epoch_loss = loss_sum / (batch_index + 1)
            cer_mean = cer_sum / (batch_index + 1)
            ser_mean = ser_sum / (batch_index + 1)

            pbar.set_description(f'Train Epoch: {epoch}  Batch_Loss: {batch_loss:.4f}  Epoch_Loss: {epoch_loss:.4f}  CER: {cer:.4f}  SER: {ser:.4f}')


# 验证函数
def valid(model, epoch, dataloader):
    model.eval()
    with tqdm(dataloader) as pbar, torch.no_grad():
        loss_sum = 0
        cer_sum = 0
        ser_sum = 0
        # acc_sum = 0
        for batch_index, (img_data, target, target_lengths) in enumerate(pbar):
            # img_data, target, target_lengths = img_data.cuda(), target.cuda(), target_lengths.cuda()
            img_data, target, target_lengths = img_data.to(device), target.to(device), target_lengths.to(device)
            # input_lengths = torch.IntTensor([25] * int(target_lengths.size(0))).cuda()

            output = model(img_data)
            output_log_softmax = F.log_softmax(output, dim=-1)

            input_length = int(output_log_softmax.shape[0])
            # input_lengths = torch.IntTensor([input_length] * int(target_lengths.size(0))).cuda()
            input_lengths = torch.IntTensor([input_length] * int(target_lengths.size(0))).to(device)
            # loss = F.ctc_loss(output_log_softmax, target, input_lengths, target_lengths)
            loss = ctc_loss(output_log_softmax, target, input_lengths, target_lengths)

            batch_loss = loss.item()

            true_labels = decode_trueLabel(target, target_lengths)
            pred_labels = ctc_greedy_decoder(output_log_softmax)
            # pred_labels = ctc_beam_search_decoder(output_log_softmax)

            cer, ser = calculate_accuracy(true_labels, pred_labels)

            loss_sum += batch_loss
            cer_sum += cer
            ser_sum += ser
            epoch_loss = loss_sum / (batch_index + 1)
            cer_mean = cer_sum / (batch_index + 1)
            ser_mean = ser_sum / (batch_index + 1)


            pbar.set_description(f'Valid Epoch: {epoch}  Batch_Loss: {batch_loss:.4f} Epoch_Loss: {epoch_loss:.4f}  CER: {cer:.4f} SER: {ser:.4f} ')


if __name__ == '__main__':
    # 训练集和验证集
    train_set = ImageDataSet(info_file='./Dataset/labels/train.txt', transform=transform)
    valid_set = ImageDataSet(info_file='./Dataset/labels/valid.txt', transform=transform)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, collate_fn=collate_fn)
    valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, collate_fn=collate_fn)

    # 模型
    model = model_v2.DenseNet_BLSTM_CTC_MODEL(growth_rate=32,
                                              block_config=(6, 12, 24, 16),
                                              num_init_features=64,
                                              bn_size=4,
                                              drop_rate=0.1,
                                              num_classes=num_classes,
                                              memory_efficient=True)

    # model = model.cuda()
    model = model.to(device)
    print(model)

    if os.path.exists('checkpoint_v2.pth.tar'):
        checkpoint = torch.load('checkpoint_v2.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
        print('model has restored')

    # 优化器定义
    optimizer = torch.optim.Adam(model.parameters(), 1e-3, amsgrad=True, weight_decay=1e-5)
    # optimizer = torch.optim.Adam(model.parameters(), 1e-3, amsgrad=True)
    
    # 损失函数定义
    ctc_loss = torch.nn.CTCLoss(blank=ctc_blank, reduction='mean')
    # ctc_loss = ctc_loss.cuda()
    ctc_loss = ctc_loss.to(device)

    # 训练过程
    for epoch in range(1, epochs + 1):
        train(model, optimizer, epoch, train_loader)
        # torch.save(model, 'ctc.pth')
        torch.save({'state_dict': model.state_dict()}, 'checkpoint_v2.pth.tar')
        print('model has saved')
        valid(model, epoch, valid_loader)

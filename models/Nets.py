#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        #self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.convbn1 = nn.BatchNorm2d(10, momentum=0.9)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.convbn2 = nn.BatchNorm2d(20)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 84)
        self.fc2 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.convbn1(x)
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.convbn2(x)
        x = self.pool(x)
        # print(x.shape)
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.softmax(self.fc2(x), dim=1)
        return x


class CNNFEMnist(nn.Module):
    def __init__(self, args):
        super(CNNFEMnist, self).__init__()
        # self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding="same")
        self.convbn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding="same")
        self.convbn2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding="same")
        self.convbn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding="same")
        self.convbn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, padding="same")
        self.convbn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding="same")
        self.convbn6 = nn.BatchNorm2d(64)
        # self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(64 * 3 * 3, 84)
        self.fc2 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.conv1(x))
        x = self.convbn1(x)
        x = F.relu(self.conv2(x))
        x = self.convbn2(x)
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.convbn3(x)
        x = F.relu(self.conv4(x))
        x = self.convbn4(x)
        x = self.pool(x)

        x = F.relu(self.conv5(x))
        x = self.convbn5(x)
        x = F.relu(self.conv6(x))
        x = self.convbn6(x)
        x = self.pool(x)
        # print(x.shape)
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 32 * 3 * 3)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=0.4)
        x = F.softmax(self.fc2(x), dim=1)
        return x


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding='same')
        self.convbn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding='same')
        self.convbn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding='same')
        self.convbn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding='same')
        self.convbn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, 3, padding='same')
        self.convbn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding='same')
        self.convbn6 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, args.num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.convbn1(x)
        x = F.relu(self.conv2(x))
        x = self.convbn2(x)
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.convbn3(x)
        x = F.relu(self.conv4(x))
        x = self.convbn4(x)
        x = self.pool(x)

        x = F.relu(self.conv5(x))
        x = self.convbn5(x)
        x = F.relu(self.conv6(x))
        x = self.convbn6(x)
        x = self.pool(x)

        x = x.view(-1, 128 * 3 * 3)  #Flatten the output
        x = F.relu(self.bn1(self.fc1(x)))
        F.dropout(x, training=self.training, p=0.2)
        x = F.softmax(self.fc2(x), dim=1)
        return x


class CNN_HAR(nn.Module):
    def __init__(self, args):
        super(CNN_HAR, self).__init__()
        self.conv1 = nn.Conv1d(args.n_channels, 32, 3, padding='same')
        self.convbn1 = nn.BatchNorm1d(32)
        # self.conv2 = nn.Conv1d(32, 32, 3, padding='same')
        self.convbn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, 3, padding='same')
        self.convbn3 = nn.BatchNorm1d(64)
        # self.conv4 = nn.Conv1d(64, 64, 3, padding='same')
        self.convbn4 = nn.BatchNorm1d(64)
        # self.conv5 = nn.Conv1d(64, 128, 3, padding='same')
        self.convbn5 = nn.BatchNorm1d(128)
        # self.conv6 = nn.Conv1d(128, 128, 3, padding='same')
        self.convbn6 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2, 2)

        self.fc1 = nn.Linear(64 * 32 * 1, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, args.n_classes)

    def forward(self, x):
                                        #[1, 9, 128]
        # print(x.shape)
        x = F.relu(self.conv1(x))
        # print(x.shape)                  #[32, 9, 128]
        x = self.convbn1(x)

        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.convbn3(x)
        x = self.pool(x)

        # print(x.shape)          #[1, 128, 16]
        x = x.view(-1, 64 * 32 * 1)  #Flatten the output
        x = F.relu(self.bn1(self.fc1(x)))
        F.dropout(x, training=self.training, p=0.2)
        x = F.softmax(self.fc2(x), dim=1)
        return x

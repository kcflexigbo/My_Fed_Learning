import sys

import torch
from torch import nn, autograd
import numpy as np
import random
from sklearn import metrics
import copy
from time import time
import os


class Clients(object):
    def __init__(self, args=None, tdata=None, lmodel=None, title=None, logPath=None):
        self.train_data = tdata
        self.args = args
        self.local_model = lmodel
        self.loss_func = nn.CrossEntropyLoss()
        if self.args is not None:
            self.batch_size = self.args.local_bs
        self.loss = 0
        self.accuracy = 0
        self.title = title
        self.model_path = os.path.join(logPath, f"client_model_{self.title}.pt")

    def train(self, net, client_queue=None, timer=None, use_multiprocessing=True, ):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_data):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.train_data.dataset),
                              100. * batch_idx / len(self.train_data), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        self.loss = sum(epoch_loss) / len(epoch_loss)
        # self.local_model = net
        print(self.loss)
        if use_multiprocessing:
            # torch.save(net, self.model_path)
            torch.save(net.state_dict(), self.model_path)
            client_queue.put(self.loss)
        else:
            return net.state_dict(), self.loss

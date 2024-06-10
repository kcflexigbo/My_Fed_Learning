#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=1, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.05, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=10, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar', help="name of dataset")
    parser.add_argument('--iid', action='store_false', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of images")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')

    parser.add_argument('--csv_path', type=str, default='data/HAR/UCI_Smartphone_Raw.csv', help='CSV data path')
    # parser.add_argument('--round', type=int, default=100, help='Round for federated learning')
    # parser.add_argument('--internal_epoch', type=int, default=10, help='Internal epoch of each client')
    parser.add_argument('--global_model_path', type=str, default='global_model/Global_CNN.pt',
                        help='Trained global model path')
    # parser.add_argument('--batch_size', type=int, default=64, help='Batch size used')
    # parser.add_argument('--lr', type=float, default=0.01, help='Learning rate used')
    # parser.add_argument('--C', type=float, default=1.0, help='Fraction of client for each round averaging')
    parser.add_argument('--val_split', type=float, default=0.3, help='Validation split for test')
    parser.add_argument('--lambda_coral', type=float, default=0.01, help='trade off parameter in CORAL loss')
    # parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    # parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay for SGD')
    # parser.add_argument('--lr_patience', type=int, default=99999,
    #                     help='learning patience before reduced when loss does not go down in each client')
    parser.add_argument('--n_channels', type=int, default=9, help="Number of channels")
    parser.add_argument('--n_classes', type=int, default=6, help="Number of classes")
    args = parser.parse_args()
    return args

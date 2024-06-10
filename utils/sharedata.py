from utils.Clients import Clients
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

    # def __getitem__(self, idx):
    #     data = torch.from_numpy(self.dataset[idx])
    #     label = self.idxs[idx]
    #
    #     return data, label


def create_clients(args, dataset_to_split, dict_users, client_path):
    data_split = []
    for i in range(args.num_users):
        data_split.append(DataLoader(DatasetSplit(dataset_to_split, dict_users[i]),
                                     batch_size=args.local_bs, shuffle=True))
    clients_list = []
    for i in range(args.num_users):
        client = Clients(title=i, tdata=data_split[i], args=args, logPath=client_path)
        clients_list.append(client)
    return clients_list

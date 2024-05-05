import numpy as np
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from Clients import Clients
from tkinter import filedialog


# num_users=100
# num_items=int(len(data)/num_users)
# dict_users,all_idxs= {}, [i for i in range(len(data))]
# for i in range(num_users):
#     dict_users[i]= set(np.random.choice(all_idxs,num_items,replace=False))
#     all_idxs= list(set(all_idxs) - dict_users[i])
#     print(len(all_idxs),",",dict_users[i])


#
# for i in clients_list:
#     print(i.title)

# logpath= filedialog.askdirectory()
# print(logpath)

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def create_clients(num_users, data_split):
    num_clients = 20
    clients_list = []
    for i in range(num_clients):
        client = Clients(title=i, tdata=data_split[i])
        clients_list.append(client)
    return clients_list


if __name__ == '__main__':
    data = np.random.randint(0, 100, 200)
    label = np.random.randint(0, 10, 200)
    newdata = list(zip(data.tolist(), label.tolist()))
    num_users = 20
    dict_users = cifar_iid(newdata, num_users)
    data_split = []
    for i in range(num_users):
        data_split.append(DataLoader(DatasetSplit(newdata, dict_users[i]), batch_size=3, shuffle=True))
    count = 0
    j = 0
    # for i in data_split:
    #     print(f'\nuser {j}:')
    #     j += 1
    #     for (images, labels) in i:
    #         print(f'images: {images} labels: {labels}')

    clients_list = create_clients(num_users, data_split)
    for client in clients_list:
        print(f'Client Title: {client.title}, length: {len(client.train_data)}')
        for i in client.train_data:
            print(i)
        print("\n")

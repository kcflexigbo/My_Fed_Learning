from Clients import Clients
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def create_clients(args, dataset_to_split, dict_users):
    data_split = []
    for i in range(args.num_users):
        data_split.append(DataLoader(DatasetSplit(dataset_to_split, dict_users[i]), batch_size=args.local_bs, shuffle=True))
    clients_list = []
    for i in range(args.num_users):
        client = Clients(title=i, tdata=data_split[i])
        clients_list.append(client)
    return clients_list
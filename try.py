import numpy as np

import pandas as pd

from Clients import Clients
from tkinter import filedialog
# data=np.random.randint(0,100,1000)
# num_users=100
# num_items=int(len(data)/num_users)
# dict_users,all_idxs= {}, [i for i in range(len(data))]
# for i in range(num_users):
#     dict_users[i]= set(np.random.choice(all_idxs,num_items,replace=False))
#     all_idxs= list(set(all_idxs) - dict_users[i])
#     print(len(all_idxs),",",dict_users[i])

# num_clients= 10
# clients_list=[]
# for i in range(num_clients):
#     client=Clients(title=i)
#     clients_list.append(client)
#
# for i in clients_list:
#     print(i.title)

logpath= filedialog.askdirectory()
print(logpath)
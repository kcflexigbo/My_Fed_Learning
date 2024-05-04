import matplotlib

#matplotlib.use('Agg')
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import *
from PIL import Image, ImageTk
from utils.options import args_parser
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
import warnings
from sharedata import create_clients
from utils.filesbrowser import createPath

global num
num = 100


def initGUI(root, args):
    root.geometry('1080x500')
    root.title('Federated Learning Demo')
    root.resizable(False, False)
    icon = Image.open("icon-32.png")
    icon = ImageTk.PhotoImage(icon)
    root.iconphoto(True, icon)
    # p1 = tk.PhotoImage(file='icon-32.png')
    # root.iconphoto(False, p1)

    lb1 = tk.Label(root, text='客户的数量:', font=('Arial', 11))
    lb1.place(relx=0.002, rely=0.002, relheight=0.04, relwidth=0.075)
    global no_clients_text
    no_clients_text = tk.StringVar()
    no_clients_text.set(str(args.num_users))
    no_clients = tk.Entry(root, textvariable=no_clients_text)
    no_clients.place(relx=0.08, rely=0.009, relheight=0.035, relwidth=0.03)

    lb2 = tk.Label(root, text="数据库类型: ", font=('Arial', 11))
    lb2.place(relx=0.12, rely=0.002, relheight=0.05, relwidth=0.075)
    dataset_options_list = ["cifar", "mnist"]
    global dataset_sel_value
    dataset_sel_value = tk.StringVar(root)
    dataset_sel_value.set(args.dataset)
    dataset_question_menu = tk.OptionMenu(root, dataset_sel_value, *dataset_options_list)
    dataset_question_menu.place(relx=0.2, rely=0.005, relheight=0.04, relwidth=0.06)

    lb3 = tk.Label(root, text="模型类型:", font=('Arial', 11))
    lb3.place(relx=0.28, rely=0.005, relheight=0.04, relwidth=0.06)
    model_options_list = ["cnn", "mlp"]
    global model_sel_value
    model_sel_value = tk.StringVar(root)
    model_sel_value.set(args.model)
    model_question_menu = tk.OptionMenu(root, model_sel_value, *model_options_list)
    model_question_menu.place(relx=0.35, rely=0.005, relheight=0.04, relwidth=0.06)

    iid_options_list = ["IID", "Non-IID"]
    global iid_sel_value
    iid_sel_value = tk.StringVar(root)
    iid_sel_value.set("IID")
    iid_question_menu = tk.OptionMenu(root, iid_sel_value, *iid_options_list)
    iid_question_menu.place(relx=0.42, rely=0.005, relheight=0.04, relwidth=0.09)

    lb4 = tk.Label(root, text="客户分数:", font=('Arial', 11))
    lb4.place(relx=0.002, rely=0.09, relheight=0.04, relwidth=0.06)
    global frac_clients_text
    frac_clients_text = tk.StringVar()
    frac_clients_text.set(str(args.frac))
    frac_clients = tk.Entry(root, textvariable=frac_clients_text)
    frac_clients.place(relx=0.065, rely=0.095, relheight=0.035, relwidth=0.03)

    lb5 = tk.Label(root, text="Rounds(n):", font=('Arial', 11))
    lb5.place(relx=0.11, rely=0.09, relheight=0.04, relwidth=0.065)
    global nrounds_text
    nrounds_text = tk.StringVar()
    nrounds_text.set(str(args.epochs))
    nrounds = tk.Entry(root, textvariable=nrounds_text)
    nrounds.place(relx=0.19, rely=0.095, relheight=0.035, relwidth=0.03)

    lb6 = tk.Label(root, text="Local Epochs(E):", font=('Arial', 11))
    lb6.place(relx=0.22, rely=0.09, relheight=0.04, relwidth=0.15)
    global nepochs_text
    nepochs_text = tk.StringVar()
    nepochs_text.set(str(args.local_ep))
    nepochs = tk.Entry(root, textvariable=nepochs_text)
    nepochs.place(relx=0.36, rely=0.095, relheight=0.035, relwidth=0.03)

    lb7 = tk.Label(root, text="Batch Size(B):", font=('Arial', 11))
    lb7.place(relx=0.385, rely=0.09, relheight=0.04, relwidth=0.13)
    global batchsize_text
    batchsize_text = tk.StringVar()
    batchsize_text.set(str(args.local_bs))
    batchsize = tk.Entry(root, textvariable=batchsize_text)
    batchsize.place(relx=0.51, rely=0.095, relheight=0.035, relwidth=0.03)

    lb8 = tk.Label(root, text="Learning Rate(lr):", font=('Arial', 11))
    lb8.place(relx=0.535, rely=0.09, relheight=0.04, relwidth=0.13)
    global lrrate_text
    lrrate_text = tk.StringVar()
    lrrate_text.set(str(args.lr))
    lrrate = tk.Entry(root, textvariable=lrrate_text)
    lrrate.place(relx=0.67, rely=0.095, relheight=0.035, relwidth=0.05)

    global textbox
    textbox = tk.Text(root, font=('Arial', 10), height=11, width=120)
    textbox.place(relx=0.02, rely=0.6)

    confirmbtn = tk.Button(root, text="开始训练", font=('Arial', 11), command=lambda: starttrain())
    confirmbtn.place(relx=0.85, rely=0.8, relheight=0.05, relwidth=0.1)

    showimagesbtn = tk.Button(root, text="显示图片", font=('Arial', 11), command=lambda: showimages())
    showimagesbtn.place(relx=0.85, rely=0.7, relheight=0.05, relwidth=0.1)

    global showclientsimagesbtn
    showclientsimagesbtn = tk.Button(root, text="向客户端显示图像", font=('Arial', 11),
                                     command=lambda: clients_showimages())
    showclientsimagesbtn.place(relx=0.83, rely=0.9, relheight=0.05, relwidth=0.14)
    showclientsimagesbtn.config(state="disabled")

    lb9 = tk.Label(root, text="客户端查看数据: ", font=('Arial', 10))
    lb9.place(relx=0.002, rely=0.18, relheight=0.04, relwidth=0.1)
    global clientno_text, clientno
    clientno_text = tk.StringVar()
    clientno_text.set(str(args.num_users - 1))
    clientno = tk.Entry(root, textvariable=clientno_text)
    clientno.place(relx=0.1, rely=0.18, relheight=0.04, relwidth=0.04)
    clientno.config(state="disabled")


def starttrain():
    global clients_list
    args.num_users = int(no_clients_text.get())
    args.dataset = dataset_sel_value.get()

    args.model = model_sel_value.get()
    if args.dataset == "cifar":
        optval = "store_false"
    elif iid_sel_value.get() == "IID":
        optval = "store_false"
    else:
        optval = "store_true"
    args.iid = optval
    args.frac = float(frac_clients_text.get())
    args.epochs = int(nrounds_text.get())
    args.local_ep = int(nepochs_text.get())
    args.local_bs = int(batchsize_text.get())
    args.lr = float(lrrate_text.get())
    createlogs()
    textbox.insert(1.0, f"""{args.lr}\n""")
    splitdataset(args)
    clients_list = create_clients(args, dataset_train, dict_users)
    clientno.config(state="normal")
    showclientsimagesbtn.config(state="normal")


def createlogs():
    log_root = "./log"
    os.makedirs(log_root, exist_ok=True)
    global log_path
    log_path = os.path.join(log_root,
                            f"{args.model}-{args.dataset}-B={args.local_bs}-E={args.local_ep}-客户={args.num_users}"
                            f"-{'iid' if args.iid else 'noniid'}-rounds={args.epochs}")
    os.makedirs(log_path, exist_ok=True)


def splitdataset(args):
    global dataset_train
    global dataset_test
    global dict_users
    global img_size
    dataset_train, dataset_test, dict_users = dataset_split()


def dataset_split():
    args.dataset = dataset_sel_value.get()
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307,),
                                                               (0.3081,))])
        datadir = os.path.join(cfilepath, "data", "mnist")
        dataset_train2 = datasets.MNIST(root=datadir, train=True, download=True, transform=trans_mnist)
        dataset_test2 = datasets.MNIST(root=datadir, train=False, download=True, transform=trans_mnist)
        #augmentimages(dataset_train2)
        # sample users
        if args.iid:
            dict_users2 = mnist_iid(dataset_train2, args.num_users)
        else:
            dict_users2 = mnist_noniid(dataset_train2, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                transforms.RandomRotation(10),
                                                transforms.RandomCrop(28),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4, 0.4, 0.4), (0.4, 0.4, 0.4))])
        trans_cifar_test = transforms.Compose([transforms.RandomCrop(28),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.4, 0.4, 0.4), (0.4, 0.4, 0.4))])
        datadir = os.path.join(cfilepath, "data", "cifar")
        dataset_train2 = datasets.CIFAR10(root=datadir, train=True, download=True, transform=trans_cifar_train)
        dataset_test2 = datasets.CIFAR10(root=datadir, train=False, download=True, transform=trans_cifar_test)
        #(dataset_train2)
        if args.iid:
            dict_users2 = cifar_iid(dataset_train2, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    return dataset_train2, dataset_test2, dict_users2


def showimages(num=num):
    args.dataset = dataset_sel_value.get()
    if args.dataset == "cifar":
        labels = ["airplane", "automobile", "bird", "cat", "deer",
                  "dog", "frog", "horse", "ship", "truck"]
    else:
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    dataset_train2, dataset_test2, dict_users2 = dataset_split()
    fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(5, 5))
    warnings.filterwarnings("ignore")
    for i in range(num, num + 36):
        img_size = dataset_train2[i][0].shape
        img = np.array(dataset_train2[i][0]).transpose([1, 2, 0])
        img = np.rot90(img)
        # print("Shape= ", img.shape)
        # axes[0,i].plot(img, 'b')
        axes[int((i % 36) / 6), i % 6].imshow(img)
        axes[int((i % 36) / 6), i % 6].set_title(labels[dataset_train2[i][1]])
        axes[int((i % 36) / 6), i % 6].axis('off')
    plt.show()
    num = num + 36


def clients_showimages():
    try:
        if clientno.__getstate__() != "disabled":
            client_num_fake = clientno_text.get().strip()
            client_num = int(clientno_text.get())
            if client_num >= args.num_users:
                messagebox.showerror(title="客户端号码输入错误",
                                     message=f'请输入一个介于0和{args.num_users - 1}之间的值')
                return
        else:
            return ValueError
    except Exception as e:
        messagebox.showerror(message=f"以 10 为基数的 int（） 无效文字：{client_num_fake}")
        return
    selClient = clients_list[client_num]
    args.dataset = dataset_sel_value.get()
    if args.dataset == "cifar":
        labels = ["airplane", "automobile", "bird", "cat", "deer",
                  "dog", "frog", "horse", "ship", "truck"]
    else:
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(5, 5))
    warnings.filterwarnings("ignore")
    cdataset = selClient.train_data
    count = 0
    print(f"selected client title: {selClient.title}")
    for i, j in enumerate(cdataset):
        if count > 35:
            break
        img_size = j[0][0].shape
        print(f"current dataset enum value: {i}, Length: {len(j[0])}")
        for image_index in range(len(j[0])):
            img = np.array(j[0][image_index]).transpose([1, 2, 0])
            img = np.rot90(img)
            # print("Shape= ", img.shape)
            # axes[0,i].plot(img, 'b')
            axes[int(count / 6), count % 6].imshow(img)
            axes[int(count / 6), count % 6].set_title(labels[j[1][image_index]])
            axes[int(count / 6), count % 6].axis('off')
            count += 1
            if count > 35:
                break
    plt.show()


def augmentimages(dataset_train2):
    # b = list(dataset_train2)
    # print(len(b))
    for i in range(0, 5000):
        img = np.array(dataset_train2[i][0]).transpose([1, 2, 0])
        img = np.rot90(img)
        img = img.reshape(3072)
        dataset_train2[i + 50000][0] = img
        dataset_train2[i + 50000][1] = dataset_train2[i][1]


if __name__ == "__main__":
    global cfilepath
    cfilepath = createPath()
    root = tk.Tk()
    args = args_parser()
    initGUI(root, args)
    objlist = [no_clients_text, dataset_sel_value, model_sel_value,
               iid_sel_value, frac_clients_text, nrounds_text,
               nepochs_text, batchsize_text, lrrate_text]

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    root.mainloop()

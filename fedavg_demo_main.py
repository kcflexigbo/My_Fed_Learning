import matplotlib

#matplotlib.use('Agg')
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

# torch.backends.cudnn.enabled = False
import torch.multiprocessing as mp
import os
from mttkinter import mtTkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import *
from PIL import Image, ImageTk
from utils.options import args_parser
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.sharedata import create_clients
from utils.filesbrowser import createPath, createlogfiles
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Federated import FedAvg
from models.test import test_img
import warnings
import threading
from time import time
from time import sleep
import multiprocessing as mp2


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

    global multi_process_option, multi_process_check
    mlb = tk.Label(root, text="用多处理: ", font=('Arial', 11))
    mlb.place(relx=0.52, rely=0.005, relheight=0.04, relwidth=0.07)
    multi_process_option = tk.StringVar()
    multi_process_option.set("1")
    multi_process_check = tk.Checkbutton(root, variable=multi_process_option, onvalue=1, offvalue=0,
                                         command=multi_process_info, )
    multi_process_check.place(relx=0.585, rely=0.005, relheight=0.04, relwidth=0.06)
    multi_process_check.deselect()

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
    textbox.config(state="disabled")

    global confirmbtn
    confirmbtn = tk.Button(root, text="开始训练", font=('Arial', 11), command=starttrainthread)
    confirmbtn.place(relx=0.85, rely=0.8, relheight=0.05, relwidth=0.1)

    showimagesbtn = tk.Button(root, text="显示图片", font=('Arial', 11), command=showimages)
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

    # lb10 = tk.Label(root, text="testme!!!", font=('Arial', 10))
    # lb10.place(relx=0.15, rely=0.18, relheight=0.04, relwidth=0.1)
    # global testme_text
    # testme_text = tk.StringVar()
    # testme_text.set(str(0.5))
    # testme = tk.Entry(root, textvariable=testme_text)
    # testme.place(relx=0.26, rely=0.18, relheight=0.04, relwidth=0.04)


def multi_process_info():
    optional_val = int(multi_process_option.get())
    if optional_val == 1:
        messagebox.showinfo(title="多处理",
                            message="暂时不建议。 时速度较慢并且第一次运行有错误。 改进正在进行中。 请在实际训练前跑一次。")


def starttrainthread():
    global use_thread_opt
    use_thread_opt = int(multi_process_option.get())
    multi_process_check.config(state="disabled")
    train_thread = threading.Thread(target=start_train)
    train_thread.daemon = True
    train_thread.start()


def start_train():
    global start_time, end_time
    start_time = time()
    confirmbtn.config(state="disabled")
    showclientsimagesbtn.config(state="disabled")
    global loss_train_file_obj, acc_file_obj, loss_file_obj, clients_list
    global loss_train_plot_list, acc_plot_list, loss_test_plot_list
    loss_train_plot_list, acc_plot_list, loss_test_plot_list = [], [], []
    assign_values()
    createlogs()
    textbox.config(state="normal")
    textbox.delete(1.0, "end")
    textbox.config(state="disabled")
    splitdataset(args)
    clients_list = create_clients(args, dataset_train, dict_users, client_path)
    start_time = time()
    createmodel()
    net_glob.train()
    loss_train_file_obj, acc_file_obj, loss_file_obj = createlogfiles(log_path, net_glob)
    textbox.config(state="normal")
    textbox.insert(1.0, f'Fraction= {args.frac}\n')
    textbox.insert(1.0, f"""{net_glob}\n""")
    textbox.config(state="disabled")
    model_training()
    end_time = time()
    loss_train_file_obj.close()
    acc_file_obj.close()
    loss_file_obj.close()
    # plot_graphs()
    final_test()
    writetimereport()
    clientno.config(state="normal")
    showclientsimagesbtn.config(state="normal")
    confirmbtn.config(state="normal")
    multi_process_check.config(state="normal")


def assign_values():
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


def writetimereport():
    timer_file = os.path.join(log_path, 'timer_log.txt')
    timer_file_obj = open(timer_file, 'w')
    timer_string = f"Time Taken= {(end_time - start_time) // 60:.0f} mins, {(end_time - start_time) % 60:.2f} secs\n"
    timer_file_obj.write(timer_string)
    timer_file_obj.close()
    textbox.config(state="normal")
    textbox.insert(1.0, timer_string)
    textbox.config(state="disabled")


def model_training():
    global acc_plot_list, loss_train_plot_list, loss_test_plot_list
    loss_train_plot_list = []
    loss_test_plot_list = []
    acc_plot_list = []
    num_clients_for_each_round = max(int(args.frac * args.num_users), 1)
    for c_round in range(args.epochs):
        local_models, loss_locals = client_train(num_clients_for_each_round)
        w_glob = FedAvg(local_models)
        net_glob.load_state_dict(w_glob)
        printloss(c_round, loss_locals)


def client_train(num_clients_for_each_round):
    global client_process
    loss_locals = []
    local_models = []
    client_thread = []
    client_process = []
    client_run_list = []
    idxs_users = np.random.choice(range(args.num_users), num_clients_for_each_round, replace=False)
    # client_queue = mp.Manager().Queue()
    # client_queue = mp2.Manager().Queue()
    client_queue = mp.Queue()
    if use_thread_opt == 1:
        for idx in idxs_users:
            client_run_list.append(clients_list[idx])
            new_net_glob = copy.deepcopy(net_glob).to(args.device)
            new_proc = mp.Process(target=clients_list[idx].train,
                                  args=(new_net_glob, client_queue, start_time))
            client_process.append(new_proc)
            new_proc.start()

        for proc in client_process:
            # if proc.is_alive():
            #     print(f"Process still alive {client_process.index(proc)}")
            proc.join()

        for idx in range(len(idxs_users)):
            try:
                # cmod = torch.load(client_run_list[idx].model_path, map_location=torch.device(args.device))
                cmod = torch.load(client_run_list[idx].model_path, map_location=torch.device(args.device))
            except FileNotFoundError:
                print("File not found")
            try:
                if os.path.exists(client_run_list[idx].model_path):
                    os.remove(client_run_list[idx].model_path)
            except Exception as e:
                print(e)
            # cmod.eval()
            closs = client_queue.get()
            # print(closs)
            loss_locals.append(closs)
            # local_models.append(cmod.state_dict())
            local_models.append(cmod)
    else:
        for idx in idxs_users:
            client_run_list.append(clients_list[idx])
            new_net_glob = copy.deepcopy(net_glob).to(args.device)
            l_model, l_loss = clients_list[idx].train(new_net_glob, client_queue, start_time, use_multiprocessing=False)
            local_models.append(copy.deepcopy(l_model))
            loss_locals.append(l_loss)
    return local_models, loss_locals


def printloss(c_round, loss_locals):
    loss_avg = sum(loss_locals) / len(loss_locals)
    acc_test, test_loss = test_img(net_glob, dataset_test, args)
    # print('Round {:3d}: Average Train loss {:.3f}, Accuracy={:.3f}, Model Test Loss={:.3f}'.format(iter, loss_avg,
    #                                                                                                acc_test,
    #                                                                                                test_loss))
    textbox.config(state="normal")
    textbox.insert(1.0,
                   'Round {:3d}: Average Train loss {:.3f}, Accuracy={:.3f}, Model Test Loss={:.3f}\n'.format(c_round,
                                                                                                              loss_avg,
                                                                                                              acc_test,
                                                                                                              test_loss))
    textbox.config(state="disabled")
    loss_train_plot_list.append(loss_avg)
    acc_plot_list.append(acc_test)
    loss_test_plot_list.append(test_loss)
    try:
        loss_train_file_obj.write(str(loss_avg) + '\n')
        loss_train_file_obj.flush()

        acc_file_obj.write(str(acc_test) + '\n')
        acc_file_obj.flush()

        loss_file_obj.write(str(test_loss) + '\n')
        loss_file_obj.flush()
    except Exception as e:
        # print("Error while writing to file. Please check and close all log files.")
        messagebox.showerror(title="Error while Writing To File", message="Error while writing to file. Please "
                                                                          "check and close all log files.")


def createlogs():
    log_root = "./log"
    try:
        os.makedirs(log_root, exist_ok=True)
    except Exception as e:
        messagebox.showerror(title="Error creating models folder", message=f"{e}. Can not make folder")
    global log_path
    log_path = os.path.join(log_root,
                            f"{args.model}-{args.dataset}-B={args.local_bs}-E={args.local_ep}-客户={args.num_users}"
                            f"-{'iid' if args.iid else 'noniid'}-rounds={args.epochs}")
    try:
        os.makedirs(log_path, exist_ok=True)
    except Exception as e:
        messagebox.showerror(title="Error creating models folder", message=f"{e}. Can not make folder")
    global client_path
    client_path = os.path.join(log_path, "models")
    try:
        os.makedirs(client_path, exist_ok=True)
    except OSError as error:
        messagebox.showerror(title="Error creating models folder", message=f"{e}. Can not make folder")


def splitdataset(args):
    global dataset_train
    global dataset_test
    global dict_users
    global img_size
    dataset_train, dataset_test, dict_users = dataset_split()


def dataset_split():
    global img_size
    args.dataset = dataset_sel_value.get()
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307,),
                                                               (0.3081,))])
        datadir = os.path.join(cfilepath, "data", "mnist")
        dataset_train2 = datasets.MNIST(root=datadir, train=True, download=True, transform=trans_mnist)
        dataset_test2 = datasets.MNIST(root=datadir, train=False, download=True, transform=trans_mnist)
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
    img_size = dataset_train2[0][0].shape
    return dataset_train2, dataset_test2, dict_users2


def createmodel():
    global net_glob
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    # net_glob.train()
    # return net_glob


def showimages():
    args.dataset = dataset_sel_value.get()
    if args.dataset == "cifar":
        labels = ["airplane", "automobile", "bird", "cat", "deer",
                  "dog", "frog", "horse", "ship", "truck"]
    else:
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    dataset_train2, dataset_test2, dict_users2 = dataset_split()
    fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(5, 5))
    warnings.filterwarnings("ignore")
    for i in range(36):
        img_size = dataset_train2[i][0].shape
        img = np.array(dataset_train2[i][0]).transpose([1, 2, 0])
        img = np.rot90(img)
        axes[int((i % 36) / 6), i % 6].imshow(img)
        axes[int((i % 36) / 6), i % 6].set_title(labels[dataset_train2[i][1]])
        axes[int((i % 36) / 6), i % 6].axis('off')
    plt.show()


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
    for i, j in enumerate(cdataset):
        if count > 35:
            break
        img_size = j[0][0].shape
        for image_index in range(len(j[0])):
            img = np.array(j[0][image_index]).transpose([1, 2, 0])
            img = np.rot90(img)
            axes[int(count / 6), count % 6].imshow(img)
            axes[int(count / 6), count % 6].set_title(labels[j[1][image_index]])
            axes[int(count / 6), count % 6].axis('off')
            count += 1
            if count > 35:
                break
    plt.show()


def plot_graphs():
    print(len(loss_train_plot_list))
    plt.figure(1)
    plt.plot(range(len(loss_train_plot_list)), loss_train_plot_list)
    plt.ylabel('train_loss')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(log_path, 'Client_Avg_Train_Loss.png'))

    #Plot Accuracy curve
    plt.figure(2)
    plt.plot(range(len(acc_plot_list)), acc_plot_list)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(log_path, 'accuracy.png'))

    #Plot Model Loss Curve
    plt.figure(3)
    plt.plot(range(len(loss_test_plot_list)), loss_test_plot_list)
    plt.ylabel('Model Loss')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(log_path, 'Model_Test_Loss.png'))


def final_test():
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    textbox.config(state="normal")
    textbox.insert(1.0, "Training accuracy: {:.2f}\n".format(acc_train))
    textbox.insert(1.0, "Testing accuracy: {:.2f}\n".format(acc_test))
    textbox.config(state="disabled")


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

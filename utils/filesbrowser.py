import userpaths as us
import os


def createPath():
    """Create New Folder for project in documents"""
    # config_filepath = us.get_appdata()
    # config_filepath = os.path.join(os.path.join(config_filepath, "New Federated Learning"))
    #
    # try:
    #     os.makedirs(config_filepath, exist_ok=True)
    #     #print("Directory Created Successfully")
    # except OSError as error:
    #     print("Directory already Exists and Not Created")

    #Get users documents location using the userpaths library
    cfilepath = us.get_my_documents()
    cfilepath = os.path.join(cfilepath, 'New Federated Learning', 'My Fed Learning')

    #Create the directory if not existing, else continue
    try:
        os.makedirs(cfilepath, exist_ok=True)
        #print("Directory Created Successfully")
    except OSError as error:
        print("Directory already Exists and Not Created")
    return cfilepath


def createlogfiles(log_path, net_glob):
    # Client training Average Loss File
    loss_train_file = os.path.join(log_path, "loss_train_file.txt")
    loss_train_file_obj = open(loss_train_file, "w")

    # For early stopping
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    # Accuracy File
    acc_file = os.path.join(log_path, 'acc_log.txt')
    acc_file_obj = open(acc_file, 'w')

    # Global Loss File
    loss_file = os.path.join(log_path, 'loss_log.txt')
    loss_file_obj = open(loss_file, 'w')

    # Model Properties file
    model_properties_file = os.path.join(log_path, 'model_properties.txt')
    model_properties_file_obj = open(model_properties_file, 'w')
    model_properties_file_obj.write(str(net_glob))
    model_properties_file_obj.close()
    return loss_train_file_obj, acc_file_obj, loss_file_obj


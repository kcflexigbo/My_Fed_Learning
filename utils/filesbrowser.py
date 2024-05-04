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
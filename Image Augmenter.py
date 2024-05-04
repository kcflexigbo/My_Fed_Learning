import numpy as np


def rotate_image(img, angle=90):
    # b = list(dataset_train2)
    # print(len(b))
    # for i in range(0, 5000):
    #     img = np.array(dataset_train2[i][0]).transpose([1, 2, 0])
    #     img = np.rot90(img)
    #     img = img.reshape(3072)
    #     dataset_train2[i + 50000][0] = img
    #     dataset_train2[i + 50000][1] = dataset_train2[i][1]

    img = np.array(img).transpose((2, 0, 1))
    try:
        if angle == 90:
            img = np.rot90(img)
        elif angle == 180:
            img = np.rot90(img)
            img = np.rot90(img)
        elif angle == 270:
            img = np.rot90(img)
            img = np.rot90(img)
            img = np.rot90(img)
        elif type(angle) == int:
            img = np.rot90(img)
    except ValueError:
        print("Value Error")
    img = img.reshape(3072)
    return img

def zoom_image(img, zoom_x, zoom_y):
    img = np.array(img).transpose((2, 0, 1))

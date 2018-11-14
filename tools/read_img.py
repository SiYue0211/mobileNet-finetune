import cv2
import numpy as np


def process_line(line):
    info = line.strip().split('\t')
    y = info[1]
    x = cv2.imread(info[0])
    if x is None:
        raise Exception("{} img is None".format(info[0]))
    if x.shape[-1] != 3:
        raise Exception("{} img channel is not 3".format(info[0]))
    x_reshape = cv2.resize(x, (128, 128, 3))
    # TODO: 去均值，归一化
    return x_reshape, y

def generate_arrays_from_file(path, batch_size):
    while 1:
        with open(path) as f:
            cnt = 0
            X = []
            Y = []
            for line in f:
                x, y = process_line(line)
                X.append(x)
                Y.append(y)
                cnt += 1
                if cnt == batch_size:
                    cnt = 0
                    yield (np.array(X), np.array(Y))
                    X = []
                    Y = []


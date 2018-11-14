import tensorflow as tf
import os
import sys
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import Input, Reshape, AvgPool2D, Dropout, Conv2D, Softmax, BatchNormalization, Activation
from tensorflow.keras import Model

rootPath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
sys.path.insert(0, rootPath)

from tools.read_img import generate_arrays_from_file

# 加载预训练参数
base_model = MobileNet(input(128, 128, 3), include_top=False)

with tf.name_scope("output"):
    x = base_model.get_layer("conv_dw_6_relu").output
    x = Conv2D(256, kernel_size=(3, 3))(x)
    x = Activation("relu")(x)
    x = AvgPool2D(pool_size=(5, 5))(x)
    x = Dropout(rate=0.5)(x)
    x = Conv2D(2, kernel_size=(1, 1))(x)
    predictions = Reshape((2, ))(x)

model = Model(inputs=base_model.input, output=predictions)

# 冻结原始层，在编译后生效
for layer in base_model.layers:
    layer.trainable = False

sgd = tf.keras.optimizers.SGD(lr=0.01)
model.compile(optimizer=sgd, loss='categorical_crossentrogy')
model.fit_generator(generate_arrays_from_file(os.path.join(rootPath, 'dataset', 'train.txt')), steps_per_epoch=10, \
                    epoch=3, verbose=1)


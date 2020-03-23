# @File  : make_3dcnn.py
# @Author: LiBin
# @Date  : 2020/3/20
# @Desc  :
from keras.layers import *
from keras.metrics import binary_accuracy, binary_crossentropy
from keras.models import Model
from keras.optimizers import Adam

class _3Dcnn(object):
    def __init__(self,shape):
        self.input_shape=shape

    def make_3dnet(self,load_weight_path=None, USE_DROPOUT=True) -> Model:#input_shape=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 1)
        inputs = Input(shape=self.input_shape, name="input_1")
        x = inputs
        x = AveragePooling3D(pool_size=(2, 1, 1), strides=(2, 1, 1), padding="same")(x)
        x = Conv3D(64, kernel_size=3, padding='same', name='conv1', subsample=(1, 1, 1))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid', name='pool1')(x)

        # 2nd layer group
        x = Conv3D(128, kernel_size=3, padding='same', name='conv2', subsample=(1, 1, 1))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool2')(x)
        if USE_DROPOUT:
            x = Dropout(p=0.3)(x)

        # 3rd layer group
        x = Conv3D(256, kernel_size=3, activation='relu', padding='same', name='conv3a', subsample=(1, 1, 1))(x)
        x = Conv3D(256, kernel_size=3, activation='relu', padding='same', name='conv3b', subsample=(1, 1, 1))(x)
        x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool3')(x)
        if USE_DROPOUT:
            x = Dropout(p=0.4)(x)

        # 4th layer group
        x = Conv3D(512, kernel_size=3, activation='relu', padding='same', name='conv4a', subsample=(1, 1, 1))(x)
        x = Conv3D(512, kernel_size=3, activation='relu', padding='same', name='conv4b', subsample=(1, 1, 1), )(x)
        x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool4')(x)
        if USE_DROPOUT:
            x = Dropout(p=0.5)(x)

        last64 = Conv3D(64, kernel_size=2, activation="relu", name="last_64")(x)
        out_class = Conv3D(1, kernel_size=1, activation="sigmoid", name="out_class_last")(last64)
        out_class = Flatten(name="out_class")(out_class)
        model = Model(input=inputs, output=[out_class])


        model.compile(optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                      loss={"out_class": "binary_crossentropy"},
                      metrics={"out_class": [binary_accuracy, binary_crossentropy]})
        return model


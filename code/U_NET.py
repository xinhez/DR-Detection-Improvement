import numpy as np
import os
import argparse
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.merge import Concatenate
from deep_preprocess.classifier import Classifier
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, Lambda, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator, array_to_img

from keras import backend as K


class DeepPreprocess:

    def __init__(self, nrow=512, ncol=512, nch=3):
        if K.image_data_format() == 'channels_first':
            axis = 1
            input_img = Input(shape=(nch, nrow, ncol))
        else:
            axis = -1
            input_img = Input(shape=(nrow, ncol, nch))

        nf = 16
        conv1 = Conv2D(nf, (3, 3), activation="relu", padding="same")(input_img)
        conv1 = Conv2D(nf, (3, 3), activation="relu", padding="same")(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(nf*2, (3, 3), activation="relu", padding="same")(pool1)
        conv2 = Conv2D(nf*2, (3, 3), activation="relu", padding="same")(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(nf*4, (3, 3), activation="relu", padding="same")(pool2)
        conv3 = Conv2D(nf*4, (3, 3), activation="relu", padding="same")(conv3)
        conv3 = BatchNormalization(axis=axis)(conv3)

        up8 = UpSampling2D(size=(2, 2))(conv3)
        up8 = Concatenate(axis=axis)([Conv2D(nf*2, (2, 2), activation="relu", padding="same")(up8), conv2])
        conv8 = Conv2D(nf*2, (3, 3), activation="relu", padding="same")(up8)
        conv8 = BatchNormalization(axis=axis)(conv8)
        conv8 = Conv2D(nf*2, (3, 3), activation="relu", padding="same")(conv8)
        conv8 = BatchNormalization(axis=axis)(conv8)

        up9 = UpSampling2D(size=(2, 2))(conv8)
        up9 = Concatenate(axis=axis)([Conv2D(nf, (2, 2), activation="relu", padding="same")(up9), conv1])
        conv9 = Conv2D(nf, (3, 3), activation="relu", padding="same")(up9)
        conv9 = Conv2D(nf, (3, 3), activation="relu", padding="same")(conv9)

        t = Conv2D(1, (1, 1), activation="sigmoid", padding="same")(conv9)

        def color_balance(t):
            t_thresh = t * 0.9 + 0.1
            t_rep = K.repeat_elements(t_thresh, 3, axis)
            img_cb = 1 - (((1 - input_img) - (1 - t_rep)) / t_rep)

            img_cb = img_cb - K.min(img_cb)
            return img_cb / K.max(img_cb)

        img_cb = Lambda(color_balance)(t)
        img_cb_norm = Lambda(lambda x: (x * 2) - 1)(img_cb)

        self.classifier = Classifier(n_features=1024)

        out = self.classifier.model(img_cb_norm)
        self.model = Model(input_img, out)
        self.preprocess_models = Model(input_img, [img_cb, t])

    def prepare_to_init(self):
        for layer in self.classifier.base_model.layers:
            layer.trainable = False

        opt = Adam(lr=2e-4)
        self.model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        return self.model

    def prepare_to_finetune(self):
        for layer in self.classifier.base_model.layers:
            layer.trainable = True

        opt = Adam(lr=2e-4)
        self.model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        return self.model


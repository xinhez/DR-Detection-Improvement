import numpy as np
import os
import re
from scipy import ndimage, misc
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, Lambda
from keras.preprocessing.image import ImageDataGenerator, array_to_img
from keras.layers.merge import Concatenate
from keras.models import Model
from deep_preprocess.classifier import Classifier

from keras import backend as K

train_data_dir = "D:\\work\\datasets\\dr\\Messidor\\mess\\train"
val_data_dir = "D:\\work\\datasets\\dr\\Messidor\\mess\\val"
test_data_dir = "D:\\work\\datasets\\dr\\Messidor\\mess\\test"

"""train_data_dir = "/Volumes/Karnik/data/train/normal"
val_data_dir = "/Volumes/Karnik/data/val/normal"
test_data_dir = "/Volumes/Karnik/data/test/normal"
"""
"""train_data_patho = "/Volumes/Karnik/data/train/pathological"
val_data_patho = "/Volumes/Karnik/data/val/pathological"
test_data_patho = "/Volumes/Karnik/data/test/pathological"
"""


"""images = []
for root, dirnames, filenames in os.walk("/Volumes/Karnik/data/test/normal/images"):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            filepath = os.path.join(root, filename)
            image = ndimage.imread(filepath, mode="RGB")
            image_resized = misc.imresize(image, (512, 512))
            images.append(image_resized)
im = np.asarray(images)

images_patho = []
for root, dirnames, filenames in os.walk("/Volumes/Karnik/data/test/pathological"):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            filepath = os.path.join(root, filename)
            image = ndimage.imread(filepath, mode="RGB")
            image_resized = misc.imresize(image, (512, 512))
            images_patho.append(image_resized)
im_patho = np.asarray(images_patho)"""

def u_net():
    nrow = 512
    ncol = 512
    nch = 3

    if K.image_data_format() == 'channels_first':
        axis = 1
        input_img = Input(shape=(nch, nrow, ncol))
    else:
        axis = -1
        input_img = Input(shape=(nrow, ncol, nch))

    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv3)
    # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    """conv4 = Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer="uniform")(pool3)
    conv4 = Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer="uniform")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same", kernel_initializer="uniform")(pool4)
    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same", kernel_initializer="uniform")(conv5)

    up6 = Concatenate(axis=axis)([Conv2D(256, (2, 2), activation="relu", padding="same", kernel_initializer="uniform")(UpSampling2D(size=(2, 2))(conv5)), conv4])
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer="uniform")(up6)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer="uniform")(conv6)

    up7 = Concatenate(axis=axis)([Conv2D(128, (2, 2), activation="relu", padding="same", kernel_initializer="uniform")(UpSampling2D(size=(2, 2))(conv6)), conv3])
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same", kernel_initializer="uniform")(up7)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same", kernel_initializer="uniform")(conv7)"""

    up8 = Concatenate(axis=axis)([Conv2D(64, (2, 2), activation="relu", padding="same")(UpSampling2D(size=(2, 2))(conv3)), conv2])
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(up8)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv8)

    up9 = Concatenate(axis=axis)([Conv2D(32, (2, 2), activation="relu", padding="same")(UpSampling2D(size=(2, 2))(conv8)), conv1])
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(up9)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv9)

    t = Conv2D(1, (1, 1), activation="sigmoid", padding="same")(conv9)

    def color_balance(t):
        # Icb = 1 - [((1 - I) - A(1 - t)) / t]
        t_rep = K.repeat_elements(t, 3, axis)
        return 1 - (((1 - input_img) - (1 - t_rep)) / (t_rep + K.epsilon()))

    img_cb = Lambda(color_balance)(t)
    img_cb_norm = Lambda(lambda x: (x * 255) - 127)(img_cb)

    classifier = Classifier()
    for layer in classifier.base_model.layers:
        layer.trainable = False

    model = Model(input_img, classifier.model(img_cb_norm))

    opt = Adam(lr=1e-3)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    preprocess_model = Model(input_img, img_cb)

    return model, preprocess_model


if __name__ == '__main__':

    target_width = 512
    target_height = 512
    target_ch = 3
    b_size = 4
    e = 20

    train_data = ImageDataGenerator(rescale=1./255)
    val_data = ImageDataGenerator(rescale=1. / 255)
    test_data = ImageDataGenerator(rescale=1. / 255)

    train_gen = train_data.flow_from_directory(train_data_dir, target_size=(target_width, target_height), batch_size=b_size, class_mode='binary')
    val_gen = val_data.flow_from_directory(val_data_dir, target_size=(target_width, target_height), batch_size=b_size, class_mode='binary')
    test_gen = test_data.flow_from_directory(test_data_dir, target_size=(target_width, target_height), batch_size=b_size, class_mode='binary')

    model, preprocess_model = u_net()

    num_train = train_gen.n
    num_val = val_gen.n
    num_test = test_gen.n

    print("Fitting the model...")

    model.fit_generator(generator=train_gen, steps_per_epoch=num_train//b_size, epochs=e, validation_data=val_gen, validation_steps=num_val)

    print("Predicting from the model...")

    test_res = model.predict(im, batch_size=4, verbose=1)
    test_res_patho = model.predict(im_patho, batch_size=4, verbose=1)

    np.save('/Volumes/Karnik/test_res.npy', test_res)
    np.save('/Volumes/Karnik/test_res_patho.npy', test_res_patho)

    print("Converting to image")
    load_array = np.load('/Volumes/Karnik/test_res.npy')
    for l in range(load_array.shape[0]):
        frame = load_array[l]
        frame = array_to_img(frame)
        frame.save("/Volumes/Karnik/result_normal/%d.jpg" % (l))

    load_array = np.load('/Volumes/Karnik/test_res_patho.npy')
    for l in range(load_array.shape[0]):
        frame = load_array[l]
        frame = array_to_img(frame)
        frame.save("/Volumes/Karnik/result_patho/%d.jpg" % (l))

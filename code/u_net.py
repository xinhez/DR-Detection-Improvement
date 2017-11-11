import numpy as np
import os
import re
from scipy import ndimage, misc
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from keras.preprocessing.image import ImageDataGenerator, array_to_img
from keras.layers.merge import Concatenate
from keras.models import Model

from keras import backend as K

train_data_dir = "/Volumes/SANIYA/NOW_LAPPY/Project/image_pre/code/data/train/normal"
val_data_dir = "/Volumes/SANIYA/NOW_LAPPY/Project/image_pre/code/data/val/normal"
test_data_dir = "/Volumes/SANIYA/NOW_LAPPY/Project/image_pre/code/data/test/normal"


images = []
for root, dirnames, filenames in os.walk("/Volumes/SANIYA/NOW_LAPPY/Project/image_pre/code/data/test/normal/images"):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            filepath = os.path.join(root, filename)
            image = ndimage.imread(filepath, mode="RGB")
            image_resized = misc.imresize(image, (512, 512))
            images.append(image_resized)

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

    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same", kernel_initializer="uniform")(input_img)
    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same", kernel_initializer="uniform")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer="uniform")(pool1)
    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer="uniform")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same", kernel_initializer="uniform")(pool2)
    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same", kernel_initializer="uniform")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer="uniform")(pool3)
    conv4 = Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer="uniform")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same", kernel_initializer="uniform")(pool4)
    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same", kernel_initializer="uniform")(conv5)
    
    up6 = Concatenate(axis=axis)([Conv2D(256, (2, 2), activation="relu", padding="same", kernel_initializer="uniform")(UpSampling2D(size=(2, 2))(conv5)), conv4])
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer="uniform")(up6)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer="uniform")(conv6)
    
    up7 = Concatenate(axis=axis)([Conv2D(128, (2, 2), activation="relu", padding="same", kernel_initializer="uniform")(UpSampling2D(size=(2, 2))(conv6)), conv3])
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same", kernel_initializer="uniform")(up7)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same", kernel_initializer="uniform")(conv7)
    
    up8 = Concatenate(axis=axis)([Conv2D(64, (2, 2), activation="relu", padding="same", kernel_initializer="uniform")(UpSampling2D(size=(2, 2))(conv7)), conv2])
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer="uniform")(up8)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer="uniform")(conv8)
    
    up9 = Concatenate(axis=axis)([Conv2D(32, (2, 2), activation="relu", padding="same", kernel_initializer="uniform")(UpSampling2D(size=(2, 2))(conv8)), conv1])
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same", kernel_initializer="uniform")(up9)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same", kernel_initializer="uniform")(conv9)
    
    conv10 = Conv2D(1, (1, 1), activation="sigmoid", padding="same", kernel_initializer="uniform")(conv9)
    
    model = Model(inputs=input_img, outputs=conv10)
    
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


def auxiliar_generator(gen):
    """This generator should be used to test the unet implementation only."""
    while True:
        x, _ = next(gen)
        yield x, np.mean(x, axis=-1, keepdims=True)


if __name__ == '__main__':

    target_width = 512
    target_height = 512
    target_ch = 3
    b_size = 4
    e = 1
    
    train_data = ImageDataGenerator(rescale=1./255)
    val_data = ImageDataGenerator(rescale=1. / 255)
    test_data = ImageDataGenerator(rescale=1. / 255)
    
    train_gen = train_data.flow_from_directory(train_data_dir, target_size=(target_width, target_height), batch_size=b_size, class_mode='binary')
    val_gen = val_data.flow_from_directory(val_data_dir, target_size=(target_width, target_height), batch_size=b_size, class_mode='binary')
    test_gen = test_data.flow_from_directory(test_data_dir, target_size=(target_width, target_height), batch_size=b_size, class_mode='binary')
    
    model = u_net()
    
    x, y = next(train_gen)
    
    num_train = train_gen.n
    num_val = val_gen.n
    num_test = test_gen.n
    
    print("Fitting the model...")
    aux_train_gen = auxiliar_generator(train_gen)
    aux_val_gen = auxiliar_generator(val_gen)
    aux_test_gen = auxiliar_generator(test_gen)
    model.fit_generator(generator=aux_train_gen, steps_per_epoch=num_train//b_size, epochs=e, validation_data=aux_val_gen, validation_steps=num_val)
    
    print("Predicting from the model...")
    result=[]
    for j in range(len(images)):
        test_res = model.predict(images[j], batch_size=1, verbose=1)
        result.append(test_res)
    np.save('/Volumes/SANIYA/NOW_LAPPY/Project/image_pre/code/data/test_res.npy', result)
    
    print("Converting to image")
    load_array = np.load('/Volumes/SANIYA/NOW_LAPPY/Project/image_pre/code/data/test_res.npy')
    for l in range(load_array.shape[0]):
        frame = load_array[l]
        frame = array_to_img(frame)
        frame.save("/Volumes/SANIYA/NOW_LAPPY/Project/image_pre/code/data/result/%d.jpg" % (l))

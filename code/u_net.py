import os, os.path
from os.path import join
import numpy as np
import cv2
import keras
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras.layers.merge import Concatenate
from keras.models import Model
from PIL import Image
from matplotlib import pyplot as plt


train_data_dir = "/Volumes/SANIYA/NOW_LAPPY/Project/image_pre/code/data/train/normal"
val_data_dir = "/Volumes/SANIYA/NOW_LAPPY/Project/image_pre/code/data/val/normal"
test_data_dir = "/Volumes/SANIYA/NOW_LAPPY/Project/image_pre/code/data/test/normal"

#To count the number of training and validation examples
def number_images(dirName):
    ims = [name for name in os.listdir(dirName)]
    return ims

all_train = number_images(train_data_dir)
num_train = len(all_train)

all_val = number_images(val_data_dir)
num_val = len(all_val)

all_test = number_images(test_data_dir)
num_test = len(all_test)


#
# all_images = []
# image = np.empty(len(all_ims), dtype=object)
# for i in range(num_train):
#
#     image[i] = cv2.imread(join("/Volumes/SANIYA/NOW_LAPPY/Project/image_pre/code/data/train/normal", all_ims[i]))
#
#     # Resizing the images to 512 x 512 x 3 dimensions
#
#     res = cv2.resize(image[i], (512, 512), interpolation=cv2.INTER_CUBIC)
#
#     # Flattening the images in rows
#     im_flat = res.flatten()
#
#     all_images.append(im_flat)

def u_net():
    nrow = 512
    ncol = 512
    nch = 3
    
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
    
    up6 = Concatenate(axis=1)([Conv2D(256, (2, 2), activation="relu", padding="same", kernel_initializer="uniform")(UpSampling2D(size=(2, 2))(conv5)), conv4])
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer="uniform")(up6)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer="uniform")(conv6)
                                                                                                                    
    up7 = Concatenate(axis=1)([Conv2D(128, (2, 2), activation="relu", padding="same", kernel_initializer="uniform")(UpSampling2D(size=(2, 2))(conv6)), conv3])
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same", kernel_initializer="uniform")(up7)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same", kernel_initializer="uniform")(conv7)
                                                                                                                    
    up8 = Concatenate(axis=1)([Conv2D(64, (2, 2), activation="relu", padding="same", kernel_initializer="uniform")(UpSampling2D(size=(2, 2))(conv7)), conv2])
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer="uniform")(up8)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer="uniform")(conv8)
                                                                                                                    
    up9 = Concatenate(axis=1)([Conv2D(32, (2, 2), activation="relu", padding="same", kernel_initializer="uniform")(UpSampling2D(size=(2, 2))(conv8)), conv1])
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same", kernel_initializer="uniform")(up9)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same", kernel_initializer="uniform")(conv9)
    
    conv10 = Conv2D(1, (1, 1), activation="sigmoid", padding="same", kernel_initializer="uniform")(conv9)
                                                                                                                    
    model = Model(inputs=input_img, outputs=conv10)
                                                                                                                    
    model.compile(optimizer='Adam', lr=1e-4, loss='binary_crossentropy', metrics=['accuracy'])
                                                                                                                    
    return model


if __name__ == '__main__':
    
    target_width=512
    target_height=512
    target_ch =3
    b_size=10
    e=1
    
    train_data = ImageDataGenerator(rescale=1./255)
    val_data = ImageDataGenerator(rescale=1. / 255)
    test_data = ImageDataGenerator(rescale=1. / 255)
    
    train_gen = train_data.flow_from_directory(train_data_dir, target_size= (target_width,target_height,target_ch),batch_size=b_size, class_mode='binary')
    val_gen = val_data.flow_from_directory(val_data_dir, target_size= (target_width,target_height,target_ch),batch_size=b_size, class_mode='binary')
    test_gen = test_data.flow_from_directory(test_data_dir,target_size=(target_width, target_height, target_ch),batch_size=b_size, class_mode='binary')
    
    model = u_net()
    # array =np.array (all_images, dtype=float)
    #
    # reshaped_array=array.reshape(num_train,512,512,3)
    #
    
    x, y = next(train_gen)
    print(x.shape)
    print "Fitting the model..."
    model.fit_generator(generator=(x,y), steps_per_epoch= num_train//b_size, epochs=e, validation_data=val_gen, validation_steps=num_val)
    
    print "Predicting from the model..."
    test_res = model.predict(test_gen, batch_size=1, verbose=1)
    np.save('/Volumes/SANIYA/NOW_LAPPY/Project/image_pre/code/data/test_res.npy',test_res)
    
    print "Converting to image"
    load_array=np.load('/Volumes/SANIYA/NOW_LAPPY/Project/image_pre/code/data/test_res.npy')
    for l in range(load_array.shape[0]):
        frame=load_array[l]
        frame=array_to_img(frame)
        frame.save("/Volumes/SANIYA/NOW_LAPPY/Project/image_pre/code/data/result/%d.jpg" % (l))










import numpy as np
import os
import argparse
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.merge import Concatenate
from deep_preprocess.classifier import Classifier
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, Lambda
from keras.preprocessing.image import ImageDataGenerator, array_to_img

from keras import backend as K

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

    up8 = UpSampling2D(size=(2, 2))(conv3)
    up8 = Concatenate(axis=axis)([Conv2D(64, (2, 2), activation="relu", padding="same")(up8), conv2])
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(up8)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv8)

    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = Concatenate(axis=axis)([Conv2D(32, (2, 2), activation="relu", padding="same")(up9), conv1])
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(up9)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv9)

    t = Conv2D(1, (1, 1), activation="sigmoid", padding="same")(conv9)

    def color_balance(t):
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
    transmission_model = Model(input_img, t)

    return model, preprocess_model, transmission_model


def parse_params():
    """Parse command line arguments and return them."""
    parser = argparse.ArgumentParser(prog='Deep Preprocess', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    """
    Data parameters' definition
    """
    parser.add_argument('--dir', help='The directory where the data is present.', dest='dir', default='/Volumes/Karnik/data')
    parser.add_argument('--out_dir', help='The directory where the data is saved.', dest='out_dir', default='/Volumes/Karnik')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_params()
    target_size = (512, 512)
    b_size = 4
    e = 10

    train_data = ImageDataGenerator(rescale=1./255)
    val_data = ImageDataGenerator(rescale=1./255)
    test_data = ImageDataGenerator(rescale=1./255)

    train_gen = train_data.flow_from_directory(os.path.join(args.dir, 'train'), target_size=target_size, batch_size=b_size, class_mode='binary')
    val_gen = val_data.flow_from_directory(os.path.join(args.dir, 'val'), target_size=target_size, batch_size=b_size, class_mode='binary')
    test_gen = test_data.flow_from_directory(os.path.join(args.dir, 'test'), target_size=target_size, batch_size=b_size, class_mode='binary', shuffle=False)

    num_train = train_gen.n // b_size
    num_val = val_gen.n // b_size
    num_test = test_gen.n // b_size

    model, preprocess_model, transmission_model = u_net()

    print("Fitting the model...")
    model.fit_generator(generator=train_gen, steps_per_epoch=num_train, epochs=e, validation_data=val_gen, validation_steps=num_val, workers=8)

    print("Predicting from the model...")
    test_res = preprocess_model.predict_generator(test_gen, num_test, verbose=1)
    test_t = transmission_model.predict_generator(test_gen, num_test, verbose=1)

    save_res_path = os.path.join(args.out_dir, 'test_res.npy')
    save_t_path = os.path.join(args.out_dir, 'test_t.npy')
    np.save(save_res_path, test_res)
    np.save(save_t_path, test_t)

    print("Converting to image")
    for i, filename in enumerate(test_gen.filenames):
        frame = test_res[i]
        frame = array_to_img(frame)
        frame.save(os.path.join(args.out_dir, 'imgs', '{0}.jpg'.format(os.path.basename(filename))))

        frame = test_t[i]
        frame = array_to_img(frame)
        frame.save(os.path.join(args.out_dir, 't', '{0}.jpg'.format(os.path.basename(filename))))

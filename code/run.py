import numpy as np
import argparse, os

from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, Lambda, BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img

from deep_preprocess.classifier import Classifier
from U_NET import DeepPreprocess


def parse_params():
    """Parse command line arguments and return them."""
    parser = argparse.ArgumentParser(prog='Deep Preprocess', 
                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    """ Data parameters' definition"""
    parser.add_argument('--dir', 
                        help='The directory where the data is present.', 
                        dest='dir', default='../data')
    parser.add_argument('--out_dir', 
                        help='The directory where the data is saved.', 
                        dest='out_dir', default='../result')

    args = parser.parse_args()
    return args


def process_transmission_map():
    args = parse_params()
    target_size = (512, 512)
    b_size = 4
    # e = 5
    e = 1

    train_data = ImageDataGenerator(rescale=1./255)
    val_data   = ImageDataGenerator(rescale=1./255)
    test_data  = ImageDataGenerator(rescale=1./255)

    train_gen = train_data.flow_from_directory(
        os.path.join(args.dir, 'train'), 
        target_size=target_size, batch_size=b_size, class_mode='binary')

    val_gen   = val_data.flow_from_directory(
        os.path.join(args.dir, 'val'), 
        target_size=target_size, batch_size=b_size, class_mode='binary')
    
    test_gen  = test_data.flow_from_directory(
        os.path.join(args.dir, 'test'), 
        target_size=target_size, batch_size=b_size, class_mode='binary', 
        shuffle=False)

    num_train = train_gen.n // b_size
    num_val = val_gen.n // b_size
    num_test = test_gen.n // b_size

    unet = DeepPreprocess()
    model = unet.prepare_to_init()

    print("Fitting the model...")
    model.fit_generator(generator=train_gen, steps_per_epoch=num_train, 
                        epochs=e, validation_data=val_gen, 
                        validation_steps=num_val, workers=8)

    # print("Finetuning...")
    # model = unet.prepare_to_finetune()
    # model.fit_generator(generator=train_gen, steps_per_epoch=num_train, 
                        # epochs=e, validation_data=val_gen, 
                        # validation_steps=num_val, workers=8)

    print("Predicting from the model...")
    preprocess_models = unet.preprocess_models
    train_res = preprocess_models.predict_generator(train_gen, num_train, 
                                                    verbose=1)
    print(len(train.res))
    test_res = preprocess_models.predict_generator(test_gen, num_test, 
                                                   verbose=1)

    print("Converting to image")
    for i, filename in enumerate(train_gen.filenames):
        frame = train_res[0][i]
        frame = array_to_img(frame)
        frame.save(os.path.join(args.out_dir, 'train_imgs', 
                                '{0}'.format(os.path.basename(filename))))

        frame = train_res[1][i]
        frame = array_to_img(frame)
        frame.save(os.path.join(args.out_dir, 'train_t', 
                                '{0}'.format(os.path.basename(filename))))

    for i, filename in enumerate(test_gen.filenames):
        frame = test_res[0][i]
        frame = array_to_img(frame)
        frame.save(os.path.join(args.out_dir, 'imgs', 
                                '{0}'.format(os.path.basename(filename))))

        frame = test_res[1][i]
        frame = array_to_img(frame)
        frame.save(os.path.join(args.out_dir, 't', 
                                '{0}'.format(os.path.basename(filename))))

def classify():
    pass

def main():
    process_transmission_map()
    classify()

if __name__ == '__main__':
    main()
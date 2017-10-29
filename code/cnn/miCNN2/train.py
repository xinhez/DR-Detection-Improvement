"""Train helper functions."""
import os
import random

import numpy as np

from utils.visualization import map_to_patch
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


def get_data_iterators(horizontal_flip=False, vertical_flip=False, width_shift_range=0.0,
                       height_shift_range=0.0, rotation_range=0, zoom_range=0.0,
                       batch_size=1, data_dir='data', target_size=(512, 512),
                       fill_mode='constant', rescale=1 / 255., load_train_data=True,
                       color_mode='rgb'):
    """Create data iterator."""
    aug_gen = ImageDataGenerator(horizontal_flip=horizontal_flip, vertical_flip=vertical_flip,
                                 width_shift_range=width_shift_range, height_shift_range=height_shift_range,
                                 rotation_range=rotation_range, zoom_range=zoom_range,
                                 fill_mode=fill_mode, rescale=rescale)
    data_gen = ImageDataGenerator(rescale=rescale)

    if load_train_data:
        X_train, y_train = load_dataset(data_dir=os.path.join(data_dir, 'train'), target_size=target_size,
                                        color_mode=color_mode)
        train_it = aug_gen.flow(X_train, y_train, batch_size=batch_size)
    else:
        train_it = aug_gen.flow_from_directory(os.path.join(data_dir, 'train'),
                                               batch_size=batch_size, target_size=target_size,
                                               class_mode='binary', color_mode=color_mode)

    if load_train_data:
        X_val, y_val = load_dataset(data_dir=os.path.join(data_dir, 'val'), target_size=target_size,
                                    color_mode=color_mode)
        val_it = data_gen.flow(X_val, y_val, batch_size=batch_size)
    else:
        val_it = data_gen.flow_from_directory(os.path.join(data_dir, 'val'),
                                              batch_size=batch_size, target_size=target_size,
                                              class_mode='binary', color_mode=color_mode)
    test_it = data_gen.flow_from_directory(os.path.join(data_dir, 'test'),
                                           batch_size=batch_size, target_size=target_size,
                                           class_mode='binary', color_mode=color_mode)

    return train_it, val_it, test_it


def get_lesion_iterators(horizontal_flip=False, vertical_flip=False, width_shift_range=0.,
                         height_shift_range=0., rotation_range=0, zoom_range=0.,
                         batch_size=1, base_dir='eoptha', img_dir='combined_cut',
                         annot_dir='Annotation_combined_cut', target_size=(512, 512),
                         fill_mode='constant', rescale=1 / 255., seed=None,
                         out_size=16, f_size=63):
    """Create data iterator."""
    aug_gen = ImageDataGenerator(horizontal_flip=horizontal_flip, vertical_flip=vertical_flip,
                                 width_shift_range=width_shift_range,
                                 height_shift_range=height_shift_range,
                                 rotation_range=rotation_range, zoom_range=zoom_range,
                                 fill_mode=fill_mode, rescale=rescale)
    data_gen = ImageDataGenerator(rescale=rescale)

    if seed is None:
        seed = np.random.randint(1, 1e6)

    def pair_iterator(img_it, annot_it):
        while True:
            x, _ = next(img_it)
            xa, _ = next(annot_it)

            y = np.zeros((xa.shape[0], 1, out_size, out_size), dtype=np.float32)
            for i, xai in enumerate(xa):
                y[i, 0] = annotation_to_lesion(xai[0], out_size=out_size, f_size=f_size)

            assert np.array_equal(np.unique(y), [0, 1]) or np.array_equal(np.unique(y), [0]), 'Something odd on the y array ({0})'.format(np.unique(y))

            yield x, y

    def create_iterator(gen, phase):
        img_it = gen.flow_from_directory(os.path.join(base_dir, img_dir, phase),
                                         batch_size=batch_size, target_size=target_size,
                                         color_mode='rgb', seed=seed)
        annot_it = gen.flow_from_directory(os.path.join(base_dir, annot_dir, phase),
                                           batch_size=batch_size, target_size=target_size,
                                           color_mode='grayscale', seed=seed)
        return pair_iterator(img_it, annot_it)

    train_it = create_iterator(aug_gen, 'train')
    val_it = create_iterator(data_gen, 'val')
    test_it = create_iterator(data_gen, 'test')

    return train_it, val_it, test_it


def annotation_to_lesion(annot, out_size=16, f_size=63, threshold=0, img_size=512):
    out = np.zeros((out_size, out_size), dtype=np.float32)
    for y in range(out_size):
        for x in range(out_size):
            yi, xi = map_to_patch(y, x, f_size=f_size)
            yf, xf = yi + f_size, xi + f_size

            xmi, ymi = xi + f_size/4 - 1, yi + f_size/4 - 1
            xmf, ymf = xi + (f_size/4)*3 + 1, yi + (f_size/4)*3 + 1

            if yi < 0:
                yi = 0
            if xi < 0:
                xi = 0
            if xmi < 0:
                xmi = 0
            if ymi < 0:
                ymi = 0
            if xmf > img_size:
                xmf = img_size
            if ymf > img_size:
                ymf = img_size

            out[y, x] = np.sum(annot[ymi:ymf, xmi:xmf]) > threshold
            # out[y, x] = np.sum(annot[yi:yf, xi:xf]) > threshold

    return out


def load_dataset(data_dir='data', classes=['normal', 'pathological'], color_mode='rgb',
                 target_size=(512, 512)):
    """Load dataset into memory."""
    imgs = []
    y = []
    for i, c in enumerate(classes):
        c_imgs = os.listdir(os.path.join(data_dir, c))
        imgs.extend([os.path.join(data_dir, c, c_img) for c_img in c_imgs])
        y.extend([i]*len(c_imgs))

    N = len(imgs)
    idx = range(N)
    random.shuffle(idx)

    imgs = np.array(imgs)
    y = np.array(y, dtype=np.int8)

    imgs = imgs[idx]
    y = y[idx]

    grayscale = False
    channels = 3
    if color_mode == 'grayscale':
        grayscale = True
        channels = 1

    X = np.zeros((N, channels) + target_size, dtype=np.float32)
    for i, img_path in enumerate(imgs):
        img = load_img(img_path, grayscale=grayscale, target_size=target_size)
        X[i] = img_to_array(img)

    return X, y

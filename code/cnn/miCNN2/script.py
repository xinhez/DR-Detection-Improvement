import time
import train
import models
import argparse

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from eval.eval import get_auc_score
from keras.callbacks import ModelCheckpoint


def train_and_evaluate(batch_size=8, img_dir='mess_cb', lesion_dir='eoptha_cb',
                       out_size=32, f_size=31, horizontal_flip=True, vertical_flip=True,
                       width_shift_range=0.03, height_shift_range=0.03, rotation_range=360,
                       zoom_range=0.03, exp_name='exp0.hdf5', n_blocks=4, context_blocks=1,
                       patience=75):

    train_it, val_it, test_it = train.get_data_iterators(batch_size=batch_size, horizontal_flip=horizontal_flip,
                                                         vertical_flip=vertical_flip, width_shift_range=width_shift_range,
                                                         height_shift_range=height_shift_range, rotation_range=rotation_range,
                                                         zoom_range=zoom_range, data_dir=img_dir, target_size=(512, 512),
                                                         rescale=1/255., fill_mode='constant', load_train_data=True, color_mode='rgb')

    train_it_e, val_it_e, test_it_e = train.get_lesion_iterators(out_size=out_size, f_size=f_size, batch_size=batch_size,
                                                                 horizontal_flip=horizontal_flip, vertical_flip=vertical_flip,
                                                                 width_shift_range=width_shift_range, height_shift_range=height_shift_range,
                                                                 rotation_range=rotation_range, zoom_range=zoom_range,
                                                                 base_dir=lesion_dir)

    model, ld, _, _ = models.lesion_detector_parameterized(64, l2=.0, input_size=512, n_blocks=n_blocks,
                                                           drop_p=.0, context_blocks=context_blocks, lr=2e-4)

    checkpointer = ModelCheckpoint(filepath=exp_name, verbose=1, save_best_only=True, save_weights_only=False)
    callbacks = [checkpointer]

    epochs = 4000
    cur_patience = patience
    best_loss = 10
    for e in range(epochs):
        print 'EPOCH {0}/{1}'.format(e + 1, epochs)
        start = time.time()
        r_vals = np.random.choice(2, 191)
        for r in r_vals:
            if r == 0:
                x, y = next(train_it_e)
                ld.train_on_batch(x, y)
            else:
                x, y = next(train_it)
                model.train_on_batch(x, y)

        ld.fit_generator(train_it_e, batch_size, 1, verbose=2)
        hist = model.fit_generator(train_it, batch_size, 1, validation_data=val_it, nb_val_samples=192, verbose=2, callbacks=callbacks)

        print '{0}s'.format(int(time.time() - start))
        print ''

        val_loss = hist.history['val_loss'][0]
        if val_loss < best_loss:
            best_loss = val_loss
            cur_patience = patience
        else:
            cur_patience -= 1

        if cur_patience == 0:
            break

    print 'Evaluating model {0}:'.format(exp_name)
    model.load_weights(exp_name)

    print 'Train:', model.evaluate_generator(train_it, 768)
    print 'Validation:', model.evaluate_generator(val_it, 192)
    print 'Test:', model.evaluate_generator(test_it, 240)

    print 'Train AUC = {0}'.format(get_auc_score(train_it, 768))
    print 'Validation AUC = {0}'.format(get_auc_score(val_it, 192))
    print 'Test AUC = {0}'.format(get_auc_score(test_it, 240))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='micnn')
    parser.add_argument('--lesion_dir', nargs='?', help='Directory where the lesions data is.', dest='lesion_dir', default='eoptha_cb')
    parser.add_argument('--img_dir', nargs='?', help='Directory where the image data is.', dest='img_dir', default='mess_cb')
    args = parser.parse_args()

    out_sizes = {3: 64, 4: 32, 5: 16}
    f_sizes = {3: 15, 4: 31, 5: 63}

    experiments = [[3, 0], [3, 1]]
    for n_blocks in range(3, 6):
        max_context = 5 - n_blocks
        for context_blocks in range(max_context + 1):
            experiments.append([n_blocks, context_blocks])

    for i, exp in enumerate(experiments):
        n_blocks, context_blocks = exp

        train_and_evaluate(out_size=out_sizes[n_blocks], f_size=f_sizes[n_blocks],
                           exp_name='exp{0}.hdf5'.format(i), n_blocks=n_blocks,
                           context_blocks=context_blocks, lesion_dir=args.lesion_dir,
                           img_dir=args.img_dir)

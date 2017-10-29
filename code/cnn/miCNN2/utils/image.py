import numpy as np

from keras.preprocessing.image import transform_matrix_offset_center, apply_transform, flip_axis


def image_transform(x, theta=0, tx=0, ty=0, zx=1, zy=1, horizontal_flip=False, vertical_flip=False):
        """Apply a transformation to an image."""
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = 1
        img_col_axis = 2
        img_channel_axis = 0

        theta = np.pi / 180 * theta

        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, ty],
                                     [0, 1, tx],
                                     [0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            h, w = x.shape[img_row_axis], x.shape[img_col_axis]
            transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
            x = apply_transform(x, transform_matrix, img_channel_axis,
                                fill_mode='constant', cval=0)

        if horizontal_flip:
            x = flip_axis(x, img_col_axis)

        if vertical_flip:
            x = flip_axis(x, img_row_axis)

        return x

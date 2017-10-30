from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.core import Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape
from utils.exceptions import InvalidApplicationException, InvalidArchitectureException
from keras.applications.vgg16 import VGG16, preprocess_input as preprocess_input_vgg16
from keras.applications.vgg19 import VGG19, preprocess_input as preprocess_input_vgg19
from keras.applications.inception_v3 import InceptionV3, preprocess_input as preprocess_input_v3
from keras.applications.resnet50 import ResNet50, preprocess_input as preprocess_input_resnet

from keras import backend as K


def Convolution(f, k=3, s=2, border_mode='same', **kwargs):
    """Convenience method for Convolutions."""
    return Convolution2D(f, k, k, border_mode=border_mode, subsample=(s, s), **kwargs)


def BatchNorm(mode=2, axis=1, **kwargs):
    """Convenience method for BatchNormalization layers."""
    return BatchNormalization(mode=mode, axis=axis, **kwargs)


def ConvBlock(i, nf, k=3, s=1, border_mode='same', drop_p=.0, norm=True, **kwargs):
    """A Conv-Pool-LeakyRelu-Batchnorm-Dropout block."""
    x = Convolution(nf, k=k, s=s, border_mode=border_mode, **kwargs)(i)
    x = LeakyReLU(0.02)(x)

    if norm:
        x = BatchNorm()(x)
    if drop_p > .0:
        x = Dropout(drop_p)(x)

    return x


def UpConvBlock(i, nf, k=3, norm=True, **kwargs):
    x = UpSampling2D(size=(2, 2))(i)

    x = Convolution(nf, k=k, s=1, border_mode='same', **kwargs)(x)
    x = LeakyReLU(0.2)(x)

    if norm:
        x = BatchNorm()(x)

    return x


class PreDeep:

    def __init__(self, input_size=(512, 512), nf=64):

        if K.backend() == 'theano':
            input_shape = input_size + (3,)
        elif K.backend() == 'tensorflow':
            input_shape = (3,) + input_size
        else:
            raise ValueError('{0} backend is invalid or is not supported.'.format(K.backend()))

        i = Input(input_shape)

        x1 = ConvBlock(i, nf, s=2)
        x2 = ConvBlock(x1, nf*2, s=2)
        x3 = ConvBlock(x2, nf*4, s=2)

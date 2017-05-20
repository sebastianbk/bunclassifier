import os
import json
from PIL import Image
from imp import reload

from glob import glob
import numpy as np
from numpy.random import random, permutation
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom

import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers import Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop
from keras.preprocessing import image

import vgg16; reload(vgg16)
from vgg16 import VGG16

DATA_DIR = 'data/'
IMG_WIDTH, IMG_HEIGHT = 224, 224
SIZE = IMG_WIDTH, IMG_HEIGHT

# Resize images.
for root, dirs, files in os.walk(DATA_DIR):
    for filename in files:
        filepath = os.path.join(root, filename)
        if filepath.endswith(".JPG") or filepath.endswith(".JPEG"):
            try:
                img = Image.open(filepath)
                img.thumbnail(SIZE, Image.ANTIALIAS)
                img.save(filepath, "JPEG")
                print('Resized: %s' % filepath)
            except IOError:
                print('Could not resize: %s' % filepath)

batch_size=8
vgg = VGG16()
batches = vgg16.get_batches(vgg, os.path.join(DATA_DIR, 'train'), batch_size=batch_size)
val_batches = vgg16.get_batches(vgg, os.path.join(DATA_DIR, 'valid'), batch_size=batch_size)
finetuned_model = vgg16.finetune(vgg, batches)
finetuned_model.fit_generator(batches, steps_per_epoch=64, epochs=1, validation_data=val_batches, validation_steps=16)
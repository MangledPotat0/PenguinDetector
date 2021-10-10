from keras.models import Sequential
from keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
import keras
from PIL import Image
import numpy as np
import pickle

def penguindetectormaker():

    penguindetector = Sequential (name = 'penguin')
    
    penguindetector.add(layers.Resizing(
                            height = 500,
                            width = 500,
                            interpolation = 'bilinear'))

    penguindetector.add(layers.Conv2D(6, (25, 25), activation = 'relu',
                               input_shape = (28, 28, 1),
                               name = 'conv1'))
    penguindetector.add(layers.MaxPooling(pool_size(3,3)))

    penguindetector.add(layers.Conv2D(6, (5, 5), activation = 'relu',
                               input_shape = (28, 28, 1),
                               name = 'conv2'))
    penguindetector.add(layers.MaxPooling(pool_size(3,3)))

    penguindetector.add(layers.Flatten())

    penguindetector.add(layers.Dense(200, activation = 'tanh',
                                     name = 'dense1'))

    penguindetector.add(layers.Dense(130, activation = 'tanh',
                                     name = 'dense2'))

    penguindetector.add(layers.Dense(40, activation = 'tanh',
                                     name = 'dense3'))

    penguindetector.add(layers.Dense(2, activation = 'tanh',
                                     name = 'final'))

    penguindetector.compile(loss = 'binary_crossentropy', optimizer = 'Adam')

    Return(penguindetector)


if __name__ == '__main__':

    pd = penguindetectormaker()
    im = Image.fromarray(image)
    im = im.resize(newSize, Image.LANCZOS)
    imgnp = np.asarray(im)

    training = image_dataset_from_directory(
                        'imgs/train',
                        image_size = )

    validation = image_dataset_from_directory(
                        'imgs/test',


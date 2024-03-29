import os
from keras.models import Sequential
from keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
import keras
from PIL import Image
import numpy as np
import pickle

root_logdir = os.path.join(os.curdir, 'penguin_logs')


def get_run_logdir():
    import time
    run_id = time.strftime('run_%Y%m%d-%H%M%S')
    return os.path.join(root_logdir, run_id)


def penguindetectormaker():

    penguindetector = Sequential(name = 'penguin')

    penguindetector.add(layers.Conv2D(8, (25, 25), activation = 'relu',
                               input_shape = (None, None, 3),
                               name = 'conv1'))
    penguindetector.add(layers.MaxPooling2D(pool_size = (2,2)))

    penguindetector.add(layers.Conv2D(16, (15, 15), activation = 'relu',
                               input_shape = (None, None, 3),
                               name = 'conv2'))
    penguindetector.add(layers.MaxPooling2D(pool_size = (2,2)))

    penguindetector.add(layers.Conv2D(32, (10, 10), activation = 'relu',
                               input_shape = (None, None, 3),
                               name = 'conv3'))

    penguindetector.add(layers.Dropout(0.2))

    #penguindetector.add(layers.GlobalMaxPooling2D())

    penguindetector.add(layers.Flatten())


    penguindetector.add(layers.Dense(144, activation = 'tanh',
                                     name = 'dense1'))

    penguindetector.add(layers.Dense(72, activation = 'tanh',
                                     name = 'dense2'))

    penguindetector.add(layers.Dense(1, activation = 'sigmoid',
                                     name = 'final'))

    penguindetector.build((1,750,750,3))
    penguindetector.compile(loss = 'binary_crossentropy', optimizer = 'Adam')

    return(penguindetector)


if __name__ == '__main__':

    pd = penguindetectormaker()

    training = image_dataset_from_directory(
                        'imgs/train',
                        batch_size = 8,
                        image_size = (750, 750),
                        color_mode = 'rgb',
                        label_mode = 'binary')

    validation = image_dataset_from_directory(
                        'imgs/test',
                        batch_size = 8,
                        image_size = (750, 750),
                        color_mode = 'rgb',
                        label_mode = 'binary')

    print(pd.summary())
    run = input('run?')
    
    if run=='yes':
        run_logdir = get_run_logdir()
        tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
        hist = pd.fit(x = training,
                      epochs = 10,
                      validation_data = validation,
                      callacks = [tensorboard_cb])

        with open('history.pkl', 'wb+') as hf:
            pickle.dump(hist.history, hf)
        pd.save('penguindetector.mdl')


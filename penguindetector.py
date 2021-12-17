from keras.models import Sequential
from keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
import keras
from PIL import Image
import numpy as np
import pickle

def penguindetectormaker():

    penguindetector = Sequential(name = 'penguin')
    
    penguindetector.add(layers.Conv2D(6, (30, 30), activation = 'relu',
                               input_shape = (None, None, 3),
                               name = 'conv1'))
    penguindetector.add(layers.MaxPooling2D(pool_size = (3,3)))

    penguindetector.add(layers.Conv2D(6, (20, 20), activation = 'relu',
                               input_shape = (None, None, 3),
                               name = 'conv2'))
    penguindetector.add(layers.MaxPooling2D(pool_size = (3,3)))

    penguindetector.add(layers.Conv2D(6, (10, 10), activation = 'relu',
                               input_shape = (None, None, 3),
                               name = 'conv3'))
    penguindetector.add(layers.MaxPooling2D(pool_size = (3,3)))

    penguindetector.add(layers.GlobalMaxPooling2D())

    penguindetector.add(layers.Dense(120, activation = 'tanh',
                                     name = 'dense1'))

    penguindetector.add(layers.Dense(60, activation = 'tanh',
                                     name = 'dense2'))

    penguindetector.add(layers.Dense(1, activation = 'sigmoid',
                                     name = 'final'))

    penguindetector.compile(loss = 'binary_crossentropy', optimizer = 'Adam')

    return(penguindetector)


if __name__ == '__main__':

    pd = penguindetectormaker()

    training = image_dataset_from_directory(
                        'imgs/train',
                        batch_size = 1,
                        image_size = (750, 750),
                        color_mode = 'rgb',
                        label_mode = 'binary')

    validation = image_dataset_from_directory(
                        'imgs/test',
                        batch_size = 1,
                        image_size = (750, 750),
                        color_mode = 'rgb',
                        label_mode = 'binary')

    print(pd.summary())
    
    hist = pd.fit(x = training,
                  epochs = 4,
                  validation_data = validation)

    with open('history.pkl', 'wb+') as hf:
        pickle.dump(hist.history, hf)
    pd.save('penguindetector.mdl')


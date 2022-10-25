import keras
import numpy as np
import cv2 as cv
from PIL import Image

def find_penguins(image):
    pd = keras.models.load_model('penguindetector.mdl')
    #data = Image.open(image)
    data = cv.imread(image)
    data = cv.resize(data, (750,750), interpolation=cv.INTER_LINEAR)
    data = np.resize(data, (1, 750, 750, 3))
    print(np.shape(data))
    output = pd.predict(data)
    print(output)

src = input('Enter file name\n')
find_penguins(src)

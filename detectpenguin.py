import keras
import numpy as np
import cv2 as cv
from PIL import Image

def find_penguins(image):
    pd = keras.models.load_model('penguindetector.mdl')
    #data = Image.open(image)
    #size = data.size
    #data = np.resize(data, (1, size[0], size[1], 3))
    data = cv.imread(image).resize(1,500,500,3)
    output = pd.predict(data)
    print(output)

src = input('Enter file name\n')
find_penguins(src)

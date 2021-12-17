import keras
import numpy as np
from PIL import Image

def find_penguins(image):
    pd = keras.models.load_model('penguindetector.mdl')
    data = Image.open(image)
    size = data.size
    data = np.resize(data, (1, size[0], size[1], 3))
    output = pd.predict(data)
    print(output)

src = input('Enter file name\n')
find_penguins(src)

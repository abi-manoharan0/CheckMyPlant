import cv2
from keras.utils import img_to_array
#opencv

def image_convertor(dir):
    image=cv2.imread(dir)
    if image is not None:
        image=cv2.resize(image,(256, 256))
        return img_to_array(image)

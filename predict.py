from keras.layers import Input
from mask_rcnn import MASK_RCNN 
from PIL import Image
import os
import skimage

mask_rcnn = MASK_RCNN()
IMAGE_DIR = ('./img')
count = os.listdir(IMAGE_DIR)
while True:


    for i in range(0,len(count)):
        path = os.path.join(IMAGE_DIR, count[i])
        if os.path.isfile(path):
            file_names = next(os.walk(IMAGE_DIR))[2]
            image = skimage.io.imread(os.path.join(IMAGE_DIR, count[i]))
            # Run detection
            mask_rcnn.detect_image(image)

            # img = input('Input image filename:')
            # try:
            #     image = Image.open(img)
            # except:
            #     print('Open Error! Try again!')
            #     continue
            # else:
            #     mask_rcnn.detect_image(image)

mask_rcnn.close_session()
    
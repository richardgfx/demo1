from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.preprocessing import image as image_utils
from keras.layers import Input
from keras.applications.imagenet_utils import decode_predictions

import cv2
import numpy as np
import os
import time

# Download the model
input_tensor = Input(shape=(224, 224, 3))
model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=True)
model.summary()

# Preprocessing - Image
def processing(im):
    img = image_utils.load_img(im, target_size=(224, 224))
    img = image_utils.img_to_array(img)
    img = np.expand_dims(img, axis=0) # 1 -> 1 batch
    img = preprocess_input(img)
    return img

images_dir = 'images/'
image_filenames = []
for filename in os.listdir(images_dir):
    if 'png' in filename:
        temp = filename
        image_filenames.append(temp)
print(image_filenames)

# warm up
model.predict( processing('images/' + image_filenames[0]), verbose=None )

epoch = 10
starttime = time.time()

for i in range(epoch):
    for filename in image_filenames:

        # Forward Propagation
        pred = model.predict( processing('images/' + filename), verbose=None )
        temp = decode_predictions(pred)
        className = temp[0][0][1]
        confidence = temp[0][0][2]
        label = "Predicted = {}, Confidence = {:.3f}".format(className, confidence)

        print(filename," ",label)
#       cv2.putText(frame, label, (5, 25), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
#       cv2.imshow("Image", frame)
#       cv2.waitKey(0)

endtime = time.time()
print( str(10 * epoch) + " detections,", end ="" )
print("take " + str((endtime - starttime)) + " s")
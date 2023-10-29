import cv2
import numpy as np
import time
import os

# GoogLeNet
weightFile = "model/tensorflow_inception_graph.pb"  # model architecture and weights
classFile = "model/imagenet_comp_graph_label_strings.txt"  # # 1000 labels

# Load the 1000 class labels
classes = None
with open(classFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Load the pre-trained model
net = cv2.dnn.readNetFromTensorflow(weightFile)

# Input Params
inHeight = 224
inWidth = 224
swap_rgb = True
mean = [117, 117, 117]
scale = 1.0

images_dir = 'images/'
image_filenames = []
for filename in os.listdir(images_dir):
    if 'png' in filename:
        temp = filename
        image_filenames.append(temp)
print(image_filenames)

epoch = 10
starttime = time.time()

for i in range(epoch):
    for filename in image_filenames:
        frame = cv2.imread('images/' + filename)

        # Forward Propagation
        blob = cv2.dnn.blobFromImage(frame, scale, (inWidth, inHeight), mean, swap_rgb, crop=False)
        net.setInput(blob)
        out = net.forward()
        out = out.flatten()
        classId = np.argmax(out)
        className = classes[classId]
        confidence = out[classId]
        label = "Predicted = {}, Confidence = {:.3f}".format(className, confidence)

        print(filename," ",label)
#       cv2.putText(frame, label, (5, 25), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
#       cv2.imshow("Image", frame)
#       cv2.waitKey(0)

endtime = time.time()
print( str(10 * epoch) + " detections,", end ="" )
print("take " + str((endtime - starttime)) + " s")
import requests
import json
import os
import cv2
import time

addr = "http://127.0.0.1:8000"
#addr = "http://172.22.124.198:8000"

public_endpoint = addr + '/photo'

images_dir = 'images/'

image_filenames = []
frames = []

for filename in os.listdir(images_dir):
    if 'png' in filename:
        temp = "images/"+filename
        image_filenames.append(temp)

# Prepare headers for http request
content_type = 'image/png'
headers = {'content-type': content_type}

epoch = 10
starttime = time.time()
for i in range(epoch):
    for filename in image_filenames:
        origin_img = cv2.imread(filename)
        img = cv2.resize(origin_img, (224, 224))

        # image data -> memory cache
        _, img_encoded = cv2.imencode('.png', img)
    #   response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
        response = requests.post(public_endpoint, data=img_encoded.tobytes(), headers=headers)

        label = json.loads(response.text)
        print(filename," ",label)
    #   cv2.putText(origin_img, result['inference'], (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    #   cv2.imshow("Image", origin_img)
    #   cv2.waitKey(10)
endtime = time.time()
print( str(10 * epoch) + " detections,", end ="" )
print("take " + str((endtime - starttime)) + " s")

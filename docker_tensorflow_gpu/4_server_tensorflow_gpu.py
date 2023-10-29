from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.preprocessing import image as image_utils
from tensorflow.keras.layers import Input
from keras.applications.imagenet_utils import decode_predictions

import cv2
import numpy as np
from flask import Flask,request, Response
import socket
import jsonpickle

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

# Preprocessing - ndarray
def processing_array(array):
    img = np.expand_dims(array, axis=0) # 1 -> 1 batch
    img = preprocess_input(img)
    return img

app = Flask(__name__)

@app.route('/')
def index():
    request_ip_port = request.remote_addr+','+ str(request.environ.get('REMOTE_PORT'))
    request_host = request.host
    localhostname = socket.gethostname()
    result = 'Request:' + request_ip_port + '; HOST:' + request_host  + '; Hostname:' + localhostname + "\r\n"
    # print(result)
    return result

@app.route('/photo', methods=['POST'])
def test():
    r = request

    #nparr = np.fromstring(r.data, np.uint8)
    nparr = np.frombuffer(r.data, np.uint8)

    # memory cache -> image data
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #cv2.imshow("Image", img)
    #cv2.waitKey(0)

    # Forward Propagation
    pred = model.predict( processing_array(img), verbose=None)
    temp = decode_predictions(pred)
    className = temp[0][0][1]
    confidence = temp[0][0][2]

    # Response
    label = "Predicted = {}, Confidence = {:.3f}".format(className, confidence)
    response = {'inference': label, 'message': 'success'}
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
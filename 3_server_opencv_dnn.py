import cv2
import numpy as np
from flask import Flask,request, Response
import socket
import jsonpickle

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
#   cv2.imshow("Image", img)
#   cv2.waitKey(0)

    # Forward Propagation
    blob = cv2.dnn.blobFromImage(img, scale, (inWidth, inHeight), mean, swap_rgb, crop=False)
    net.setInput(blob)
    out = net.forward()
    out = out.flatten()
    classId = np.argmax(out)
    className = classes[classId]
    confidence = out[classId]

    # Response
    label = "Class = {}, Confidence = {:.3f}".format(className, confidence)
    response = {'inference': label,'message': 'success'}
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
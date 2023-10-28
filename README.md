# Use Case - the real-time and on-demand inference, such as image labeling, object detection and transformation, etc 

 Client 1  ------------\                                   /----- Container 1  
 Client 2  -------------  Public Endpoint (Reverse Proxy) ------- Container 2    (behind the firewalls)
 Client N  ------------/          Auth,LB,HC               \----- Container M

    | <------------------------------> | <---------------------------->|
               HTTPS                              WSS or VPN

# Goal

Test the e2e time delay, jitter and throughput for the real-time AI applications. 

# Two container images:

1)richardxgf/server-tf-gpu (Tensorflow 2.9.3, Cuda 11.2, cuDNN 8.1, OpenCV 4.2, Python 3.8, Flask 3.0, Ubuntu 20)

Dynamically download the model - GoogLeNet V3 (1000 classes). The first inference would take longer time.
The built-in web server is listening on 8000; after receiving an image, it will do the FP and return the class name and probability (The GPU is not fully utilized because only one image is processed at a time).

2)richardxgf/server-opencv-dnn (OpenCV 3.4, Python 3.8, Flask 3.0, Ubuntu 20)

Use OpenCV DNN to load the TensorFlow model - GoogLeNet V1 (1000 classes) and do the inference, no GPU support!
The built-in web server is listening on 8000; after receiving an image, it will do the FP and return the class name and probability.  

# Build and Deployment

cd docker_tensorflow_gpu
docker image build -t server-tf-gpu .
docker run --rm --gpus all -p 8000:8000 server-tf-gpu

cd docker_opencv_dnn
docker image build -t server-opencv-dnn .
docker run --rm -p 8000:8000 server-opencv-dnn

# Cloud Mode

test.py, to check the e2e time delay and jitter.
client.py, to check the performance of image classification. 
Needs to modify the public endpoint in the code after the containers are deployed in the public cloud.

# Local Mode - do the inference locally

1_singleton_opencv_dnn.py
2_singleton_tensorflow_gpu.py

# Client-Server Mode 

3_server_opencv_dnn.py
4_server_tensorflow_gpu.py

test.py, to check the time delay and jitter.
client.py, to check the performance of image classification. 
Needs to modify the endpoint in the code after the servers are deployed.



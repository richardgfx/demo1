# Use Case 

The real-time and on-demand inference, such as image labeling, object detection and transformation, etc. 

Clients -------- Public Endpoint(RP,Auth,LB,HC) ------- Containers (behind the firewalls)
                               
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


# Deployment and Test 

Run the containers:

docker run --rm --gpus all -p 8000:8000 richardxgf/server-tf-gpu

docker run --rm -p 8000:8000 richardxgf/server-opencv-dnn

Run the client:

python3 test.py, to check the e2e time delay and jitter.

python3 client.py, to check the performance of image classification. 

Needs to modify the public endpoint in the code after the containers are deployed.

# Client-Server Mode 

Run the server： python3 3_server_opencv_dnn.py or 4_server_tensorflow_gpu.py 

Run the client:

python3 test.py, to check the time delay and jitter.

python3 client.py, to check the performance of image classification. 

Needs to modify the endpoint in the code after the servers are deployed.

# Local Mode 

python3 1_singleton_opencv_dnn.py 

python3 2_singleton_tensorflow_gpu.py

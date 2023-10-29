# Use Case 

The real-time and on-demand AI inference, such as image labeling, object detection and transformation, etc. 

Clients --- Public Endpoint (Auth,RP,LB,HC) --- Containers (running in WSL2 and behind the firewall)
                               
| <------------------------------> | <---------------------------->|
  
        HTTPS                       HTTP over VPN, or WSS+RP

# Goals

Test the e2e time delay, jitter, bandwidth, throughput and process, etc.

Check whether the infrastructure could support some real-time AI applications. 

# Solutions

Two container images built to use the model - GoogLeNet for image classification (1000 classes)

(1)server-opencv-dnn (OpenCV 3.4, Python 3.8, Flask 3.0, Ubuntu 20.04)

https://hub.docker.com/repository/docker/richardxgf/server-opencv-dnn

Use OpenCV DNN to load the Tensorflow model - GoogLeNet and do the inference, no GPU support!

(2)server-tf-gpu (Tensorflow 2.9.3, Cuda 11.2, cuDNN 8.1, OpenCV 4.2, Python 3.8, Flask 3.0, Ubuntu 20.04)

https://hub.docker.com/repository/docker/richardxgf/server-tf-gpu

Dynamically download the GoogLeNet model; so, the first inference would take longer time. The GPU may not be fully utilized because only one image is processed at a time by the current implementation.

The Python/Flask web server deployed in the above two images is configured to listen on Port:8000 in IPv6, and can be considered to suppport the IPv4/IPv6 dualstack, because the Linux OS can automatically attach incoming IPv4 requests to the listening IPv6 socket by mapping A.B.C.D to ::ffff:A.B.C.D (Windows doesn't allow it). 
After receiving an image, the web server will call the FP and return the class name and probability. 

# Deployment and Test 

Run the containers in the WSL2 or Linux:

docker run --rm -p 8000:8000 richardxgf/server-opencv-dnn:1.0.0

docker run --rm --gpus all -p 8000:8000 richardxgf/server-tf-gpu:1.0.0 （needs the GPU support in the WSL2 or Linux）

The containers cannot be reached by their IPs while running in the Docket Desktop + WSL2 environment, and the WSL2 doesn't support IPv6 well, but you can still access the containers by using the WSL2 IPv4 addresses and the loopback IPv6 address ([::1]) if publishing the container's port to the WSL2.   

Run the client:

test.py, to check the e2e time delay and jitter.

client.py, to check the performance of image classification. 

Needs to modify the public endpoint in the code after the containers are deployed.

# Client-Server Mode 

Run the server in the Windows, WSL2 or Linux：

3_server_opencv_dnn.py 

4_server_tensorflow_gpu.py （needs the GPU support and Tensorflow/CUDA/cnDNN installed in the Windows, WSL2 or Linux）

The web server can be accessed by using the WSL2 IPv4 addresses and the loopback IPv6 address ([::1]). In Windows, you would have to access the server by using IPv6 addresses.

Run the client: test.py and client.py.

Needs to modify the endpoint in the code after the servers are deployed.

# Local Mode 

Run the code in the Windows, WSL2 or Linux：

1_singleton_opencv_dnn.py 

2_singleton_tensorflow_gpu.py # the host（need the GPU support and Tensorflow/CUDA/cnDNN installed in the Windows, WSL2 or Linux）

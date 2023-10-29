import requests
import json
import os
import cv2
import time

addr = "http://172.22.124.198:8000"
test_endpoint = addr + '/'

epoch = 100
starttime = time.time()
for i in range(epoch):
    response = requests.get(test_endpoint)
#   print(response.text,end="")
endtime = time.time()
print( str(epoch) + " visits ", end ="" )
print("take " + str((endtime - starttime)) + " s")


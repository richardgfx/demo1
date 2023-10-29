FROM unikrn/python-opencv

RUN mkdir -p /app/model
WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY model/* ./model
COPY 3_server_opencv_dnn.py ./main.py

CMD [ "python3", "main.py"]
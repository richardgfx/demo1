FROM tensorflow/tensorflow:2.9.3-gpu

WORKDIR /app

RUN apt update && apt install -y libopencv-dev python3-opencv

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY 4* ./main.py

CMD [ "python3", "main.py"]
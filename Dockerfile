FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime


WORKDIR /workspace/KVComm


COPY requirements.docker.txt .


RUN pip install --no-cache-dir \
    -r requirements.docker.txt


COPY . .


CMD ["bash"]
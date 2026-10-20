# 인공지능 기초 실습 환경: JupyterLab + TensorFlow(CPU) + TensorBoard
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_CPP_MIN_LOG_LEVEL=2

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 fonts-nanum \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# 교실 네트워크가 느려도 되도록 MNIST를 이미지에 미리 받아 둔다 (~11MB, ~/.keras/datasets)
RUN useradd -m -u 1000 student
USER student
RUN python -c "import keras; keras.datasets.mnist.load_data()"

WORKDIR /home/student/work
EXPOSE 8888 6006
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.token=", "--ServerApp.password="]

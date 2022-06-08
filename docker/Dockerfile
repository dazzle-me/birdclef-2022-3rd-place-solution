FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04
SHELL ["/bin/bash", "-c"]
ENV TERM=linux DEBIAN_FRONTEND=noninteractive
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update -qq -o=Dpkg::Use-Pty=0 && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-setuptools \
        vim \
	git
RUN apt-get install libsndfile1-dev -y --no-install-recommends
RUN pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
WORKDIR workspace/
COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN yes | pip3 uninstall numpy
RUN pip3 install numpy==1.18.0
ENV TORCH_HOME=/workspace/bclf/.cache/pretrained_models/
ENV NUMBA_CACHE_DIR=/workspace/bclf/./cache/numba_cache

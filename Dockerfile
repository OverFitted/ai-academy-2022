#!/bin/bash
FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update &&\
    apt-get -y install \
    python3 python3-pip python3-dev python3-setuptools python3-opencv \
    build-essential yasm nasm cmake git wget unzip &&\
    apt-get clean &&\
    apt-get autoremove &&\
    rm -rf /var/lib/apt/lists/* &&\
    rm -rf /var/cache/apt/archives/*

# Upgrade pip for cv package instalation
RUN pip install --upgrade pip==21.0.1
RUN pip install seaborn scikit-image efficientnet_pytorch lmdb pillow nltk natsort fire timm ninja git+https://github.com/OptAccount/thatoptconfig/

# RUN git clone https://github.com/SwinTransformer/Swin-Transformer-Object-Detection && cd Swin-Transformer-Object-Detection
RUN git clone https://github.com/open-mmlab/mmdetection.git mmdet-to-inst/ && pip3 install -r mmdet-to-inst/requirements/build.txt && MMCV_WITH_OPS=1 MMCV_CUDA_ARGS='-gencode=arch=compute_80, code=sm_80' pip3 install -v -e mmdet-to-inst/
RUN wget https://github.com/open-mmlab/mmcv/archive/refs/tags/v1.3.18.zip && unzip -qq v1.3.18.zip && MMCV_WITH_OPS=1 pip3 install -e mmcv-1.3.18/

RUN python -c "from efficientnet_pytorch import EfficientNet;model = EfficientNet.from_pretrained('efficientnet-b7', in_channels=1, num_classes=1)"

ENV LANG C.UTF-8

WORKDIR /home/jovyan

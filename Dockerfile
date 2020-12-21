# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

ARG BASE_IMAGE=nvcr.io/nvidia/tensorflow:20.12-tf1-py3
FROM $BASE_IMAGE

RUN pip install scipy==1.3.3
RUN pip install requests==2.22.0
RUN pip install Pillow==6.2.1
RUN pip install h5py==2.9.0
RUN pip install imageio==2.9.0
RUN pip install imageio-ffmpeg==0.4.2
RUN pip install tqdm==4.49.0
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.1/zsh-in-docker.sh)" -- \
    -t robbyrussell

 ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
#     libglib2.0-0 libxext6 libsm6 libxrender1 \
#     git mercurial subversion

# RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda2-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
#     /bin/bash ~/miniconda.sh -b -p /opt/conda && \
#     rm ~/miniconda.sh && \
#     ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
#     echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.zshrc && \
#     echo "conda activate base" >> ~/.zshrc

RUN apt-get install -y openssh-server
RUN mkdir /var/run/sshd

RUN echo 'root:root' |chpasswd

RUN sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config
RUN sed -ri 's/^#?StrictModes\s+.*/StrictModes no/' /etc/ssh/sshd_config

RUN mkdir /root/.ssh

# RUN apt-get clean && \
#     rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

EXPOSE 22 80 8080 8000




CMD    ["/usr/sbin/sshd", "-D"]

# docker build --tag jp:jp .    
# Docker images (get the image id)
# docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -it --rm \
#   --mount type=bind,source="/home/jp/Datasets/Faces-Resto-256",target=/Data \
#  -v `pwd`:/scratch d82833c70feb zsh 
FROM nvidia/cuda:8.0-runtime-ubuntu16.04

# from https://github.com/anibali/docker-pytorch/blob/master/cuda-8.0/Dockerfile

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda
RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.1-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=/home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create a Python 3.6 environment
RUN /home/user/miniconda/bin/conda install conda-build \
 && /home/user/miniconda/bin/conda create -y --name py36 python=3.6.5 \
 && /home/user/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

RUN conda install -y pytorch=0.4.0 && conda clean -ya


### bikegan from here

RUN chmod 777 /home/user

WORKDIR /home/user
RUN git   clone https://github.com/twak/bikegan.git
WORKDIR /home/user/bikegan

RUN conda install -y pillow opencv torchvision
RUN pip install watchdog
RUN chmod +x /home/user/bikegan/test_interactive.py

# download the weights into the container
RUN python /home/user/bikegan/test_interactive.py --download

# when someone starts the container, start 
CMD python /home/user/bikegan/test_interactive.py
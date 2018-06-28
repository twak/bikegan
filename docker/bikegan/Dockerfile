FROM pytorch/pytorch:0.4_cuda9_cudnn7

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        git \
        unzip

# needs conda! iwth pytorch 4!

WORKDIR /
RUN git clone https://github.com/twak/bikegan.git
RUN mkdir /bikegan/checkpoints
RUN mkdir /bikegan/input
RUN mkdir /bikegan/output
WORKDIR /bikegan/checkpoints
RUN wget http://geometry.cs.ucl.ac.uk/projects/2018/frankengan/data_s3cr3t/franken_nets.zip
RUN unzip franken_nets.zip
RUN mv franken_nets/* .

ENTRYPOINT /bikegan/test_interactive.py
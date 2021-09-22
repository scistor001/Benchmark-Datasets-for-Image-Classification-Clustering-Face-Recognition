FROM docker.io/nvidia-gpu:python3.7_base
ENV DEBIAN_FRONTEND=noninteractive
ENV TMP=/tmp/software

RUN  apt-get autoclean && apt-get clean

COPY torch $TMP/torch
RUN cd $TMP/torch \
    && pip install torch-1.5.0+cu101-cp37-cp37m-linux_x86_64.whl \
    && pip install torchvision-0.6.0+cu101-cp37-cp37m-linux_x86_64.whl \
    && pip install --timeout 10000 -r requirements.txt --no-cache-dir

# rm temporaty file
RUN rm -rf $TMP

ENV BUSS=/tmp/software
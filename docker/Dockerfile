FROM docker.io/nvidia-gpu:python3.7_base
ENV DEBIAN_FRONTEND=noninteractive
ENV TMP=/tmp/software


COPY torch $TMP/torch
RUN cd $TMP/torch \
    && pip install torch-1.9.0-cp37-cp37m-manylinux1_x86_64.whl \
    && pip install torchvision-0.10.0-cp37-cp37m-manylinux1_x86_64.whl \
    && pip install --timeout 10000 -r requirements.txt --no-cache-dir

# rm temporaty file
RUN rm -rf $TMP
# Docker部署Flask服务说明

## Flask code

```python
import imageio
from flask import Flask, jsonify, request

app = Flask(__name__)


def parse_video_stream():
    file_obj = request.files['file'].stream.read()
    file_name = request.files['file'].filename
    video_reader = imageio.get_reader(file_obj, 'ffmpeg')
    fps = int(video_reader.get_meta_data()['fps'])
    size = video_reader.get_meta_data()['size']
    return video_reader, fps, size, file_name


@app.route('/action/', methods=['post'])
def action():
    try:
        return jsonify(dict(status=0, message="success", result=0))
    except Exception as e:
        return jsonify(dict(status=0, message=e, result=-1))


@app.route('/reid/', methods=['post'])
def reid():
    try:
        return jsonify(dict(status=0, message="success", result=0))
    except Exception as e:
        return jsonify(dict(status=0, message=e, result=-1))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=30080, debug=False)

```

## Docker 部署（CUDA版本）
1. 拉nvidia的基础镜像: docker pull nvidia/cuda:10.1-devel-ubuntu18.04
2. 安装基础环境 Python3.7.5(wget https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz)

```dockerfile
FROM docker.io/nvidia/cuda:10.1-devel-ubuntu18.04
ENV DEBIAN_FRONTEND=noninteractive
ENV TMP=/tmp/software

RUN apt-get update -y \
    && apt-get install -y vim wget libsm6 libxext6 libxrender-dev liblzma-dev libbz2-dev \
    && apt-get install -y make zlib1g zlib1g-dev build-essential libbz2-dev libsqlite3-dev libssl-dev libxslt1-dev libffi-dev openssl python3-tk

# Install Python-3.7.5, 需要将Python-3.7.5.tgz拷到与dockerfile同一级别下面
COPY Python-3.7.5.tgz $TMP/
RUN cd  $TMP \
    && tar -zxvf Python-3.7.5.tgz \
    && cd Python-3.7.5 \
    && ./configure --prefix=/usr/local/python3.7.5 --enable-loadable-sqlite-extensions --enable-shared \
    && make \
    && make install \
    && ln -s /usr/local/python3.7.5/bin/python3 /usr/local/python3.7.5/bin/python3.7.5 \
    && ln -s /usr/local/python3.7.5/bin/pip3 /usr/local/python3.7.5/bin/pip3.7.5 \
    && ln -sf /usr/local/python3.7.5/bin/python3 /usr/bin/python3 \
    && ln -sf /usr/local/python3.7.5/bin/python3 /usr/bin/python \
    && ln -sf /usr/local/python3.7.5/bin/pip3 /usr/bin/pip3 \
    && ln -sf /usr/local/python3.7.5/bin/pip3 /usr/bin/pip
ENV PATH=/usr/local/python3.7.5/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/python3.7.5/lib:$LD_LIBRARY_PATH
RUN pip config set global.index-url http://mirrors.aliyun.com/pypi/simple
RUN pip config set install.trusted-host mirrors.aliyun.com
```

3. docker build -t nvidia-gpu:python3.7_base .
4. 安装项目需要的Python包

```dockerfile
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
```

5. docker build -t flask_demo:v1.0 .
6. 将项目打包到镜像里面并随着容器的启动而启动

```dockerfile
FROM docker.io/flask_demo:v1.0
ENV DEBIAN_FRONTEND=noninteractive		# 安装命令的时候不需要人机交互
ENV DEMO=/demoApi
ENV GPU_ID=0
ENV START=demo.sh

#　需要将项目文件copy到与dockerfile同一级别
COPY flaskapi-video $DEMO

RUN touch /etc/init.d/$START \
    && echo "cd $DEMO" >> /etc/init.d/$START \
    && echo "export CUDA_VISIBLE_DEVICES=$GPU_ID" >> /etc/init.d/$START \
    && echo "nohup python -u server_api.py &" >> /etc/init.d/$START \
    && echo "/bin/bash" >> /etc/init.d/$START \    # 这个是必须的，防止容器启动就结束
    && chmod 755 /etc/init.d/$START \
    && update-rc.d $START defaults 90	# 开机默认级别0-99，越大越慢

CMD /etc/init.d/$START	# 随着容器启动而启动的命令
```

7. docker build -t flask_demo:test .
8. 启动docker容器命令 docker run --rm -itd -p 30080:30080 flask_demo:test
9. 将镜像打包：docker save -o flask_demo.tar flask_demo:test
10. 加载tar文件为镜像：docker load --input flask_demo.tar

## 测试

```python
import requests

if __name__ == "__main__":
    files = {'file': ("test.mp4", open(r'/mnt/2t/home/datasets/TB/test.mp4', 'rb'))}
    response = requests.post("http://127.0.0.1:30080/action/", data=None, files=files, verify=False, stream=True)
    print(response.status_code)
    print(response.text)
```


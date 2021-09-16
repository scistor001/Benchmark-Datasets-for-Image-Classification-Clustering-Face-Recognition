from flask import Flask, jsonify, request
import os
import cv2
import numpy as np
import imageio
from action_interface import ActionClassify

gpu_id = 1
action_classify = ActionClassify(model_path='./weights/action.pt', gpu_id=gpu_id)

app = Flask(__name__)


def wirte_pic(index, img):
    cv2.imwrite(os.path.join('/mnt/2t/home/zhengbowen/jobs/web_api/pic', 'test_{}.jpeg'.format(index)), img,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def parse_video_stream():
    file_obj = request.files['file'].stream.read()
    video_reader = imageio.get_reader(file_obj, 'ffmpeg')
    fps = video_reader.get_meta_data()['fps']
    return video_reader, fps


@app.route('/action/', methods=['post'])
def action():
    try:
        video_reader, fps = parse_video_stream()
        for index, img in enumerate(video_reader):
            classify, confidence = action_classify(img)
            print("Index: {}, C: {}, C: {}".format(index, classify, confidence))
        return jsonify((dict(zip(("status", "message"), [0, 'success']))))
    except Exception as e:
        return jsonify((dict(zip(("status", "message"), [1, e]))))


@app.route('/reid/', methods=['post'])
def reid():
    try:
        video_reader, fps = parse_video_stream()
        for index, img in enumerate(video_reader):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            wirte_pic(index, img)
        return jsonify((dict(zip(("status", "message"), [0, 'success']))))
    except Exception as e:
        return jsonify((dict(zip(("status", "message"), [1, e]))))


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=False)

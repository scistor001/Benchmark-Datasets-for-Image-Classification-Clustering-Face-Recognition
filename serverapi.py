from flask import Flask, jsonify, request
import os
import cv2
import numpy as np
import imageio
app = Flask(__name__)


def wirte_pic(index, img):
    cv2.imwrite(os.path.join('/mnt/2t/home/zhengbowen/jobs/web_api/pic', 'test_{}.jpeg'.format(index)), img,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100])

@app.route('/getmp4/', methods=['post'])
def getmp4():
    try:
        fileObj = request.files['file'].stream.read()
        vid_reader = imageio.get_reader(fileObj, 'ffmpeg')
        fps = vid_reader.get_meta_data()['fps']
        print("fps:",fps)
        for index, img in enumerate(vid_reader):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            wirte_pic(index, img)
        return jsonify((dict(zip(("status", "message"), [0, 'success']))))
    except Exception as e:
        return jsonify((dict(zip(("status", "message"), [1, e]))))


if __name__ == '__main__':
    app.run(host='127.0.0.1',port=5002, debug=True)

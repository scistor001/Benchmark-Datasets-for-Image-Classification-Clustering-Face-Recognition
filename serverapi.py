from flask import Flask, jsonify,request
import os
import cv2
import numpy as np
import imageio
app = Flask(__name__)

@app.route('/getmp4/', methods=['post'])
def getmp4():
    try:
        fileObj = request.files['file'].stream.read()
        vid_reader = imageio.get_reader(fileObj, 'ffmpeg')
        for index, img in enumerate(vid_reader):
            cv2.imwrite(os.path.join('/mnt/2t/home/zhengbowen/jobs/web_api/pic', 'test_{}.jpeg'.format(index)), img,
                       [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        return jsonify((dict(zip(("status", "message"), [0, 'success']))))
    except Exception as e:
        return jsonify((dict(zip(("status", "message"), [1, e]))))


if __name__ == '__main__':
    app.run(debug=True)
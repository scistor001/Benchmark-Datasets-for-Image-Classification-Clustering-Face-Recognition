from flask import Flask, jsonify, request
import os
import cv2
import imageio
from action_interface import ActionClassify
from reid_interface import ReID
import torch

# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
gpu_id = 1
action_classify = ActionClassify(model_path='./weights/action.pt', gpu_id=gpu_id)
preson_id = ReID(yolo_model_path="./weights/yolov3.weights", yolo_model_cfg="./weights/yolov3.cfg",
                 yolo_data_path="./weights/coco.names", reid_model_path="./weights/719rank1.pth")

app = Flask(__name__)

def parse_video_stream():
    file_obj = request.files['file'].stream.read()
    file_name = request.files['file'].filename
    video_reader = imageio.get_reader(file_obj, 'ffmpeg')
    fps = video_reader.get_meta_data()['fps']
    size = video_reader.get_meta_data()['size']
    return video_reader, fps, size, file_name


@app.route('/action/', methods=['post'])
def action():
    try:
        video_reader, fps, size, file_name = parse_video_stream()
        predicts = []
        for index, img in enumerate(video_reader):
            classify, confidence = action_classify(img)
            predicts.append(classify)
        summary_result = action_classify.summary(predicts, fps)
        return jsonify(dict(status=0, message="success", result=str(summary_result)))
    except Exception as e:
        return jsonify(dict(status=0, message=e, result=-1))


@app.route('/reid/', methods=['post'])
def reid():
    try:
        # debug=False
        debug = True
        video_writer = None
        all_id = []
        video_reader, fps, size, file_name = parse_video_stream()
        if debug:
            os.makedirs('./output', exist_ok=True)
            new_video_path = "./output/{}".format(file_name)
            video_writer = cv2.VideoWriter(new_video_path, cv2.VideoWriter_fourcc('M', 'P', 'E', 'G'),
                                           fps, size)
        for index, img in enumerate(video_reader):
            with torch.no_grad():
                all_id = preson_id(img, video_writer, all_id, debug=debug)
        if debug:
            video_writer.release()
        return jsonify(dict(status=0, message="success", result=str(all_id)))
    except Exception as e:
        return jsonify(dict(status=0, message=e, result=-1))


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)

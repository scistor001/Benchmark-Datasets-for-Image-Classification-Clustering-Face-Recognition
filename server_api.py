import imageio
from flask import Flask, jsonify, request

from action_interface import ActionClassify
from reid_interface import ReID

ACTIONCLASSIFY = ActionClassify(model_path='./weights/action.pt')
PERSONREID = ReID(yolo_model_path="./weights/yolov3.weights", yolo_model_cfg="./weights/yolov3.cfg",
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
            classify, confidence = ACTIONCLASSIFY(img)
            predicts.append(classify)
        summary_result = ACTIONCLASSIFY.summary(predicts, fps)
        return jsonify(dict(status=0, message="success", result=str(summary_result)))
    except Exception as e:
        return jsonify(dict(status=0, message=e, result=-1))


@app.route('/reid/', methods=['post'])
def reid():
    try:
        all_id = []
        video_reader, fps, size, file_name = parse_video_stream()
        for index, img in enumerate(video_reader):
            PERSONREID(img, all_id)
        return jsonify(dict(status=0, message="success", result=str(all_id)))
    except Exception as e:
        return jsonify(dict(status=0, message=e, result=-1))


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)

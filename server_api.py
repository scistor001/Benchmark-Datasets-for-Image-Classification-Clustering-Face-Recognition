import imageio
from flask import Flask, jsonify, request
import os
from action_interface import ActionClassify
from reid_interface import ReID
import cv2

# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
ACTIONCLASSIFY = ActionClassify(model_path='./weights/action.pt')
PERSONREID = ReID(yolo_model_path="./weights/yolov3.weights", yolo_model_cfg="./weights/yolov3.cfg",
                  yolo_data_path="./weights/coco.names", reid_model_path="./weights/719rank1.pth")

app = Flask(__name__)


def parse_video_stream():
    file_obj = request.files['file'].stream.read()
    file_name = request.files['file'].filename
    video_reader = imageio.get_reader(file_obj, 'ffmpeg')
    fps = int(video_reader.get_meta_data()['fps'])
    size = video_reader.get_meta_data()['size']
    frame_num = int(video_reader.get_meta_data()['duration'] * fps)
    return video_reader, fps, size, file_name, frame_num


def read_lable(label_path):
    results = {}
    with open(label_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split('|')
            video_name = line[0]
            if line[1] == "[]":
                continue
            all_label = line[1][1:-1]
            labels = all_label.split(',')
            results[video_name] = labels
    return results


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
        video_reader, fps, size, file_name, frame_num = parse_video_stream()
        for index, img in enumerate(video_reader):
            if (index + 1) % 10 == 0 and index < frame_num:
                PERSONREID(img, all_id)
        return jsonify(dict(status=0, message="success", result=str(all_id)))
    except Exception as e:
        return jsonify(dict(status=0, message=e, result=-1))


@app.route('/reid/pr', methods=['post'])
def reid_pr():
    try:
        # video_path =request.json['video_path']
        # label_txt = request.json['label_txt']
        result = read_lable('./video_label.txt')
        all_videoes = os.listdir('./test_video')
        all_videoes.sort()
        pr_result = {}
        all_tp = 0
        all_fp = 0
        all_true_labels = 0
        all_pre_labels = 0
        for video in all_videoes:
            video_path = os.path.join('./test_video', video)
            video_capture = cv2.VideoCapture(video_path)
            total = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            all_id = []
            for i in range(total):
                if (i + 1) % 10 == 0 and i < total:
                    video_capture.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, video_frame = video_capture.read()
                    if video_frame is None:
                        continue
                    PERSONREID(video_frame, all_id)
            true_labels = result[video]
            pre_labels = all_id
            tp, fp = 0, 0
            for true_label in true_labels:
                for pre_label in pre_labels:
                    if int(true_label) == int(pre_label):
                        tp += 1
                        break
            pr_result[video] = all_id
            fp = len(pre_labels) - tp
            all_tp = all_tp + tp
            all_fp = all_fp + fp
            all_true_labels = all_true_labels + len(true_labels)
            all_pre_labels = all_pre_labels + len(pre_labels)
        p = all_tp / (all_tp + all_fp)
        r = all_tp / all_true_labels
        f1 = (2 * p * r) / ((p + r) if (p + r) else 1)
        pr = "P:{:.4f}, R:{:.4f}, F1:{:.4f}".format(p, r, f1)
        pr_result['PR'] = pr
        return jsonify(dict(status=0, message="success", result=str(pr_result)))
    except Exception as e:
        return jsonify(dict(status=0, message=e, result=-1))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=30080, debug=False)

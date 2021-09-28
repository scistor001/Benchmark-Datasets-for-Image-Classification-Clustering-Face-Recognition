# -*- coding: utf-8 -*- #
# --------------------------------------------------------
# @Author: Lintao
# @Date: 9/16/21 3:19 PM
# @Version	1.0
# --------------------------------------------------------
import cv2
import torch
from torch.nn.functional import softmax


class ActionClassify:
    classification = {
        0: 'seat',
        1: 'walk',
        2: 'others'
    }

    def __init__(self, model_path=None, gpu_id=0):
        self.gpu_id = gpu_id
        self.end_second = - 1
        if model_path is not None:
            self.model = torch.jit.load(model_path)

    def data_process(self, img):
        img = cv2.resize(img, (256, 256)) / 255.
        img = torch.from_numpy(img)
        img = torch.reshape(img, (1, *img.shape))
        img = img.permute(0, 3, 1, 2).contiguous()
        img = img.cuda(self.gpu_id).float()
        return img

    def __call__(self, img):
        img = self.data_process(img)
        with torch.no_grad():
            pred = self.model(img)
        pred = softmax(pred).cpu().numpy()
        classify = pred.argmax()
        confidence = pred.max()

        return classify, confidence

    def summary(self, predicts, fps):
        second_blocks = [predicts[i:i + fps] for i in range(0, len(predicts), fps)]
        second_blocks = [max(block, key=block.count) for i, block in enumerate(second_blocks)]
        # smoothness
        self.end_second = len(second_blocks) - 1
        blocks = []
        for i in range(len(second_blocks)):
            curr_label = second_blocks[i]
            pre_label = second_blocks[i] if i != 0 else curr_label
            suf_label = second_blocks[i] if i != (len(second_blocks) - 1) else curr_label
            if pre_label == suf_label:
                curr_label = suf_label
            blocks.append(curr_label)
        # merge
        second_result = {}
        for i, label in enumerate(blocks):
            tmp = second_result.get(label, [])
            tmp.append(i)
            second_result[label] = tmp
        # refine result
        return self.refine_result(second_result)

    def refine_result(self, second_result, interval=3):
        result = {}
        exist_sec = []
        print(second_result)
        for label, seconds in second_result.items():
            start = -1
            end = -1
            for i in range(len(seconds)):
                if seconds[i] in exist_sec and start == -1:
                    continue
                if start == -1:
                    start = seconds[i]
                    continue
                if seconds[i] - seconds[i - 1] > interval + 1:
                    end = seconds[i - 1]

                if i == len(seconds) - 1 and end == -1:
                    end = seconds[-1]

                if end != -1:
                    if interval <= end - start or start == 0:
                        key = '{}_{}'.format(start, end)
                        result[key] = label
                        exist_sec.extend(list(range(start, end + 1)))
                    start = seconds[i]
                    end = -1

            if (seconds[-1] - start >= 1 and seconds[-1] == self.end_second) or start == 0:
                key = '{}_{}'.format(start, seconds[-1])
                exist_sec.extend(list(range(start, seconds[-1] + 1)))
                result[key] = label
        return result


if __name__ == '__main__':
    data = {
        0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
            30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 60,
            61, 62, 63, 64, 65, 66, 67, 78, 79, 80, 81, 84, 85, 86, 87, 88, 89, 90, 91],
        1: [16, 43, 57, 58, 59, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 82, 83]}

    var = ActionClassify()
    var.end_second = 91
    print(var.refine_result(data))

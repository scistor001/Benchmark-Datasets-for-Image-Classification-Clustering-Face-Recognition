# -*- coding: utf-8 -*- #
# --------------------------------------------------------
# @Author: Lintao
# @Date: 9/16/21 3:19 PM
# @Version	1.0
# --------------------------------------------------------
import torch
import cv2
from torch.nn.functional import softmax
import math


class ActionClassify:
    classification = {
        0: 'seat',
        1: 'walk',
        2: 'others'
    }

    def __init__(self, model_path, gpu_id=0):
        self.gpu_id = gpu_id
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

    def refine_result(self, second_result, interval=2):
        result = {}
        for label, seconds in second_result.items():
            start = -1
            end = -1
            for i in range(len(seconds)):
                if start == -1:
                    start = seconds[i]
                    continue
                if seconds[i] - seconds[i - 1] > interval:
                    end = seconds[i - 1]

                if end != -1:
                    if end - start > interval:
                        key = '{}_{}'.format(start, end)
                        result[key] = label
                    start = seconds[i]
                    end = -1

            if seconds[-1] - start > interval or start == 0:
                key = '{}_{}'.format(start, seconds[-1])
                result[key] = label
        return result

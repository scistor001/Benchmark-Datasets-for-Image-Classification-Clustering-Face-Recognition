# -*- coding: utf-8 -*- #
# --------------------------------------------------------
# @Author: Lintao
# @Date: 9/16/21 3:19 PM
# @Version	1.0
# --------------------------------------------------------
import torch
import cv2
from torch.nn.functional import softmax


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

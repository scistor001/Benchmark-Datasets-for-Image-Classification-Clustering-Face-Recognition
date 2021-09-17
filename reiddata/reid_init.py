import torch
import torchvision.transforms as T
from reiddata.market1501 import Market1501
from reiddata.dataset_loader import ImageDataset
from torch.utils.data import DataLoader
from reiddata.baseline import Baseline
import numpy as np
import cv2

def val_collate_fn(batch):
    imgs, pids, camids, _ = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids

def parse_data_cfg(path):
    options = dict()
    with open(path, 'r') as fp:
        lines = fp.readlines() # 返回列表，包含所有的行。

    for line in lines: # 'classes= 1\n'
        line = line.strip()
        if line == '' or line.startswith('#'): # 去掉空白行和以#开头的注释行
            continue
        key, val = line.split('=') # 按等号分割 key:'classes'  value:' 1'
        options[key.strip()] = val.strip()

    return options

def select_device(force_cpu=False, apex=False):
    cuda = False if force_cpu else torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')

    if not cuda:
        print('Using CPU')
    if cuda:
        torch.backends.cudnn.benchmark = True  # set False for reproducible results
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        cuda_str = 'Using CUDA ' + ('Apex ' if apex else '')
        for i in range(0, ng):
            if i == 1:
                # torch.cuda.set_device(0)  # OPTIONAL: Set GPU ID
                cuda_str = ' ' * len(cuda_str)
            print("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                  (cuda_str, i, x[i].name, x[i].total_memory / c))
    return device

def build_transforms(cfg):
    normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        normalize_transform
    ])
    return transform

def make_data_loader(cfg):
    val_transforms = build_transforms(cfg)
    num_workers = cfg.DATALOADER.NUM_WORKERS  # 加载图像进程数 8
    dataset = Market1501(root=cfg.DATASETS.ROOT_DIR)

    val_set = ImageDataset(dataset.query, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return val_loader, dataset.query


def make_query(query_loader, device, reidModel, query_feats):
    for i, batch in enumerate(query_loader):
        with torch.no_grad():
            img, pid, camid = batch
            img = img.to(device)
            feat = reidModel(img)  # 一共2张待查询图片，每张图片特征向量2048 torch.Size([2, 2048])
            query_feats.append(feat)
    query_feats = torch.cat(query_feats, dim=0)  # torch.Size([2, 2048])
    query_feats = torch.nn.functional.normalize(query_feats, dim=1, p=2)  # 计算出查询图片的特征向量
    return query_feats

def build_model(cfg,reidModel_path,device):
    model = Baseline()
    model.load_param(reidModel_path)
    model.to(device).eval()
    return model

def load_classes(path):
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

def select_device(force_cpu=False, apex=False):
    # apex if mixed precision training https://github.com/NVIDIA/apex
    cuda = False if force_cpu else torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')

    if not cuda:
        print('Using CPU')
    if cuda:
        torch.backends.cudnn.benchmark = True  # set False for reproducible results
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        cuda_str = 'Using CUDA ' + ('Apex ' if apex else '')
        for i in range(0, ng):
            if i == 1:
                # torch.cuda.set_device(0)  # OPTIONAL: Set GPU ID
                cuda_str = ' ' * len(cuda_str)
            print("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                  (cuda_str, i, x[i].name, x[i].total_memory / c))

    print('')  # skip a line
    return device

def letterbox(img, new_shape=416, color=(128, 128, 128)):
    shape = img.shape[:2]  # current shape [height, width] (1080, 810)
    if isinstance(new_shape, int):
        ratio = float(new_shape) / max(shape)  # 416.0 / 1080 = 0.3851851851851852
    else:
        ratio = max(new_shape) / max(shape)  # ratio  = new / old
    ratiow, ratioh = ratio, ratio
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))  # WH:(312, 416)

    dw = np.mod(new_shape - new_unpad[0], 32) / 2  # width padding  4.0
    dh = np.mod(new_shape - new_unpad[1], 32) / 2  # height padding 0.0

    if shape[::-1] != new_unpad:  # new_unpad: (312, 416)
        img = cv2.resize(img, new_unpad,
                         interpolation=cv2.INTER_AREA)  # INTER_AREA is better, INTER_LINEAR is faster
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))  # 0, 0
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))  # 4, 4

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratiow, ratioh, dw, dh
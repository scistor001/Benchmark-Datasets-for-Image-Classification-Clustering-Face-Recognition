from reiddata.models import *
import reiddata.defaults as reidCfg
from reiddata.reid_init import *
from reiddata.utils import *

class ReID:
    def __init__(self, yolo_model_path, yolo_model_cfg, yolo_data_path, reid_model_path):
        self.device = select_device(force_cpu=False)
        self.query_loader, self.data_query = make_data_loader(reidCfg._C)
        self.query_feats = []
        self.reidModel = build_model(reidCfg._C, reid_model_path, self.device)
        self.model = Darknet(yolo_model_cfg, 416)
        load_darknet_weights(self.model, yolo_model_path)
        self.model.to(self.device).eval()
        # self.query_feats = make_query(self.query_loader, self.device, self.reidModel, self.query_feats)
        self.classes = load_classes(yolo_data_path)
        # np.save('./query_0927.npy', self.query_feats.cpu())
        self.query_feats = torch.from_numpy(np.load('./query_0927.npy')).to(self.device)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.classes))]

    def data_process(self, im0):
        im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
        img, *_ = letterbox(im0, new_shape=416)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)
        return img, im0

    def __call__(self, im0, all_id, video_writer=None, debug=False):
        img, im0 = self.data_process(im0)
        with torch.no_grad():
            pred, _ = self.model(img)
        det = non_max_suppression(pred.float(), 0.1, 0.45)[0]
        if det is not None and len(det) > 0:
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            gallery_img = []
            gallery_loc = []
            pic_list = []
            for *xyxy, conf, cls_conf, cls in det:
                if self.classes[int(cls)] == 'person':
                    xmin, ymin, xmax, ymax = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    # w = xmax - xmin
                    # h = ymax - ymin
                    # # if w * h > 500:
                    gallery_loc.append((xmin, ymin, xmax, ymax))
                    crop_img = im0[ymin:ymax, xmin:xmax]
                    crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
                    crop_img = build_transforms(reidCfg._C)(crop_img).unsqueeze(0)
                    gallery_img.append(crop_img)
            if gallery_img:
                gallery_img = torch.cat(gallery_img, dim=0)
                gallery_img = gallery_img.to(self.device)
                with torch.no_grad():
                    gallery_feats = self.reidModel(gallery_img)
                gallery_feats = torch.nn.functional.normalize(gallery_feats, dim=1, p=2)
                m, n = self.query_feats.shape[0], gallery_feats.shape[0]
                distmat = torch.pow(self.query_feats, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                          torch.pow(gallery_feats, 2).sum(dim=1, keepdim=True).expand(n, m).t()
                distmat.addmm_(1, -2, self.query_feats, gallery_feats.t())
                distmat = distmat.cpu().numpy()  # <class 'tuple'>: (3, 12)
                distmat_min_value = distmat.min(axis=0)
                for k in range(n):
                    if distmat_min_value[k] < 1.0:
                        # print(distmat_min_value)
                        # print(k)
                        index = int(np.argwhere(distmat[:, k] == distmat_min_value[k])[0])
                        # print(index)
                        pic_list.append([k, index, distmat_min_value[k]])
                        # print(self.data_query[index][0])
                        name = self.data_query[index][0].split('/')[-1].split('_')[0]
                        # print(name)
                        if name not in all_id:
                            all_id.append(name)
                if debug:
                    for draw_pic in pic_list:
                        name = self.data_query[draw_pic[1]][0].split('/')[-1].split('_')[0]
                        plot_one_box(gallery_loc[draw_pic[0]], im0, label=name, color=self.colors[int(cls)])
        if debug:
            video_writer.write(im0)

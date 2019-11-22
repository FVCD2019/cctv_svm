import cv2
import numpy as np
import torch
import os

from model import Model
from utils import get_center_point_contour, perspective_transform
from utils import vehicle_mask_crop

from torch.autograd import Variable

class Detector:
    def __init__(self):
        self.input_size = (512, 512)
        self.out_size = (512//4, 512//4)
        self.mean = (0.485,0.456,0.406)
        self.var = (0.229,0.224,0.225)

        self.thresh = 0.5
        self.scale = 1.1

        checkpoint = 'ckpt_30'

        self.load_network(checkpoint)

    def load_network(self, checkpoint=None):
        print("load network...")
        self.model = Model()

        """ set CUDA """
        self.model.cuda()

        if checkpoint is not None:
            checkpoint = torch.load('/home/siit/catkin_ws/src/cctv_svm/script/Deep_detector/checkpoints/%s.pth' % checkpoint)
            self.model.load_state_dict(checkpoint['model'])
            checkpoint = None

        self.model.eval()

    def pre_processing(self, x):
        h, w = self.input_size

        x = cv2.resize(x, (w, h))

        x = x.astype(np.float32)
        x /= 255.
        x -= self.mean
        x /= self.var

        x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]

        x = x.cuda()

        return x

    def forward(self, img):
        x = self.pre_processing(img)

        out = self.model(x)

        return out

    def post_processing(self, out):
        h, w = self.input_size

        results = []

        out = out[0, :, :, 0].cpu().detach().numpy()

        offsets = get_center_point_contour(out, self.thresh, self.scale)

        out = cv2.resize(out, (w, h))

        for offset in offsets:
            box = offset['rbox']
            (cx, cy) = offset['rect'][0]

            heading = vehicle_mask_crop(out, offset['rect'])

            results.append({"center" : (cx, cy),
                             "rbox" : box,
                             "heading" : heading})

        return results

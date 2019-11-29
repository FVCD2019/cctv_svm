import cv2
import numpy as np
import torch

from model import Model
from utils import vehicle_mask_crop, get_center_point_contour
from torch.autograd import Variable


class Detector:
    def __init__(self):
        self.input_size = (384, 384)
        self.out_size = (384//2, 384//2)
        self.mean = (0.485,0.456,0.406)
        self.var = (0.229,0.224,0.225)

        self.thresh = 0.3
        self.scale = 1.0
        self.ps_thresh = 25

        checkpoint = "/home/siit/Desktop/Deep_detector/checkpoints/416_mse_70.pth"
        self.load_network(checkpoint)

        self.ps = np.int0([
            [[110, 1], [290, 310]],
            [[294, 1], [470, 310]],
            [[473, 1], [650, 310]],
            [[655, 1], [830, 310]],
            [[835, 1], [1015, 310]],
            [[111, 816], [290, 1120]],
            [[294, 816], [470, 1120]],
            [[475, 816], [655, 1120]]
        ])


    def load_network(self, checkpoint=None):
        print("load network...")
        self.model = Model()

        """ set CUDA """
        self.model.cuda()

        if checkpoint is not None:
            checkpoint = torch.load(checkpoint)
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

    def space_recognition(self, image):
        empty_space_id = []

        _image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        for idx, s in enumerate(self.ps):
            (x1, y1), (x2, y2) = s
            space_crop = _image[y1:y2, x1:x2]

            space_crop = cv2.resize(space_crop, (40, 80))

            edged = cv2.Canny(space_crop, 50, 80)

            if edged.mean() < self.ps_thresh:
                empty_space_id.append( [idx, (x1+x2)/2, (y1+y2)/2] )

        return empty_space_id if len(empty_space_id) != 0 else [[8, 0, 0]]


    def draw_parking_space(self, image, space_id):
        space_id = [idx[0] for idx in space_id ]

        for idx, s in enumerate(self.ps):
            (x1, y1), (x2, y2) = s

            color = (0, 255, 0) if idx in space_id else (255, 0, 0)
      
            image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

        return image
      

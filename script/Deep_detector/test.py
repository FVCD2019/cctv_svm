import cv2
import numpy as np
import os
import torch
import time
import glob
import argparse

from model import Model

from utils import get_center_point_contour, perspective_transform
from utils import vehicle_crop

from torch.autograd import Variable

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='512_maskv4_fp_1e4/ckpt_50',
                    help='select training dataset')
parser.add_argument('--input_size', default=512, type=int, help='Number of workers used in dataloading')
parser.add_argument('--thresh', default=0.4, type=float, help='Number of workers used in dataloading')
parser.add_argument('--scale', default=0.9, type=float, help='Number of workers used in dataloading')
parser.add_argument('--save_img', type=str2bool, default=True, help='save result images')
parser.add_argument('--flip', type=str2bool, default=False, help='save result images')

opt = parser.parse_args()


if not os.path.exists("result_imgs/"):
    os.makedirs("result_imgs/")

#data_dir = '/home/siit/Desktop/Vehicle/DB/DLDB/test/img/'
data_dir = '/home/siit/Desktop/Vehicle/DB/single_camera/'
img_paths = glob.glob(data_dir+"*.jpg")

save_img = opt.save_img
out_size = int(opt.input_size / 4)

mean = (0.485,0.456,0.406)
var = (0.229,0.224,0.225)
#mean = (0, 0, 0)
#var = (1, 1, 1)

""" Networks : Generator & Discriminator """
print("create model")
model = Model()

""" set CUDA """
model.cuda()

print("load weight")
checkpoint = torch.load('checkpoints/%s.pth' % opt.checkpoint)

model.load_state_dict(checkpoint['model'])
checkpoint = None

""" training mode  """
_ = model.eval()

sum_total_FPS = 0
sum_infer_FPS = 0
sum_post_FPS = 0

print("inference start")
for idx, img_path in enumerate(img_paths):

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = perspective_transform(img)

    ori_h, ori_w, _ = img.shape
    h = opt.input_size  # int(ori_h)
    w = opt.input_size  # int(ori_w)
    gt_size = (ori_w, ori_h)

    img = cv2.resize(img, (w, h))

    x = img.copy()

    x = x.astype(np.float32)
    x /= 255.
    x -= mean
    x /= var

    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]

    x = x.cuda()

    t0 = time.time()

    out = model(x)

    if opt.flip:
        x_flip = torch.flip(x, [3])
        out_flip = model(x_flip)

        out_flip = torch.flip(out_flip, [2])
        out = (out + out_flip) / 2

        x_flip, out_flip = None, None

    torch.cuda.synchronize()

    out = out[0, :, :, 0].cpu().detach().numpy()

    t1 = time.time()

    results = get_center_point_contour(out, opt.thresh, opt.scale)

    t2 = time.time()

    total_FPS = 1 / ((t2 - t0))
    infer_FPS = 1 / ((t1 - t0))
    post_FPS = 1 / ((t2 - t1))

    sum_total_FPS += total_FPS
    sum_infer_FPS += infer_FPS
    sum_post_FPS += post_FPS

    left = 1 / (sum_total_FPS / (idx + 1)) * (len(img_paths) - idx)

    print("[%d/%d]  %s  , FPS=%.2f (Infer=%.2f + post=%.2f) ,  left=%.2f"
          % (idx, len(img_paths), idx,
             sum_total_FPS / (idx + 1),
             sum_infer_FPS / (idx + 1),
             sum_post_FPS / (idx + 1),
             left), end='\n')

    if opt.save_img:
        out = np.clip(out * 255, 0, 255)

        out = cv2.resize(out, (w, h))

        mask = out.copy()
        mask /= 255
        #mask = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET)

        _img = img.copy()

        for result in results:
            box = result['rbox']
            (cx, cy) = result['rect'][0]

            heading = vehicle_crop(img, mask, result['rect'])

            _img = cv2.drawContours(_img, [box.astype(np.int0)], -1, (0, 255, 0), 3)  # green
            _img = cv2.putText(_img , "(%d,%d) / %d" % (cx, cy, heading), tuple(box[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 0), thickness=2)

        out = cv2.applyColorMap(out.astype(np.uint8), cv2.COLORMAP_JET)

        result_img = cv2.hconcat([_img[:, :, ::-1], out])

        cv2.imshow("output", result_img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
        cv2.destroyAllWindows()

print(
    "--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")


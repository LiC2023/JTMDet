import numpy as np

np.random.seed(0)
from utils.general import xywh2xyxy, non_max_suppression, check_file, scale_boxes
from models.experimental import attempt_load
from torchvision.utils import draw_bounding_boxes
from torchvision.io.image import read_image
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import transforms
from utils.dataloaders import create_dataloader_rgb_ir
import argparse
from tqdm import tqdm
import torch
# from utils.datasets import LoadStreams, LoadImages

parser = argparse.ArgumentParser(prog='test.py')
parser.add_argument('--weights', nargs='+', type=str, default='path to ckpt', help='model.pt path(s)')
parser.add_argument('--data', type=str, default='./data/multispectral/M3FD.yaml', help='*.data path')
parser.add_argument('--batch-size', type=int, default=1, help='size of each image batch')
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
parser.add_argument('--task', default='val', help='train, val, test, speed or study')
parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
parser.add_argument('--augment', default=False, action='store_true', help='augmented inference')
parser.add_argument('--verbose', action='store_true', help='report mAP by class')
parser.add_argument('--save-txt', default=False, action='store_true', help='save results to *.txt')
parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
parser.add_argument('--save-conf', default=False, action='store_true', help='save confidences in --save-txt labels')
parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
parser.add_argument('--project', default='runs/vis', help='save to project/name')
parser.add_argument('--name', default='vis1', help='save to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
opt = parser.parse_args()
opt.save_json = False
opt.data = check_file(opt.data)  # check file
print(opt)
print(opt.data)

img_size = 640
batch_size = 1
gs = 32
device = "cuda:1"

rgb_path = "path to rgb.txt"
ir_path = "path to ir.txt"
model = attempt_load("path to ckpt").to(device).eval()  # flir
dataset = "flir"

dataloader = create_dataloader_rgb_ir(rgb_path, ir_path, img_size, batch_size, gs, opt, pad=0.5, rect=True,)[0]

color_list = ["red", "blue", "green", "yellow", "cyan", "magenta"]

for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader)):
    nb, ch, height, width = img.shape

    img = img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    img_rgb = img[:, :3, :, :].to(device)
    img_ir = img[:, 3:, :, :].to(device)

    path1 = paths[0]
    # path2 = path1.replace("vi", "ir")  # for m3fd
    path2 = path1.replace("_RGB", "_PreviewData").replace(".jpg", ".jpeg")  # for flir
    # path2 = path1.replace("visible", "infrared")  # for llvip

    rgb_name = path1.split('/')[-1].split('.')[0] + "_rgb"
    target_name = path1.split('/')[-1].split('.')[0] + "_target"
    ir_name = path2.split('/')[-1].split('.')[0] + "_ir"

    targets[:, 2:] *= torch.Tensor([width, height, width, height])
    targets[:, 2:] = xywh2xyxy(targets[:, 2:])
    targets[:, 2:] = scale_boxes(img.shape[1:], targets[:, 2:], shapes[0][0], shapes[0][1])

    target_boxes = targets[:, 2:]
    target_class = targets[:, 1]

    img1 = read_image(path1)
    img2 = read_image(path2)

    pred = model(img_rgb, img_ir)
    pred = non_max_suppression(pred[0], conf_thres=0.5, iou_thres=0.5)[0]

    boxes = pred[:, :4]
    pred_class = pred[:, 5]
    boxes = scale_boxes(img.shape[1:], boxes, shapes[0][0], shapes[0][1])

    targets_colors = [color_list[int(i)] for i in target_class]
    pred_colors = [color_list[int(i)] for i in pred_class]

    box = draw_bounding_boxes(img1, boxes=boxes,
                              labels=None,  # labels=None,
                              colors=pred_colors,
                              width=2, font_size=60)
    im = to_pil_image(box.detach())
    im.save(f"./runs/vis/{dataset}/{rgb_name}.jpg")

    box = draw_bounding_boxes(img2, boxes=boxes,
                              labels=None,  # labels=None,
                              colors=pred_colors,
                              width=2, font_size=60)
    im = to_pil_image(box.detach())
    im.save(f"./runs/vis/{dataset}/{ir_name}.jpg")

    target = draw_bounding_boxes(img1, boxes=target_boxes,
                              labels=None,  # labels=None,
                              colors=targets_colors,
                              width=2, font_size=60)
    im = to_pil_image(target.detach())
    im.save(f"./runs/vis/{dataset}/{target_name}.jpg")

    target = draw_bounding_boxes(img2, boxes=target_boxes,
                              labels=None,  # labels=None,
                              colors=targets_colors,
                              width=2, font_size=60)
    im = to_pil_image(target.detach())
    im.save(f"./runs/vis/{dataset}/{target_name}_ir.jpg")




import argparse
import os.path

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from models.assan import ASSAN
from skimage import metrics
import data, utils

def get_crop_info(dir):
    if 'test01' in dir:
        angle = 3.5 # angle to the x-axis
        lt = 235 # left top
        lb = 435 # left bottom
    else:
        raise AssertionError('no records of the volume')
    return angle, lt, lb


def crop(img, angle, lt, lb):
    # align with matlab's 1-index
    lt = lt - 1 # left top
    lb = lb - 1 # left bottom

    rad = np.deg2rad(angle)
    depth = lb - lt
    slope = np.tan(rad)

    width = img.shape[-1]
    cropped = np.zeros((depth, width), dtype=img.dtype)
    for j in range(img.shape[-1]):
        b = round(lb - slope*j)
        t = b - depth
        cropped[:, j] = img[t+1:b+1, j]  # to include the bottom
    return cropped


def main(args):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = ASSAN(sparsity=args.sp, in_chans=4, depths=[6, 6, 6, 6], embed_dim=60, width=args.img_size)
    stride = args.stride
    
    data_path = os.path.join(args.data_path, 'raw')
    target_path = os.path.join(args.data_path, 'gt')

    model.load_state_dict(torch.load('odt_rec_' + str(args.sp) + '_ckpt_200000.pth', map_location='cpu'))
    model = model.to(device)
    model.eval()

    valid_loader = data.build_dataset(args.dataset, data_path, target_path, num_workers=4, sp=args.sp)
    
    save_root = 'odt_rec_' + str(args.sp) + '_stride'
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    psnrs = []
    ssims = []
    valid_bar = utils.ProgressBar(valid_loader)
    for sample_id, (inputs, sample_name, targets, raw_width) in enumerate(valid_bar):
        with torch.no_grad():
            inputs = inputs.cuda()
            targets = targets.cuda()
            b, c, h, w = targets.shape
            wid = args.img_size
            assert wid == 64 # currently only support the input patch width 64
            input_w = inputs.size(-1)
            padding = stride - (input_w - wid) % stride
            if padding!= 0:
                steps = (input_w - wid) // stride + 2
                inputs = F.pad(inputs, (0, padding, 0, 0), mode='constant', value=0)
            else:
                steps = (input_w - wid) // stride + 1
            weights = torch.zeros((b, c, h, raw_width + padding*args.sp)).cuda()
            out = torch.zeros((b, c, h, raw_width + padding*args.sp)).cuda()

            for i in range(steps):
                s = stride*i
                input_patch = inputs[:, :, :, s:s+wid]
                out_patch = model(input_patch)
                weights[:, :, :, s*args.sp:(s+wid)*args.sp] += 1.
                out[:, :, :, s*args.sp:(s+wid)*args.sp] = out[:, :, :, s*args.sp:(s+wid)*args.sp] + out_patch

            assert weights.min() == 1.0
            out = out / weights
            out = out[:, :, :, :-padding*args.sp]

            img = out.cpu().squeeze().numpy()
            img = np.clip(img, 0, 1)
            img = img * 65535
            img = img.astype('uint16')

            img_resize = cv2.resize(img, (1000,img.shape[0]), interpolation=cv2.INTER_AREA)
            dir_name = sample_name[0].split('_')[0]
            save_dir = os.path.join(save_root, dir_name)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            cv2.imwrite(os.path.join(save_dir, sample_name[0][:-4] + '.tif'), img_resize)

            targets = targets.cpu().squeeze().numpy() # H, W
            targets = np.clip(targets, 0, 1)
            targets = targets * 65535
            targets = targets.astype('uint16')

            # crop the background and calculate metrics
            angle, lt, lb = get_crop_info(sample_name[0])
            img_resize_crop = crop(img_resize, angle, lt, lb)
            targets_crop = crop(targets, angle, lt, lb)

            psnr = metrics.peak_signal_noise_ratio(img_resize_crop, targets_crop, data_range=65535)
            psnrs.append(psnr)
            ssim = metrics.structural_similarity(img_resize_crop, targets_crop, data_range=65535)
            ssims.append(ssim)

    print('PSNR: ', np.mean(psnrs), 'SSIM: ', np.mean(ssims))

def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Add data arguments
    parser.add_argument("--data-path", default="data", help="path to data directory")
    parser.add_argument("--dataset", default="ODT_val", help="train dataset name")
    parser.add_argument('--img_size', type=int, default=64, help='input patch size (width) of network input')
    parser.add_argument("--sp", default=2, type=int, help="sparsity factor")
    parser.add_argument("--stride", default=8, type=int, help="stride of the sliding window")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    main(args)

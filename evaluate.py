"""Evaluate super-resolution results against ground-truth high-resolution images.

Reports the metrics used in the CorPiR paper: PSNR, SSIM, LPIPS, Consistency and
(optionally) FID.

Example:
    python evaluate.py --hr_dir /path/to/test/X4/hr_128 \
                       --sr_dir experiments/Infer_X4_.../results/X4 \
                       --scale 4 --fid
"""
import argparse
import os

import cv2
import numpy as np
import torch
import lpips as lpips_lib
from PIL import Image
from torchvision.transforms import functional as trans_fn

from core.image_quality_assessment import PSNR, SSIM

IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def list_images(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXT))


def load_image(path):
    """Read an image as a float RGB tensor of shape (1, 3, H, W) in [0, 1]."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError('Cannot read image: {}'.format(path))
    img = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()
    return tensor.unsqueeze_(0)


def evaluate(hr_dir, sr_dir, scale, device):
    hr_names = list_images(hr_dir)
    sr_names = list_images(sr_dir)
    if len(hr_names) == 0:
        raise RuntimeError('No images found in {}'.format(hr_dir))
    if len(hr_names) != len(sr_names):
        raise RuntimeError('Image count mismatch: {} in {} vs {} in {}'.format(
            len(hr_names), hr_dir, len(sr_names), sr_dir))
    mismatched = [(a, b) for a, b in zip(hr_names, sr_names) if a != b]
    if mismatched:
        print('[warning] {} filename(s) differ between the two folders; '
              'pairing by sorted order (e.g. {} <-> {}).'.format(
                  len(mismatched), mismatched[0][0], mismatched[0][1]))

    psnr_model = PSNR(0, False)
    ssim_model = SSIM(0, False)
    lpips_model = lpips_lib.LPIPS(net='vgg', spatial=True).to(device)
    l1_sum = torch.nn.L1Loss(reduction='sum')

    psnr_all, ssim_all, lpips_all, cons_all = [], [], [], []

    for hr_name, sr_name in zip(hr_names, sr_names):
        hr = load_image(os.path.join(hr_dir, hr_name))
        sr = load_image(os.path.join(sr_dir, sr_name))
        if sr.shape != hr.shape:
            raise RuntimeError('Size mismatch for {}: HR is {} but SR is {}.'.format(
                hr_name, tuple(hr.shape[-2:]), tuple(sr.shape[-2:])))

        # Down-sample both images to LR size for the consistency metric.
        height, width = hr.shape[-2:]
        if height % scale or width % scale:
            raise RuntimeError('Image {} is {}x{}, which is not divisible by scale {}.'.format(
                hr_name, height, width, scale))
        lr_size = [height // scale, width // scale]
        hr_down = trans_fn.resize(hr, lr_size, Image.BICUBIC)
        sr_down = trans_fn.resize(sr, lr_size, Image.BICUBIC)

        psnr_all.append(psnr_model(sr, hr).item())
        ssim_all.append(ssim_model(sr.clamp(0, 1), hr.clamp(0, 1)).item())
        with torch.no_grad():
            lpips_all.append(lpips_model.forward(
                sr.clamp(0, 1).to(device), hr.clamp(0, 1).to(device)).mean().item())
        # Eq. (18): Consistency = sum(|LR - down(SR)|) / 1e5 * scale^2
        cons_all.append((l1_sum(sr_down * 255.0, hr_down * 255.0) * scale ** 2 / 1e5).item())

    return {
        'PSNR': (np.mean(psnr_all), np.std(psnr_all)),
        'SSIM': (np.mean(ssim_all), np.std(ssim_all)),
        'LPIPS': (np.mean(lpips_all), np.std(lpips_all)),
        'Consistency': (np.mean(cons_all), np.std(cons_all)),
    }, len(hr_names)


def main():
    parser = argparse.ArgumentParser(description='Evaluate CorPiR super-resolution results.')
    parser.add_argument('--hr_dir', type=str, required=True,
                        help='Folder with ground-truth HR images')
    parser.add_argument('--sr_dir', type=str, required=True,
                        help='Folder with reconstructed SR images')
    parser.add_argument('--scale', type=int, required=True, choices=[2, 4, 8],
                        help='Up-sampling factor')
    parser.add_argument('--fid', action='store_true',
                        help='Also compute FID (requires pytorch-fid; downloads '
                             'the InceptionV3 weights on first use)')
    parser.add_argument('--fid_batch_size', type=int, default=40)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    results, num_images = evaluate(args.hr_dir, args.sr_dir, args.scale, device)

    print('Images     : {}'.format(num_images))
    print('Upscale    : x{}'.format(args.scale))
    print('PSNR       : {:.2f} +/- {:.2f}'.format(*results['PSNR']))
    print('SSIM       : {:.4f} +/- {:.4f}'.format(*results['SSIM']))
    print('LPIPS      : {:.4f} +/- {:.4f}'.format(*results['LPIPS']))
    print('Consistency: {:.4f} +/- {:.4f}'.format(*results['Consistency']))

    if args.fid:
        from pytorch_fid import fid_score
        fid = fid_score.calculate_fid_given_paths(
            [args.hr_dir, args.sr_dir], batch_size=args.fid_batch_size,
            device=device, dims=2048, num_workers=4)
        print('FID        : {:.4f}'.format(fid))


if __name__ == '__main__':
    main()

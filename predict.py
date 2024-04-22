import argparse
import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask
from utils.astrago import Inference

def predict_img(net,
                full_img,
                device,
                imgsz=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, imgsz, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', type=str, help='Filenames of input images', required=True)
    parser.add_argument('--save', '-s', metavar='SAVE PATH', type=str, default='./predict_result/out', 
                        help='Save Directory Path to save result images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--imgsz', '-is', type=int, default=640,
                        help='Image resize for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
    return parser.parse_args()


def get_output_filenames(in_files):
    def _generate_name(fn):
        return f'{os.path.splitext(os.path.basename(fn))[0]}_OUT.png'

    return [_generate_name(fn) for fn in in_files]


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)

def make_result_folder(args):
    path = args.save
    path = path.rstrip("/")
    folder_name = os.path.basename(path)
    folder_path = os.path.dirname(path)
    new_folder_path = path
    i = 1
    while os.path.exists(new_folder_path): 
        new_folder_name = f"{folder_name}_{i}"
        new_folder_path = os.path.join(folder_path, new_folder_name)
        i += 1
    os.makedirs(new_folder_path)
    return new_folder_path


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = [os.path.join(args.input, files) for files in os.listdir(args.input)]
    out_files = get_output_filenames(in_files)
    save_path = make_result_folder(args)

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    
    
    Inference.get_model_params(net)
    Inference.get_data_num(in_files)
    Inference.get_image_size(args.imgsz)
    for i, filename in Inference(enumerate(in_files), total=len(in_files)):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)

        start_time = time.time()
        mask = predict_img(net=net,
                           full_img=img,
                           imgsz=args.imgsz,
                           out_threshold=args.mask_threshold,
                           device=device)
        Inference.get_elapsed_inference_time(start_time)

        start_time = time.time()
        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask, mask_values)
            result.save(os.path.join(save_path, out_filename))
            logging.info(f'Mask saved to {os.path.join(save_path, out_filename)}')
        Inference.get_elapsed_save_time(start_time)

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)

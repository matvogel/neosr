import logging
from os import path as osp
from time import time

import torch

from neosr.data import build_dataloader, build_dataset
from neosr.data.single_dataset import single
from neosr.models import build_model
from neosr.utils import get_root_logger, get_time_str, make_exp_dirs
from neosr.utils.options import parse_options
from PIL import Image
import numpy as np
from neosr.utils import FileClient, imfrombytes, img2tensor, scandir
import math

upscalingAmount = 4
tileSize = 192
min_overlap = 0.1

def inference(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, args = parse_options(root_path, is_train=False)

    torch.set_default_device("cuda")
    torch.backends.cudnn.benchmark = True

    # create model
    model = build_model(opt)
    
    lq_path = args.input_file
    file_client = FileClient()
    img_bytes = file_client.get(lq_path, 'lq')
    img = imfrombytes(img_bytes, float32=True)
    img = img2tensor(img, bgr2rgb=False, float32=True, color=True)
    img = np.transpose(img, (1, 2, 0))
    
    print("Image shape: ", img.shape)
    
    # create tiles of constant shape tileSize with overlaps of minimum min_overlap
    imgOutput = np.zeros((img.shape[0]*upscalingAmount, img.shape[1]*upscalingAmount, img.shape[2]), dtype=np.uint8)
    for y in range(0, img.shape[0], tileSize):
        for x in range(0, img.shape[1], tileSize):
            tile = img[y:y+tileSize, x:x+tileSize, :]
            tile = np.transpose(tile, (2, 0, 1))
            tile = tile.unsqueeze(0).float().cuda()
            print("Tile shape: ", tile.shape)
            # zero pad to shape B C tileSize, tileSize
            tiled = False
            if tile.shape[2] < tileSize or tile.shape[3] < tileSize:
                padded_x = tileSize - tile.shape[3]
                padded_y = tileSize - tile.shape[2]
                tile = torch.nn.functional.pad(tile, (0, padded_x, 0, padded_y), mode='constant', value=0)
                tiled = True
            print("Tile shape after padding: ", tile.shape)
            tile = model.upscale_img(tile)
            #tile = tile.squeeze(0).cpu().numpy()
            #tile = np.transpose(tile, (1, 2, 0))
            tile = np.clip(tile, 0, 255)
            tile = tile.astype(np.uint8)
            # remove the tiling artifacts, not that the image is upscaled
            if tiled:
                print("Removing padding", padded_y, padded_x)
                padded_x *= upscalingAmount
                padded_y *= upscalingAmount
                if padded_x > 0:
                    tile = tile[:, :-padded_x, :]
                if padded_y > 0:
                    tile = tile[:-padded_y, :, :]
            
            imgOutput[y*upscalingAmount:(y+tileSize)*upscalingAmount, x*upscalingAmount:(x+tileSize)*upscalingAmount, :] = tile
    
    # save image
    img_sr = Image.fromarray(imgOutput)
    img_sr.save("test.png")

def test_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)

    torch.set_default_device("cuda")
    torch.backends.cudnn.benchmark = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt["path"]["log"], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name="neosr", log_level=logging.INFO, log_file=log_file
    )

    # create test dataset and dataloader
    test_loaders = []
    for dataset_type, dataset_opt in sorted(opt["datasets"].items()):
        if dataset_type == "train":
            continue
        test_set = build_dataset(dataset_opt)
        num_gpu = opt.get("num_gpu", "auto")
        test_loader = build_dataloader(
            test_set,
            dataset_opt,
            num_gpu=num_gpu,
            dist=opt["dist"],
            sampler=None,
            seed=opt["manual_seed"],
        )
        logger.info(f"Number of test images in Val dataset: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)

    for test_loader in test_loaders:
        #test_set_name = test_loader.dataset.opt["name"]
        logger.info(f"Testing ...")
        start_time = time()
        model.validation(
            test_loader,
            current_iter=opt["name"],
            tb_logger=None,
            save_img=opt["val"]["save_img"],
        )
        end_time = time()
        total_time = end_time - start_time
        n_img = len(test_loader.dataset)
        fps = n_img / total_time
        logger.info(f"Inference took {total_time:.2f} seconds, at {fps:.2f} fps.")


if __name__ == "__main__":
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    inference(root_path)
    #test_pipeline(root_path)

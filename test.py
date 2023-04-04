from argparse import ArgumentParser
import importlib
from tqdm import tqdm
import torch
import glob
import json
from pathlib import Path
import os
import cv2
import math
import albumentations as A
from lightning_module import LightningModule
from utils import *




class Dataset(torch.utils.data.Dataset):

    def __init__(self, cfg):
        self.cfg = cfg
        test_img_path = os.path.join(cfg.img_path, 'test')
        self.test_images = glob.glob(test_img_path + '/*.jpg')
        self.transforms =  A.Compose([
                A.Resize(self.cfg.val_resolution, self.cfg.val_resolution),
                A.Normalize(mean=self.cfg.image_means, std=self.cfg.image_stds)
                ])

    def __len__(self):
        return len(self.test_images)

    def __getitem__(self, idx):
        img_path = self.test_images[idx]
        img = cv2.imread(self.test_images[idx])
        h, w , c = img.shape
        img =  self.transforms(image=img)['image']


        out = {
            'pixel_values': torch.from_numpy(img).permute(2, 0, 1),
            'image_height':h,
            'image_width': w,
        }
        return out


def get_dataloader(cfg):
    dataset = Dataset(
        cfg=cfg,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.val_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    return loader





if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--config_name', type=str, default='default')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--val_resolution', type=int)
    parser.add_argument('--val_batch_size', type=int)
    parser.add_argument("--tta", action='store_true')
    parser.add_argument("--refine_mask", action='store_true')
    parser.add_argument('--ckpt', type=str)
    args = parser.parse_args()

    cfg = getattr(importlib.import_module(f'config.{args.config_name}'), 'config')

    if args.refine_mask:
        cfg.refine_mask = True

    if args.val_resolution is not None:
        cfg.val_resolution = args.val_resolution
    if args.val_batch_size is not None:
        cfg.val_batch_size = args.val_batch_size
    module = LightningModule.load_from_checkpoint(
        #os.path.join(cfg.output_dir, args.config_name, 'model/epoch=139-step=24220.ckpt'), #'model/last.ckpt'),
        args.ckpt,
        map_location='cpu',
    )
    module = module.eval().to(device=args.device)
    print(module.hparams)

    dataloader = get_dataloader(cfg)

    batch_masks_prob = []
    batch_image_heights = []
    batch_image_widths = []
    for batch in tqdm(dataloader):
        for k, v in batch.items():
            batch[k] = v.to(device=args.device)
        out = module(batch)
        out_prob_map = out['batch_masks_prob']
        
        if args.tta:
            orig_pixel_values = batch['pixel_values'].clone()

            # hflip
            hflip_pixel_values = torch.flip(orig_pixel_values, dims=[2])
            batch['pixel_values'] = hflip_pixel_values
            out_hflip =  module(batch)
            out_prob_map +=  torch.flip(out_hflip['batch_masks_prob'], dims=[2])

            # vflip
            vflip_pixel_values = torch.flip(orig_pixel_values, dims=[3])
            batch['pixel_values'] = vflip_pixel_values
            out_vflip =  module(batch)
            out_prob_map +=  torch.flip(out_vflip['batch_masks_prob'], dims=[3])

            # # flip both
            # vhflip_pixel_values =  torch.flip(orig_pixel_values, dims=[2,3])
            # batch['pixel_values'] = vhflip_pixel_values
            # out_vhflip = module(batch)
            # out_prob_map +=  torch.flip(out_vhflip['batch_masks_prob'], dims=[2,3])

            # permute RGB channels
            permute_pixel_values = orig_pixel_values[:, [2, 1, 0], :, :]
            batch['pixel_values'] = permute_pixel_values
            out_permute = module(batch)
            out_prob_map += out_permute['batch_masks_prob']

            # scale 0.8
            sf = 0.8
            gs = 32
            _, _, h, w = orig_pixel_values.shape
            size = [math.ceil(x * sf / gs) * gs for x in orig_pixel_values.shape[2:]]
            scaled_pixel_values = torch.nn.functional.interpolate(orig_pixel_values, size=size, mode='bilinear')
            batch['pixel_values'] = scaled_pixel_values
            out_scaled = module(batch)
            out_prob_map += torch.nn.functional.interpolate(out_scaled['batch_masks_prob'], size=(h,w), mode='bilinear')

            sf = 1.2
            gs = 32
            size = [math.ceil(x * sf / gs) * gs for x in orig_pixel_values.shape[2:]]
            scaled_pixel_values = torch.nn.functional.interpolate(orig_pixel_values, size=size, mode='bilinear')
            batch['pixel_values'] = scaled_pixel_values
            out_scaled = module(batch)
            out_prob_map += torch.nn.functional.interpolate(out_scaled['batch_masks_prob'], size=(h,w), mode='bilinear')

            out_prob_map /= 6
                
        batch_masks_prob.append(out_prob_map.cpu())
        batch_image_heights.append(batch['image_height'].cpu())
        batch_image_widths.append(batch['image_width'].cpu())
    
    
    
    batch_masks_prob = torch.cat(batch_masks_prob)[:, :3, :, :]
    print('batch_masks_prob', batch_masks_prob.shape)
    batch_image_heights = torch.cat(batch_image_heights)
    batch_image_widths = torch.cat(batch_image_widths)
    batch_image_ids = [x.split('/')[-1] for x in dataloader.dataset.test_images] # test_images
    to_save = {
        'batch_masks_prob': batch_masks_prob,
        'batch_image_heights': batch_image_heights,
        'batch_image_widths': batch_image_widths,
        'batch_image_ids': batch_image_ids
    }
    save_path = os.path.join(cfg.output_dir, args.config_name, 'submission/test.pt')
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    torch.save(to_save, save_path)
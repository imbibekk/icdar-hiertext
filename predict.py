from argparse import ArgumentParser
import importlib
from tqdm import tqdm
import torch
import json
from pathlib import Path
import os
import gc
import math
from data import get_dataloader
from lightning_module import LightningModule
from utils import *

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--config_name', type=str, default='default')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataloader_mode', type=str, default='validation')
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

    dataloader = get_dataloader(cfg, args.dataloader_mode)

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
        
        #batch_masks_prob.append(out['batch_masks_prob'].cpu())
        
        batch_masks_prob.append(out_prob_map.cpu())
        batch_image_heights.append(batch['image_height'].cpu())
        batch_image_widths.append(batch['image_width'].cpu())
    
    
    
    batch_masks_prob = torch.cat(batch_masks_prob)
    batch_image_heights = torch.cat(batch_image_heights)
    batch_image_widths = torch.cat(batch_image_widths)
    batch_image_ids = [x['image_id'] for x in dataloader.dataset.anns]
    to_save = {
        'batch_masks_prob': batch_masks_prob,
        'batch_image_heights': batch_image_heights,
        'batch_image_widths': batch_image_widths,
        'batch_image_ids': batch_image_ids
    }
    save_path = os.path.join(cfg.output_dir, args.config_name, f'submission/{args.dataloader_mode}.pt')
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    torch.save(to_save, save_path)
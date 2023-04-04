from argparse import ArgumentParser
import importlib
from tqdm import tqdm
import torch
from pathlib import Path
import os

from data import get_dataloader
from lightning_module import LightningModule
from utils import *

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--config_name', type=str, default='default')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--val_resolution', type=int, default=2048)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--component_min_prob', type=float)
    parser.add_argument('--component_min_area', type=float)
    parser.add_argument('--pred_mask_dilate_iter', type=int)
    parser.add_argument('--pred_vertices_expand_ratio', type=float)
    args = parser.parse_args()

    cfg = getattr(importlib.import_module(f'config.{args.config_name}'), 'config')

    if args.val_resolution is not None:
        cfg.val_resolution = args.val_resolution
    if args.val_batch_size is not None:
        cfg.val_batch_size = args.val_batch_size

    kwargs = {}
    if args.component_min_prob is not None:
        kwargs['component_min_prob'] = args.component_min_prob
    if args.component_min_area is not None:
        kwargs['component_min_area'] = args.component_min_area
    if args.pred_mask_dilate_iter is not None:
        kwargs['pred_mask_dilate_iter'] = args.pred_mask_dilate_iter
    if args.pred_vertices_expand_ratio is not None:
        kwargs['pred_vertices_expand_ratio'] = args.pred_vertices_expand_ratio
        
    module = LightningModule.load_from_checkpoint(
        os.path.join(cfg.output_dir, args.config_name, 'model/last.ckpt'),
        map_location='cpu',
        **kwargs
    )
    module = module.eval().to(device=args.device)
    print(module.hparams)

    dataloader = get_dataloader(cfg, 'validation')

    batch_masks_prob = []
    batch_image_heights = []
    batch_image_widths = []
    pbar = tqdm(dataloader)
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.to(device=args.device)
        out = module.validation_step(batch, 0)
        pbar.set_postfix(module.accumulator.compute())
    
    print(args)
    print(module.accumulator.compute())
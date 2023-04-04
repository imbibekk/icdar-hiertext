from argparse import ArgumentParser
import importlib
from tqdm import tqdm
import torch
import json
from pathlib import Path
import os
import gc

from data import get_dataloader
from lightning_module import LightningModule
from utils import *

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--config_name', type=str, default='default')
    parser.add_argument('--dataloader_mode', type=str, default='validation')
    parser.add_argument('--gt_path', type=str, default='input/gt/validation.jsonl')
    parser.add_argument('--sub_path1', type=str, default='input/gt/validation.jsonl')
    parser.add_argument('--sub_path2', type=str, default='input/gt/validation.jsonl')
    parser.add_argument('--sub_path3', type=str, default='input/gt/validation.jsonl')
    parser.add_argument('--sub_path4', type=str, default='input/gt/validation.jsonl')

    parser.add_argument('--save_path', type=str, default='input/gt/validation.jsonl')
    
    parser.add_argument('--th_word', type=float)
    parser.add_argument('--th_line', type=float)
    parser.add_argument('--th_para', type=float)

    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    cfg = getattr(importlib.import_module(f'config.{args.config_name}'), 'config')

    cfg.component_min_prob = [args.th_word, args.th_line, args.th_para]

    print(cfg)
    #out = torch.load(os.path.join(cfg.output_dir, args.config_name, f'submission/{args.dataloader_mode}.pt'))
    out1 = torch.load(args.sub_path1) 
    out2 = torch.load(args.sub_path2)
    out3 = torch.load(args.sub_path3)
    out4 = torch.load(args.sub_path4)

    print(out1['batch_masks_prob'].shape, out2['batch_masks_prob'].shape, out3['batch_masks_prob'].shape, out4['batch_masks_prob'].shape) # torch.Size([1724, 6, 2048, 2048]), torch.Size([1724, 3, 2048, 2048])

    outs = torch.stack([out1['batch_masks_prob'][:, :3, :, :],  out2['batch_masks_prob'][:, :3, :, :], out3['batch_masks_prob'][:, :3, :, :], out4['batch_masks_prob'][:, :3, :, :]])
    outs = torch.mean(outs, dim=0)
    print(outs.shape)
    submission = batch_masks2submission(
        #out['batch_masks_prob'][:, :3, :, :],
        outs,
        out1['batch_image_ids'],
        out1['batch_image_heights'].tolist(),
        out1['batch_image_widths'].tolist(),
        min_prob=cfg.component_min_prob,
        component_min_area=cfg.component_min_area,
        mask_dilate_iter=cfg.pred_mask_dilate_iter,
        vertices_expand_ratio=cfg.pred_vertices_expand_ratio,
        device=args.device
    )

    save_path = os.path.join(cfg.output_dir, f'{args.config_name}/{args.save_path}/{args.dataloader_mode}.jsonl')
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    with open(save_path, 'w') as f:
        json.dump(submission, f)

    # if args.gt_path and (args.dataloader_mode != 'test'):
    #     score_path = os.path.join(cfg.output_dir, f'{args.config_name}/submission/scores.txt')
    #     os.system(f'python eval.py --gt={args.gt_path} --result={save_path} --output={score_path} --mask_stride=1 --eval_lines --eval_paragraphs --num_workers=4')
    #     os.system(f'cat {score_path}-00000-of-00001')
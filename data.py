import numpy as np
from pathlib import Path
import torch
import os
import json
import albumentations as A
from skimage.morphology import dilation, square
from scipy.ndimage import binary_erosion, binary_dilation
from skimage import measure

from skimage.segmentation import relabel_sequential, watershed

from utils import *

class Dataset(torch.utils.data.Dataset):

    def __init__(self, cfg, mode):
        self.cfg = cfg
        self.mode = mode
        with open(os.path.join(self.cfg.input_dir, 'gt', f'{mode}.jsonl'), 'r') as f:
            self.anns = json.load(f)['annotations']
        if self.cfg.debug:
            self.anns = self.anns[:self.cfg.batch_size]

    def __len__(self):
        return len(self.anns)
    
    def _get_gap_mask(self, mask):
        mask_multi = measure.label(mask, connectivity=1, background=0)
        tmp = dilation(mask_multi>0, square(11))
        tmp2 = watershed(tmp, mask_multi, mask=tmp, watershed_line=True) > 0
        tmp = tmp ^ tmp2
        tmp = dilation(tmp, square(5))
        return tmp.astype(np.uint8)
    
    def _get_contour_mask(self, mask):
        eroded = binary_erosion(mask, iterations=2)
        countour_mask = mask ^ eroded
        return countour_mask

    def _get_center_and_offset(self, mask):
        center_pts = []
        height, width = mask.shape
        center = np.zeros((1, height, width), dtype=np.float32)
        sigma = 2

        y_coord = np.ones_like(mask, dtype=np.float32)
        x_coord = np.ones_like(mask, dtype=np.float32)
        y_coord = np.cumsum(y_coord, axis=0) - 1
        x_coord = np.cumsum(x_coord, axis=1) - 1
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        offset = np.zeros((2, height, width), dtype=np.float32)

        for i in np.unique(mask):
            if i == 0:
                continue
            mask_index = np.where(mask == i)
            if len(mask_index[0]) == 0:
                # the instance is completely cropped
                continue

            # Find instance area
            ins_area = len(mask_index[0])
            center_y, center_x = np.mean(mask_index[0]), np.mean(mask_index[1])
            center_pts.append([center_y, center_x])
            # generate center heatmap
            y, x = int(center_y), int(center_x)
            # upper left
            ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
            # bottom right
            br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

            c, d = max(0, -ul[0]), min(br[0], width) - ul[0]
            a, b = max(0, -ul[1]), min(br[1], height) - ul[1]

            cc, dd = max(0, ul[0]), min(br[0], width)
            aa, bb = max(0, ul[1]), min(br[1], height)
            center[0, aa:bb, cc:dd] = np.maximum(
                center[0, aa:bb, cc:dd], g[a:b, c:d])

            # generate offset (2, h, w) -> (y-dir, x-dir)
            offset_y_index = (np.zeros_like(mask_index[0]), mask_index[0], mask_index[1])
            offset_x_index = (np.ones_like(mask_index[0]), mask_index[0], mask_index[1])
            offset[offset_y_index] = center_y - y_coord[mask_index]
            offset[offset_x_index] = center_x - x_coord[mask_index]
        return center #, offset


    def __getitem__(self, idx):
        if (self.mode == 'train') and self.cfg.aug:
            transforms = A.Compose([
                # A.Resize(cfg.trn_resolution, cfg.trn_resolution, interpolation=0),
                A.HorizontalFlip(p=0.5),
                A.Affine(),
                A.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=10),
                A.RandomResizedCrop(self.cfg.trn_resolution, self.cfg.trn_resolution, scale=(0.5, 3.0)),
                A.HueSaturationValue(),
                A.RandomGamma(),
                A.RandomBrightnessContrast(),
                A.Transpose(), 
                A.Normalize(mean=self.cfg.image_means, std=self.cfg.image_stds),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        else:
            transforms = A.Compose([
                A.Resize(self.cfg.val_resolution, self.cfg.val_resolution),
                A.Normalize(mean=self.cfg.image_means, std=self.cfg.image_stds)
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

        data = parse_annotation(self.anns[idx], image_dir=os.path.join(self.cfg.input_dir, self.mode))
        keypoints = []
        num_points = []
        for k in ['word', 'line', 'para']:
            keypoints.append(np.concatenate(data[f'{k}_polygons']))
            num_points += [len(x) for x in data[f'{k}_polygons']]
        keypoints = np.concatenate(keypoints)
        indices = np.cumsum(num_points)
        transformed = transforms(image=data['image'], keypoints=keypoints)
        polygons = np.split(transformed['keypoints'], indices)
        data['image'] = transformed['image']
        for k in ['word', 'line', 'para']:
            n_polygons = len(data[f'{k}_polygons'])
            data[f'{k}_polygons'] = polygons[:n_polygons]
            polygons = polygons[n_polygons:]

        h, w, _ = data['image'].shape
        masks = np.zeros((3, h, w), dtype=np.uint8)
        illegible_masks = masks.copy()
        omasks_list = [[], [], []]
        for i, k in enumerate(['word', 'line', 'para']):
            for legible, vertices in zip(data[f'{k}_legibles'], data[f'{k}_polygons']):
                if (self.mode == 'train') & (self.cfg.vertices_shrink_factor > 0):
                    vertices = shrink_vertices(vertices, self.cfg.vertices_shrink_factor, self.cfg.vertices_shrink_max_dist_factor * min(h, w))
                omask = vertices2omask(vertices, h, w)
                if self.mode == 'train':
                    if self.cfg.mask_erode_iter > 0:
                        for _ in range(self.cfg.mask_erode_iter):
                            eroded_omask = cv2.erode(omask, np.ones((3, 3)), iterations=1)
                            if eroded_omask.sum() == 0:
                                break
                            omask = eroded_omask
                omasks_list[i].append(omask)
                overlap = None
                if (self.mode == 'train') and (self.cfg.gaps[i] > 0):
                    overlap = cv2.dilate(masks[i], np.ones((3, 3)), iterations=self.cfg.gaps[i]) & cv2.dilate(omask, np.ones((3, 3)), iterations=self.cfg.gaps[i])
                masks[i] |= omask
                if not legible:
                    illegible_masks[i] |= omask
                if (self.mode == 'train') and (overlap is not None):
                    masks[i] &= ~overlap
                    if not legible:
                        illegible_masks[i] &= ~overlap

        omasks_list = [np.stack(omasks) for omasks in omasks_list]
        omaps = omasks_list2omaps(omasks_list)

        bounday_masks = np.zeros((3, h, w), dtype=np.uint8)
        for i in range(3):
            #bounday_masks[i] = self._get_boundary_mask(omaps[i])
            
            #if i < 3:
            bounday_masks[i] = self._get_contour_mask(masks[i])
            
            #else:
            #bounday_masks[i] = self._get_gap_mask(masks[i-3])

            #bounday_masks[i] = self._get_center_and_offset(omaps[i])
            
            #if self.mode == 'train':
            #    masks[i] = masks[i] - bounday_masks[i]

        #masks = np.vstack((masks, boundary_mask[None, ...]))

        out = {
            'pixel_values': torch.from_numpy(data['image']).permute(2, 0, 1),
            'omaps': torch.from_numpy(omaps.astype(np.int64)),
            'bounday_masks': torch.from_numpy(bounday_masks.astype(np.float32)),
            'mask_labels': torch.from_numpy(masks.astype(np.float32)),
            'illegible_masks': torch.from_numpy(illegible_masks),
            'image_height': data['image_height'],
            'image_width': data['image_width']
        }
        return out

def get_dataloader(cfg, mode):
    dataset = Dataset(
        cfg=cfg,
        mode=mode,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size if mode == 'train' else cfg.val_batch_size,
        shuffle=True if mode == 'train' else False,
        num_workers=cfg.num_workers,
    )
    return loader



def get_combined_dataloader(cfg):
    tr_dataset = Dataset(
        cfg=cfg,
        mode='train',
    )

    val_dataset =  Dataset(
        cfg=cfg,
        mode='validation',
    )

    dataset = torch.utils.data.ConcatDataset([tr_dataset, val_dataset])
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size, # if mode == 'train' else cfg.val_batch_size,
        shuffle=True, #if mode == 'train' else False,
        num_workers=cfg.num_workers,
    )
    return loader



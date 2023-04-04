import numpy as np
import torch
import cv2
import os
from PIL import Image
from tqdm import tqdm
from typing import Union
from shapely.geometry import Polygon
import pyclipper

'''
mask: [height, width] binary
masks: [n_classes, height, width] binary
omap: [height, width] integer
omaps: [n_classes, height, width] integer
omasks: [n_objects, height, width] binary
omasks_list: List[n_classes x [n_objects, height, width]] binary
'''

def hmean(list):
    return len(list) / sum([1/(x+1e-8) for x in list])

def get_iou_mat(masks1:torch.Tensor, masks2:torch.Tensor):
    '''
    masks1: [n_objects1, height, width]
    masks2: [n_objects2, height, width]
    '''
    masks1 = masks1.view(masks1.shape[0], -1).to(torch.float32)
    masks2 = masks2.view(masks2.shape[0], -1).to(torch.float32)
    xandy = masks1 @ masks2.T
    xory = masks1 @ torch.ones_like(masks2.T) + torch.ones_like(masks1) @ masks2.T - xandy
    iou_mat = xandy / xory
    return iou_mat

@torch.no_grad()
def get_metric_info(omasks_pred:torch.Tensor, omasks_true:torch.Tensor):
    '''
    omasks_pred: [n_pred_objects, height, width]
    omasks_true: [n_true_objects, height, width]
    '''
    assert omasks_pred.ndim == omasks_true.ndim == 3
    
    iou_mat = get_iou_mat(omasks_pred, omasks_true)
    n_pred, n_true = iou_mat.shape
    ious = iou_mat.max(0).values
    is_tp = ious > 0.5
    tp_iou_sum = ious[is_tp].sum()
    tp = is_tp.sum()

    # local
    eps = 1e-8
    recall = tp/(n_true+eps)
    precision = tp/(n_pred+eps)
    f1 = 2*recall*precision/(recall+precision+eps)
    sq = tp_iou_sum/(tp+eps)
    pq = sq * f1

    info = {
        'n_pred': n_pred,
        'n_true': n_true,
        'tp': tp,
        'tp_iou_sum': tp_iou_sum,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'sq': sq,
        'pq': pq,
        'n_sample': 1
    }
    return info

def get_batch_metric_info(batch_omasks_pred, batch_omasks_true):
    '''
    batch_omasks_pred: List[batch_size x torch.Tensor[n_pred_objects, height, width]]
    batch_omasks_true: List[batch_size x torch.Tensor[n_true_objects, height, width]]
    '''
    keys = ['n_pred', 'n_true', 'tp', 'tp_iou_sum', 'recall', 'precision', 'f1', 'sq', 'pq', 'n_sample']
    batch_info = {k: 0 for k in keys}
    for omasks_pred, omasks_true in zip(batch_omasks_pred, batch_omasks_true):
        info = get_metric_info(omasks_pred, omasks_true)
        for k, v in info.items():
            batch_info[k] += v
    return batch_info

def masks2omaps(masks:torch.Tensor, min_prob:Union[float,list]=0.5):
    '''
    masks: torch.Tensor[n_classes, height, width]
    '''
    assert masks.ndim == 3
    omaps = torch.zeros_like(masks).to(dtype=torch.int16)
    for i, mask in enumerate(masks):
        cur_min_prob = min_prob[i] if hasattr(min_prob, '__iter__') else min_prob
        mask = (mask>cur_min_prob).cpu().numpy().astype(np.uint8)
        n, omap = cv2.connectedComponents(mask, 4, cv2.CV_32S)
        omaps[i] = torch.from_numpy(omap).to(device=omaps.device, dtype=omaps.dtype)
    return omaps

def omaps2omasks_list(
        omaps:torch.Tensor,
        min_area:Union[float,list]=0,
        dilate_iter:Union[int,list]=0,
        expand_factor:Union[float,list]=0.,
        expand_max_dist:Union[int,list]=None,
        max_objects:Union[int,list]=None,
        ignore_zero:Union[bool,list]=True,
        masks:torch.Tensor=None
    ):
    '''
    omaps: torch.Tensor[n_classes, height, width]
    omasks_list: List[n_class x torch.Tensor[n_objects, height, width]]
    masks: torch.Tensor[n_classes, height, width] - if provided, compute score for each omask
    '''
    # omaps: [n, height, width]
    assert omaps.ndim == 3
    n, h, w = omaps.shape
    area = h * w
    omasks_list = []
    scores_list = []
    for j in range(len(omaps)):
        cur_min_area = min_area[j] if hasattr(min_area, '__iter__') else min_area
        cur_dilate_iter = dilate_iter[j] if hasattr(dilate_iter, '__iter__') else dilate_iter
        cur_expand_factor = expand_factor[j] if hasattr(expand_factor, '__iter__') else expand_factor
        cur_expand_max_dist = expand_max_dist[j] if hasattr(expand_max_dist, '__iter__') else expand_max_dist
        cur_max_objects = max_objects[j] if hasattr(max_objects, '__iter__') else max_objects
        cur_ignore_zero = ignore_zero[j] if hasattr(ignore_zero, '__iter__') else ignore_zero
        omap = omaps[j]
        omasks = []
        scores = []
        object_numbers = torch.unique(omap)
        if cur_max_objects is not None:
            object_numbers = object_numbers[-cur_max_objects:]
        for k in object_numbers:
            if cur_ignore_zero:
                if k == 0:
                    continue
            omask = (omap == k).to(torch.uint8)
            if masks is not None:
                score = ((masks[j] * omask).sum() / omask.sum()).item()
            if cur_expand_factor > 0:
                vertices = omask2vertices(omask.cpu().numpy())
                if len(vertices) >= 4:
                    omask = torch.from_numpy(vertices2omask(expand_vertices(vertices, factor=cur_expand_factor, max_dist=cur_expand_max_dist), h=h, w=w)).to(omask.device)
            if cur_dilate_iter > 0:
                omask = torch.from_numpy(cv2.dilate(omask.cpu().numpy(), np.ones((3, 3)), iterations=cur_dilate_iter)).to(omask.device)
            if omask.sum() > cur_min_area * area:
                omasks.append(omask)
                if masks is not None:
                    scores.append(score)
        omasks = torch.stack(omasks) if len(omasks) > 0 else torch.zeros_like(omaps[[0]])
        omasks_list.append(omasks.to(torch.uint8))
        if masks is not None:
            if len(scores) == 0: scores = [0]
            scores_list.append(scores)
    if masks is None:
        return omasks_list
    else:
        return omasks_list, scores_list
    

def omasks_list2omaps(omasks_list, channel_last=False):
    '''
    omasks_list: List[n_class x np.ndarray[n_objects, height, width]] (channel_last=False)
    '''
    if channel_last:
        omasks_list = [np.transpose(omasks, (2,0,1)) for omasks in omasks_list]
    omaps = np.zeros((len(omasks_list), omasks_list[0].shape[1], omasks_list[0].shape[2]), dtype=np.uint16)
    for i, omasks in enumerate(omasks_list):
        for j, omask in enumerate(omasks):
            omaps[i] = np.where(omask, j+1, omaps[i])  # 0 is class for ignorable background
    if channel_last:
        omaps = np.transpose(omaps, (1,2,0))
    return omaps


def remove_illegible_omasks(omasks:torch.Tensor, illegible_mask:torch.Tensor):
    '''
    omasks: torch.Tensor[n_objects, height, width]
    illegible_mask: torch.Tensor[height, width]
    '''
    assert omasks.ndim == 3
    assert illegible_mask.ndim == 2
    intersection = omasks * illegible_mask[None, :, :]
    illegible_ratio = intersection.sum((1,2)) / omasks.sum((1,2))
    legible_omasks = omasks[illegible_ratio<0.5]
    if legible_omasks.shape[0] == 0:
        legible_omasks = torch.zeros_like(omasks[[0]])
    return legible_omasks

def intersect_word_omasks(word_omasks:torch.Tensor, line_omasks:torch.Tensor, para_omasks:torch.Tensor):
    '''
    word_omasks: torch.Tensor[n_words, height, width]
    '''
    assert word_omasks.ndim == line_omasks.ndim == para_omasks.ndim == 3
    word_line_iou_mat = get_iou_mat(word_omasks, line_omasks)
    line_scores, line_indexes = word_line_iou_mat.max(1)
    line_para_iou_mat = get_iou_mat(line_omasks, para_omasks)
    para_scores, para_indexes = line_para_iou_mat.max(1)
    new_line_omasks = torch.zeros_like(line_omasks)
    for word_i, line_i in enumerate(line_indexes):
        if line_scores[word_i] > 0: # if intersection exists
            new_line_omasks[line_i] = new_line_omasks[line_i] | word_omasks[word_i]
    new_para_omasks = torch.zeros_like(para_omasks)
    for line_i, para_i in enumerate(para_indexes):
        if para_scores[line_i] > 0: # if intersection exists
            new_para_omasks[para_i] = new_para_omasks[para_i] | new_line_omasks[line_i]
    new_line_omasks = new_line_omasks[new_line_omasks.sum((1,2))>0]
    new_para_omasks = new_para_omasks[new_para_omasks.sum((1,2))>0]
    if new_line_omasks.shape[0] == 0:
        new_line_omasks = torch.zeros_like(line_omasks[[0]])
    if new_para_omasks.shape[0] == 0:
        new_para_omasks = torch.zeros_like(para_omasks[[0]])
    return word_omasks, new_line_omasks, new_para_omasks

def process_word_line_para_omasks_for_eval(word_omasks:torch.Tensor, line_omasks:torch.Tensor, para_omasks:torch.Tensor, illegible_masks:torch.Tensor):
    '''
    word_omasks: torch.Tensor[n_words, height, width]
    illegible_masks: torch.Tensor[3, heigt, width]
    '''
    assert word_omasks.ndim == line_omasks.ndim == para_omasks.ndim == 3
    assert (illegible_masks.ndim == 3) & (illegible_masks.shape[0] == 3)
    word_omasks, line_omasks, para_omasks = intersect_word_omasks(word_omasks, line_omasks, para_omasks)
    word_omasks, line_omasks, para_omasks = [remove_illegible_omasks(masks, illegible_masks[i]) for i, masks in enumerate([word_omasks, line_omasks, para_omasks])]
    return word_omasks, line_omasks, para_omasks

def shrink_vertices(vertices:np.ndarray, factor:float, max_dist:float=None):
    # polygon = Polygon(vertices)
    # distance = polygon.area * factor / polygon.length
    ((center_x, center_y), (width, height), angle) = cv2.minAreaRect(vertices[:, None, :].astype(np.float32))
    size = min(width, height)
    distance = size * factor/2
    if max_dist is not None:
        distance = min(distance, max_dist)
    subject = [tuple(l) for l in vertices]
    padding = pyclipper.PyclipperOffset()
    padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    vertices = padding.Execute(-distance)
    if len(vertices) == 0:
        return None
    vertices = np.array(vertices[0]).reshape(-1, 2)
    return vertices

def expand_vertices(vertices:np.ndarray, factor:float, max_dist:float=None):
    # polygon = Polygon(vertices)
    # distance = polygon.area * factor / polygon.length
    ((center_x, center_y), (width, height), angle) = cv2.minAreaRect(vertices[:, None, :].astype(np.int32))
    size = min(width, height)
    distance = size * factor/2
    if max_dist is not None:
        distance = min(distance, max_dist)
    subject = [tuple(l) for l in vertices]
    padding = pyclipper.PyclipperOffset()
    padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    vertices = padding.Execute(distance)
    vertices = np.array(vertices[0]).reshape(-1, 2)
    return vertices

def vertices2omask(vertices:np.ndarray, h:int, w:int):
    omask = np.zeros((h, w), dtype=np.uint8)
    if vertices is None:
        return omask
    cv2.fillPoly(omask, [np.array(vertices).astype(np.int32)], 1)
    return omask

def omask2vertices(omask:np.ndarray, approx:bool=False, convex_hull:bool=False):
    '''
    omask: np.ndarray[height, width]
    '''
    contours, _ = cv2.findContours(omask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        # print('no contours found')
        return
    contour = max(contours, key=cv2.contourArea)
    if approx:
        epsilon = 0.002 * cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, epsilon, True)
    if convex_hull:
        convex_contour = cv2.convexHull(contour, clockwise=True)
        convex_omask = vertices2omask(convex_contour, *omask.shape)
        if omask.sum() / convex_omask.sum() > 0.75:
            contour = convex_contour
    return contour[:, 0, :]

def normalize_vertices(vtx, target_h, target_w, src_h, src_w):
    vtx = np.array(vtx)
    vtx[:, 0] = vtx[:, 0] / src_w * target_w
    vtx[:, 1] = vtx[:, 1] / src_h * target_h
    return vtx.round().astype(np.int32).tolist()

def masks2submission_paras(
        masks:torch.Tensor,
        image_height:int,
        image_width:int,
        min_prob:float=0.5,
        component_min_area:float=0,
        mask_dilate_iter:int=0,
        vertices_expand_factor:int=0,
        vertices_expand_max_dist:float=None,
        promotion_min_score:float=1,
    ):
    '''
    masks: torch.Tensor[n_classes, height, width]
    '''
    omaps = masks2omaps(masks, min_prob=min_prob)
    (words, lines, paras), (word_scores, line_scores, para_scores) = omaps2omasks_list(
        omaps,
        min_area=component_min_area,
        dilate_iter=mask_dilate_iter,
        expand_factor=vertices_expand_factor,
        expand_max_dist=vertices_expand_max_dist,
        masks=masks
    )
    words, lines, paras = map(lambda x: torch.nn.functional.interpolate(x[None, ...], (image_height, image_width))[0], [words, lines, paras])
    submission_lines = [{
        'text': '',
        'legible': 1,
        'words': [],
        'vertices': omask2vertices(line, convex_hull=True).tolist() if line.any() else None,
        'score': score,
    } for line, score in zip(lines.cpu().numpy(), line_scores)]
    iou_mat = get_iou_mat(words, lines)
    line_ious, line_indexes = iou_mat.max(1)
    for word_i, line_i in enumerate(line_indexes):
        contour = omask2vertices(words[word_i].cpu().numpy(), convex_hull=True)
        if contour is None: continue
        contour = contour.tolist()
        if len(contour) > 2: # contour should have at least 3 vertices
            # contour = normalize_vertices(contour, image_height, image_width, *words[word_i].shape)
            word_score = word_scores[word_i]
            if line_ious[word_i] > 0: # if intersection exists
                submission_lines[line_i]['words'].append({'text': '', 'legible': 1, 'vertices': contour, 'score': word_score})
            elif word_score > promotion_min_score:
                lines = torch.cat((lines, words[word_i].unsqueeze(0)), dim=0)
                line_scores.append(word_score)
                submission_lines.append({'text': '', 'legible': 1, 'words': [{'text': '', 'legible': 1, 'vertices': contour}], 'vertices': contour, 'score': word_score})
    submission_paras = [{
        'text': '',
        'lines': [],
        'legible': 1,
        'vertices': omask2vertices(para, convex_hull=True).tolist() if para.any() else None,
        'score': score
    } for para, score in zip(paras.cpu().numpy(), para_scores)]
    iou_mat = get_iou_mat(lines, paras)
    para_ious, para_indexes = iou_mat.max(1)
    for line_i, para_i in enumerate(para_indexes):
        if len(submission_lines[line_i]['words']) > 0:
            line_score = line_scores[line_i]
            if para_ious[line_i] > 0: # if intersection exists
                submission_paras[para_i]['lines'].append(submission_lines[line_i])
            elif line_score > promotion_min_score:
                submission_paras.append({'text': '', 'lines': [submission_lines[line_i]], 'legible': 1, 'vertices': submission_lines[line_i]['vertices'], 'score': line_score})
    submission_paras = [x for x in submission_paras if len(x['lines']) > 0]
    return submission_paras

def batch_masks2submission(
        batch_masks:torch.Tensor, 
        image_ids:list, 
        image_heights:list, 
        image_widths:list, 
        min_prob:float=0.5, 
        component_min_area:float=0, 
        mask_dilate_iter:int=0, 
        vertices_expand_factor:float=0, 
        vertices_expand_max_dist:float=None,
        promotion_min_score:float=1,
        device:str='cuda:0'
    ):
    '''
    batch_masks: torch.Tensor[batch_size, n_classes, height, width]
    '''
    submission = {'annotations': []}
    for i in tqdm(range(len(batch_masks)), desc='batch_masks2submission'):
        submissionparas = masks2submission_paras(
            batch_masks[i].to(device),
            image_heights[i],
            image_widths[i],
            min_prob=min_prob,
            component_min_area=component_min_area,
            mask_dilate_iter=mask_dilate_iter,
            vertices_expand_factor=vertices_expand_factor,
            vertices_expand_max_dist=vertices_expand_max_dist,
            promotion_min_score=promotion_min_score
        )
        submission_single = {
            'image_id': image_ids[i],
            'image_height': image_heights[i], 
            'image_width': image_widths[i],
            'paragraphs': submissionparas
        }
        submission['annotations'].append(submission_single)
    return submission

def visualize(pixel_values, omaps):
    if type(pixel_values) == torch.Tensor:
        pixel_values = pixel_values.detach().cpu().numpy()
    if type(omaps) == torch.Tensor:
        omaps = omaps.detach().cpu().numpy()
    if pixel_values.shape[-1] != 3:
        pixel_values = np.transpose(pixel_values, (1,2,0))
    if omaps.shape[-1] != 3:
        omaps = np.transpose(omaps, (1,2,0))
    if pixel_values.dtype == np.float32:
        image_means = np.array([122.67891434, 116.66876762, 104.00698793])
        image_stds = np.array([255, 255, 255])
        pixel_values = (pixel_values * image_stds[None, None, :] + image_means[None, None, :]).astype(np.uint8)
    for j in range(3):
        omap = omaps[:, :, j]
        rgb = (255 if j == 0 else 0, 255 if j == 1 else 0, 255 if j == 2 else 0)
        for i in np.unique(omap):
            if i != 0:
                contour = omask2vertices((omap == i).astype(np.uint8))
                # contour = cv2.convexHull(contour, clockwise=True)
                pixel_values = cv2.drawContours(pixel_values.copy(), [contour], 0, rgb, 2)
    return Image.fromarray(pixel_values)

def resize_masks(masks:np.ndarray, height:int, width:int):
    '''
    masks: np.ndarray[height, width, n]
    '''
    chunk_size = 500
    chunks = []
    for sid in range(0, masks.shape[-1], chunk_size):
        chunk = cv2.resize(masks[:, :, sid:sid+chunk_size], (height, width))
        if chunk.ndim == 2: chunk = chunk[:, :, None]                    
        chunks.append(chunk)
    masks = np.concatenate(chunks, axis=-1)
    return masks

def erode_masks(masks:np.ndarray, iterations=1):
    '''
    masks: np.ndarray[height, width, n]
    '''
    chunk_size = 500
    eroded_chunks = []
    for sid in range(0, masks.shape[-1], chunk_size):
        chunk = masks[:, :, sid:sid+chunk_size]
        eroded_chunk = cv2.erode(chunk, np.ones((3, 3)), iterations=iterations)
        if eroded_chunk.ndim == 2: eroded_chunk = eroded_chunk[:, :, None]
        removed_bool = eroded_chunk.sum((0,1)) == 0
        eroded_chunk[:, :, removed_bool] = chunk[:, :, removed_bool]
        eroded_chunks.append(eroded_chunk)
    masks = np.concatenate(eroded_chunks, axis=-1)
    return masks


def erode_mask_adaptive(mask, factor):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    ((center_x, center_y), (width, height), angle) = cv2.minAreaRect(contour)
    size = min(width, height)
    iterations = round(size * factor/2)
    if iterations > 0:
        mask= cv2.erode(mask, np.ones((3,3)), iterations=iterations)
    return mask


def dilate_mask_adaptive(mask, factor):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    ((center_x, center_y), (width, height), angle) = cv2.minAreaRect(contour)
    size = min(width, height)
    iterations = round(size * factor/2)
    if iterations > 0:
        mask= cv2.dilate(mask, np.ones((3,3)), iterations=iterations)
    return mask 


def parse_annotation(ann, image_dir='input/validation'):
    image = cv2.imread(os.path.join(image_dir, ann['image_id']+'.jpg'))
    word_polygons = []
    line_polygons = []
    para_polygons = []
    word_legibles = []
    line_legibles = []
    para_legibles = []
    for para in ann['paragraphs']:
        para_polygons.append(para['vertices'])
        para_legibles.append(para['legible'])
        for line in para['lines']:
            line_polygons.append(line['vertices'])
            line_legibles.append(line['legible'])
            for word in line['words']:
                word_polygons.append(word['vertices'])
                word_legibles.append(word['legible'])
    out = {
        'image_id': ann['image_id'],
        'image_height': ann['image_height'],
        'image_width': ann['image_width'],
        'image': image,
        'word_polygons': word_polygons,
        'line_polygons': line_polygons,
        'para_polygons': para_polygons,
        'word_legibles': word_legibles,
        'line_legibles': line_legibles,
        'para_legibles': para_legibles,
    }
    return out



def rand_bbox(size, lam):
    """
    Retuns the coordinate of a random rectangle in the image for cutmix.
    Args:
        size (torch tensor [batch_size x c x W x H): Input size.
        lam (int): Lambda sampled by the beta distribution. Controls the size of the squares.
    Returns:
        int: 4 coordinates of the rectangle.
        int: Proportion of the unmasked image.
    """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    return bbx1, bby1, bbx2, bby2, lam


def cutmix_data(x, y, z, alpha=1.0, device="cuda"):
    """
    Applies cutmix to a sample
    Args:
        x (torch tensor [batch_size x input_size]): Input batch.
        y (torch tensor [batch_size x num_classes]): Labels.
        alpha (float, optional): Parameter of the beta distribution. Defaults to 1..
        device (str, optional): Device for torch. Defaults to "cuda".
    Returns:
        torch tensor [batch_size x input_size]: Mixed input.
        torch tensor [batch_size x num_classes]: Mixed labels.
        float: Probability sampled by the beta distribution.
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    index = torch.randperm(x.size()[0]).to(device)

    bbx1, bby1, bbx2, bby2, lam = rand_bbox(x.size(), lam)

    mixed_x = x.clone()
    mixed_y = y.clone()
    mixed_z = z.clone()

    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    mixed_y[:, :, bbx1:bbx2, bby1:bby2] = y[index, :, bbx1:bbx2, bby1:bby2]
    mixed_z[:, :, bbx1:bbx2, bby1:bby2] = z[index, :, bbx1:bbx2, bby1:bby2]

    #             state.input[f][:, :, bbx1:bbx2, bby1:bby2] = state.input[f][self.index, :, bbx1:bbx2, bby1:bby2]


    return mixed_x, mixed_y, mixed_z




def polygon2rect(polygon):
    rect = cv2.minAreaRect(np.array(polygon))
    rect = cv2.boxPoints(rect)
    rect = np.round(rect).astype(np.int64)
    return rect
import torch
import pytorch_lightning as pl
import transformers
from torchmetrics import Metric
from timm.optim import create_optimizer_v2

from model import Model
from utils import *
from losses import symmetric_lovasz
from combo_loss import ComboLoss


class Accumulator(Metric):
    
    def __init__(self, dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        for object in ['word', 'line', 'para']:
            for to_accum in ['n_pred', 'n_true', 'tp', 'tp_iou_sum', 'recall', 'precision', 'f1', 'sq', 'pq', 'n_sample']:
                self.add_state(f'{object}_{to_accum}', default=torch.tensor(0.), dist_reduce_fx='sum')

    def update(self, word_metric_info:dict, line_metric_info:dict, para_metric_info:dict):
        for object, info in [('word', word_metric_info), ('line', line_metric_info), ('para', para_metric_info)]:
            for k, v in info.items():
                state_name = f'{object}_{k}'
                setattr(self, state_name, getattr(self, state_name) + torch.tensor(v))

    def compute(self):
        out = {}
        for object in ['word', 'line', 'para']:
            tp = getattr(self, f'{object}_tp')
            recall = tp / getattr(self, f'{object}_n_true')
            precision = tp / getattr(self, f'{object}_n_pred')
            f1 = 2*recall*precision/(recall+precision)
            sq = getattr(self, f'{object}_tp_iou_sum') / tp
            pq = sq * f1
            out[f'{object}_pq'] = pq.item()
            out[f'{object}_f1'] = f1.item()
        for metric in ['pq', 'f1']:
            out[f'mean_{metric}'] = hmean([out[f'{object}_{metric}'] for object in ['word', 'line', 'para']])
        return out
    

class LightningModule(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model = Model(cfg)
        self.accumulator = Accumulator()
    
    def refine_mask(self, batch_masks_prob):
        for idx in range(3):
            cls_seg_map = batch_masks_prob[:, idx, :, :]
            cls_bound_map = 1 - batch_masks_prob[:, idx+3, :, :]
            batch_masks_prob[:, idx, :, :] = cls_seg_map * cls_bound_map
        return batch_masks_prob
        
    @torch.no_grad()
    def forward(self, data):
        batch_masks_logit = self.model(**data)  # [batch_size, n_classes, height, width]
        batch_masks_prob = torch.sigmoid(batch_masks_logit) #[:, :3, :, :]
        
        #if self.hparams.refine_mask:
        #batch_masks_prob = self.refine_mask(batch_masks_prob)
        
        out = {
            'batch_masks_logit': batch_masks_logit,
            'batch_masks_prob': batch_masks_prob,
        }
        return out
    
    @torch.no_grad()
    def evaluate(self, batch_masks_prob_pred:torch.Tensor, batch_omaps_true:torch.Tensor, batch_illegible_masks:torch.Tensor):
        batch_masks_prob_pred = batch_masks_prob_pred[:, :3, :, :]
        #batch_omaps_true = batch_omaps_true[:, :3, :, :]
        #batch_illegible_masks = batch_illegible_masks[:, :3, :, :]

       # print(batch_masks_prob_pred.shape, batch_omaps_true.shape, batch_illegible_masks.shape)

        batch_omaps_pred = torch.stack([masks2omaps(x, min_prob=self.hparams.component_min_prob) for x in batch_masks_prob_pred])
        batch_word_omasks_pred = []
        batch_line_omasks_pred = []
        batch_para_omasks_pred = []
        batch_word_omasks_true = []
        batch_line_omasks_true = []
        batch_para_omasks_true = []       
        batch_illegible_masks *= batch_omaps_true[:, [0]] != 0  # intersect illegible masks with true word masks
        for omaps_pred, omaps_true, illegible_masks in zip(batch_omaps_pred, batch_omaps_true, batch_illegible_masks):
            word_omasks_pred, line_omasks_pred, para_omasks_pred = omaps2omasks_list(
                omaps=omaps_pred,
                min_area=self.hparams.component_min_area,
                dilate_iter=self.hparams.pred_mask_dilate_iter,
                expand_ratio=self.hparams.pred_vertices_expand_ratio,
                max_objects=self.hparams.max_objects_for_eval
            )
            word_omasks_pred, line_omasks_pred, para_omasks_pred = process_word_line_para_omasks_for_eval(word_omasks_pred, line_omasks_pred, para_omasks_pred, illegible_masks)
            batch_word_omasks_pred.append(word_omasks_pred)
            batch_line_omasks_pred.append(line_omasks_pred)
            batch_para_omasks_pred.append(para_omasks_pred)
            word_omasks_true, line_omasks_true, para_omasks_true = omaps2omasks_list(
                omaps=omaps_true,
                min_area=0,
                dilate_iter=0,
                expand_ratio=0,
                max_objects=self.hparams.max_objects_for_eval
            )
            word_omasks_true, line_omasks_true, para_omasks_true = process_word_line_para_omasks_for_eval(word_omasks_true, line_omasks_true, para_omasks_true, illegible_masks)
            batch_word_omasks_true.append(word_omasks_true)
            batch_line_omasks_true.append(line_omasks_true)
            batch_para_omasks_true.append(para_omasks_true)
        word_metric_info = get_batch_metric_info(batch_word_omasks_pred, batch_word_omasks_true)
        line_metric_info = get_batch_metric_info(batch_line_omasks_pred, batch_line_omasks_true)
        para_metric_info = get_batch_metric_info(batch_para_omasks_pred, batch_para_omasks_true)
        out = {
            'word_metric_info': word_metric_info,
            'line_metric_info': line_metric_info,
            'para_metric_info': para_metric_info,
        }
        return out

    def compute_loss(self, pred, true, boundary_mask, illegible_mask):
        
        boundary_pred = pred[:, 3:, :, :]
        pred = pred[:, :3, :, :]
        
        assert pred.shape == true.shape == illegible_mask.shape
        legible_weight = 1-illegible_mask + illegible_mask*self.hparams.illegible_weight
        
        comb_loss = 0.0
        if self.hparams.combo_loss_coeff > 0:
            # combo_loss
            comb_loss_fn = ComboLoss(weights={'dice': 1, 'focal': 1, 'bce': 0.5})
            #legible_pred, legible_true = map(lambda x: x*legible_weight, [pred, true])

            if boundary_pred.shape[1] == 6:
                comb_loss = comb_loss_fn(boundary_pred[:, :3, :, :], boundary_mask[:, :3, :, :])
                comb_loss += comb_loss_fn(boundary_pred[:, 3:, :, :], boundary_mask[:, 3:, :, :])
            else:
                comb_loss = comb_loss_fn(boundary_pred, boundary_mask)
        
        # center_loss = 0.0
        # if self.hparams.center_loss_coeff > 0:
        #     center_loss = torch.nn.MSELoss()(boundary_pred, boundary_mask)
        #     center_loss *= 50

        # BCE LOSS
        bce_loss = 0
        if self.hparams.bce_coef > 0:
            bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')(pred, true)
            # weight
            weight = torch.ones_like(loss)
            # focal
            if self.hparams.focal > 0:
                with torch.no_grad():
                    probas = torch.sigmoid(pred)
                focal_weight = torch.where(true >= 0.5, self.hparams.pos_weight*(1.-probas)**self.hparams.focal, probas**self.hparams.focal)
                weight *= focal_weight
            # illegible
            weight *= legible_weight
            bce_loss = (bce_loss * weight).sum() / weight.sum()
        # IOU LOSS
        iou_loss = 0
        if self.hparams.iou_coef > 0:
            pred_prob = torch.sigmoid(pred)
            legible_pred_prob, legible_true = map(lambda x: x*legible_weight, [pred_prob, true])
            intersection = (legible_pred_prob * legible_true).sum((2,3))
            union = (legible_pred_prob + legible_true).sum((2,3)) - intersection
            iou = (intersection+1)/(union+1)
            iou_loss = (1 - iou.mean())
        lovasz_loss = 0
        if self.hparams.lovasz_coef > 0:
            b, c, h, w = pred.shape
            pred = pred.contiguous()
            true = true.contiguous()
            illegible_mask = illegible_mask.contiguous()
            lovasz_loss = symmetric_lovasz(pred.view(b*c, h, w), true.view(b*c, h, w), (1-illegible_mask).view(b*c, h, w))
        # FINAL LOSS
        loss = (self.hparams.bce_coef*bce_loss + self.hparams.iou_coef*iou_loss + self.hparams.lovasz_coef*lovasz_loss + self.hparams.combo_loss_coeff * comb_loss) / (self.hparams.bce_coef+self.hparams.iou_coef+self.hparams.lovasz_coef + self.hparams.combo_loss_coeff)
        return loss
    
    def training_step(self, data, batch_idx):
        
        if self.hparams.cutmix:
            if np.random.random() > self.hparams.mix_prob:
                data['pixel_values'], data['mask_labels'], data['illegible_masks'] = cutmix_data(data['pixel_values'], data['mask_labels'], data['illegible_masks'], alpha=self.hparams.mix_alpha)
            
        batch_masks_logit = self.model(**data)
        loss = self.compute_loss(batch_masks_logit, data['mask_labels'], data['bounday_masks'], data['illegible_masks'])
        self.log('trn/loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def validation_step(self, data, batch_idx):
        out = self(data)
        loss = self.compute_loss(out['batch_masks_logit'], data['mask_labels'], data['bounday_masks'], data['illegible_masks'])
        self.log('val/loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        eval_out = self.evaluate(out['batch_masks_prob'], data['omaps'], data['illegible_masks'])
        self.accumulator.update(**eval_out)
        return out
    
    def validation_epoch_end(self, outputs):
        metric = self.accumulator.compute()
        self.accumulator.reset()
        for k, v in metric.items():
            self.log(f'val/{k}', v, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        # define optimizer
        # if self.hparams.optimizer == 'sgd':
        #     optimizer = torch.optim.SGD(self.trainer.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        # elif self.hparams.optimizer == 'adamw':
        #     optimizer = torch.optim.AdamW(self.trainer.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        # elif self.hparams.optimizer == 'adam':
        #     optimizer = torch.optim.Adam(self.trainer.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        # else:
        #     raise NotImplementedError

        optimizer = create_optimizer_v2(self.trainer.model.parameters(), 'adamw', lr=self.hparams.lr, weight_decay=self.hparams.wd)

        # define lr scheduler
        if self.hparams.scheduler == 'constant':
            scheduler = transformers.get_constant_schedule_with_warmup(optimizer, self.hparams.num_warmup_steps)
        elif self.hparams.scheduler == 'cosine':
            scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer, self.hparams.num_warmup_steps, self.hparams.num_training_steps, num_cycles=1
            )
        elif self.hparams.scheduler == 'linear':
            scheduler = transformers.get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.hparams.num_warmup_steps, num_training_steps=self.hparams.num_training_steps
            )
        else:
            return optimizer
        scheduler = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]
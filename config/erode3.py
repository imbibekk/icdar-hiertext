from omegaconf import OmegaConf

config = OmegaConf.create({
    # io
    'input_dir': '/data/project/yoonsoo/projects/challenges-icdar2023-detector/yoonsoo/input',
    'output_dir': '/data/project/yoonsoo/projects/challenges-icdar2023-detector/yoonsoo/output',
    # data
    'trn_resolution': 1024, # 1600
    'val_resolution': 1024,
    'image_means': (122.67891434/255, 116.66876762/255, 104.00698793/255),
    'image_stds': (1., 1., 1.),
    'aug': True,
    'true_mask_erode_iter': 3,
    'true_vertices_shrink_ratio': 0,
    # postprocess
    'component_min_prob': 0.5,
    'component_min_area': 0,
    'max_objects_for_eval': 500,
    'pred_mask_dilate_iter': 3,
    'pred_vertices_expand_ratio': 0,
    # training
    'batch_size': 12,
    'val_batch_size': 12,
    'max_epochs': 100,
    'swa_proportion': None,
    'gradient_clip_val': None,
    'optimizer': 'adamw',
    'scheduler': 'cosine',
    'lr': 3e-4,
    'wd': 0,
    'num_warmup_steps': 0,
    # loss
    'bce_coef': 0,
    'illegible_weight': 0,
    'pos_weight': 1,
    'focal': 0,
    'iou_coef': 0,
    'lovasz_coef': 1,
    # setting
    'seed': 0,
    'debug': False,
    'num_workers': 12,
    'check_val_every_n_epoch': 10,
    'num_sanity_val_steps': 0,
    # device
    'accelerator': 'gpu',
    'num_nodes': 1,
    'devices': [0,1,2,3],
    'strategy': 'ddp',  # ddp
    'precision': 16,
    # model
    'encoder_name': 'tu-tf_efficientnet_b7_ns', # 'tu-tf_efficientnet_l2_ns',
    # pretrained
    'checkpoint_path': None,
    'pretrained_path': '/data/project/kfaceapi/repos/workspace/icdar2023/challenges-icdar2023-svrd/limerobot/results/cfg_pretrain_512_synthText_effiB7_Unet_lovaszLoss_schLinear_lr1e4_ep20/202303091415/ep19_f1_-0.53987.pth'
})

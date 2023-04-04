from omegaconf import OmegaConf

config = OmegaConf.create({
    # io
    'input_dir': '/data/project/bibek/icdar/hier/data/HierText',
    'img_path': '/data/project/bibek/icdar/hier/data/HierText',
    'output_dir': './output',
    # data
    'trn_resolution': 1024, # 1600
    'val_resolution': 1024,
    'image_means': (122.67891434/255, 116.66876762/255, 104.00698793/255),
    'vertices_shrink_factor': 0,
    'vertices_shrink_max_dist_factor': 0,
    'gaps': [1, 2, 1], # word, line, para
    'image_stds': (1., 1., 1.),
    'aug': True,
    'true_mask_erode_iter': 0, #2,
    'true_vertices_shrink_ratio': 0,
    'mask_erode_iter': 2,
    # postprocess
    #'component_min_prob': [0.4, 0.8, 0.6],
    #'component_min_area': 0,
    #'max_objects_for_eval': 500,
    #'pred_mask_dilate_iter': 2,
    #'pred_vertices_expand_ratio': 0,

    'component_min_probs': [0.4, 0.8, 0.6], # word, line, para
    'component_min_area': 0,
    'max_objects_for_eval': 500,
    'promotion_min_score': 0.9,
    
    # training
    'batch_size': 12,
    'val_batch_size': 12,
    'max_epochs': 150,
    'swa_proportion': 1,
    'gradient_clip_val': None,
    'optimizer': 'adamw',
    'scheduler': 'cosine',
    'lr': 3e-4,
    'wd': 0,
    'num_warmup_steps': 0,
    'mix_prob': 0.5,
    'mix_alpha': 0.4,
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
    'devices': [1,2,4,6],
    'strategy': 'ddp',  # ddp
    'precision': 16,
    'cutmix': 1,
    'combo_loss_coeff': 1,
    # model
    'encoder_name': 'tu-tf_efficientnetv2_l_in21ft1k',
    # pretrained
    'checkpoint_path': None,
    'pretrained_path': '/data/project/bibek/icdar/hier/oclip/logs/lr=1e-05_wd=0.1_agg=True_model=RN50_batchsize=32_workers=8_date=2023-03-24-01-47-59/checkpoints/epoch_30.pt',
    #'/data/project/kfaceapi/repos/workspace/icdar2023/challenges-icdar2023-svrd/limerobot/results/cfg_pretrain_512_synthText_effiB7_Unet_lovaszLoss_schLinear_lr1e4_ep20/202303091415/ep19_f1_-0.53987.pth',
    #'/data/project/bibek/icdar/hier/oclip/outputs/effnet_v2_hiertext/lr=0.0001_wd=0.1_agg=True_model=RN50_batchsize=32_workers=8_date=2023-03-17-21-06-25/checkpoints/epoch_100.pt',
    #'/data/project/kfaceapi/repos/workspace/icdar2023/challenges-icdar2023-svrd/limerobot/results/cfg_pretrain_512_synthText_effiB7_Unet_lovaszLoss_schLinear_lr1e4_ep20/202303091415/ep19_f1_-0.53987.pth',
    #'/data/project/bibek/icdar/hier/oclip/outputs/hiertext/lr=0.0001_wd=0.1_agg=True_model=RN50_batchsize=32_workers=1_date=2023-03-16-01-42-12/checkpoints/epoch_100.pt'
    #'/data/project/kfaceapi/repos/workspace/icdar2023/challenges-icdar2023-svrd/limerobot/results/cfg_pretrain_512_synthText_effiB7_Unet_lovaszLoss_schLinear_lr1e4_ep20/202303091415/ep19_f1_-0.53987.pth'
})

import numpy as np
from pathlib import Path
import importlib
from argparse import ArgumentParser
import pytorch_lightning as pl
import torch
import shutil
from pytorch_lightning.callbacks import StochasticWeightAveraging, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from dotenv import load_dotenv
load_dotenv()

from lightning_module import LightningModule
from data import get_combined_dataloader, get_dataloader

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--config_name', type=str, default='default')
    parser.add_argument('--run_name', type=str, default='default')
    args = parser.parse_args()

    # hyperparameters
    cfg = getattr(importlib.import_module(f'config.{args.config_name}'), 'config')
    cfg.run_name = f'{args.config_name}/{args.run_name}'
    output_dir = Path(cfg.output_dir) / args.config_name

    pl.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision('high')

    model_dir = Path(output_dir) / 'model'
    if model_dir.is_dir():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True)
        
    # logger
    #logger = WandbLogger(name=f'{cfg.run_name}', save_dir=output_dir, project='hiertext')
    logger = TensorBoardLogger( save_dir=output_dir, name=f'{cfg.run_name}')


    # define callbacks
    callbacks = []
    if cfg.swa_proportion:
        swa_callback = StochasticWeightAveraging(swa_lrs=1e-5, swa_epoch_start=0.75) #swa_epoch_start=int(cfg.max_epochs * (1-cfg.swa_proportion)), annealing_epochs=0)
        callbacks.append(swa_callback)
    lr_monitor = LearningRateMonitor()
    callbacks.append(lr_monitor)
    model_checkpoint = ModelCheckpoint(dirpath=model_dir, save_last=True, save_top_k=5, mode='max', monitor='val/mean_pq', save_weights_only=False)
    callbacks.append(model_checkpoint) # mode='max',p

    # dataloader
    train_loader = get_dataloader(cfg, 'validation') if cfg.debug else get_dataloader(cfg, 'train')
    valid_loader = None if cfg.debug else get_dataloader(cfg, 'validation')
    cfg.num_training_steps = int(np.ceil(cfg.max_epochs * len(train_loader) / len(cfg.devices)))

    # lightning module
    if cfg.checkpoint_path is not None:
        module = LightningModule.load_from_checkpoint(
            cfg.checkpoint_path,
        )
        #module.hparams.lr = 3e-5
        #module.hparams.devices = [0,1,2,7]
        print(module.hparams)
    else:
        module = LightningModule(cfg)

    # create trainer and train
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        gradient_clip_val=cfg.gradient_clip_val,
        accelerator=cfg.accelerator,
        num_nodes=cfg.num_nodes,
        devices=cfg.devices,
        strategy=cfg.strategy,
        precision=cfg.precision,
        benchmark=True,
        log_every_n_steps=1,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
        num_sanity_val_steps=cfg.num_sanity_val_steps,
        logger=logger,
        callbacks=callbacks
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=valid_loader)
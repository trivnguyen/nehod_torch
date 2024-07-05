
import os
import pickle
import sys
import shutil
import yaml

import ml_collections
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch
from torch.utils.data import DataLoader, TensorDataset
from absl import flags, logging
from ml_collections import config_flags

import datasets
from models import vdm, utils

logging.set_verbosity(logging.INFO)

def train(
    config: ml_collections.ConfigDict, workdir: str = "./logging/"
):
    # set up work directory
    if not hasattr(config, "name"):
        name = utils.get_random_name()
    else:
        name = config["name"]
    logging.info("Starting training run {} at {}".format(name, workdir))

    # set up random seed
    pl.seed_everything(config.seed)

    workdir = os.path.join(workdir, name)
    checkpoint_path = None
    if os.path.exists(workdir):
        if config.overwrite:
            shutil.rmtree(workdir)
        elif config.get('checkpoint', None) is not None:
            checkpoint_path = os.path.join(
                workdir, 'lightning_logs/checkpoints', config.checkpoint)
        else:
            raise ValueError(
                f"Workdir {workdir} already exists. Please set overwrite=True "
                "to overwrite the existing directory.")

    os.makedirs(workdir, exist_ok=True)

    # Save the configuration to a JSON file
    config_dict = config.to_dict()
    config_path = os.path.join(workdir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    # Read in and prepare the dataset
    train_x, train_cond, train_mask, norm_dict = datasets.read_preprocess_dataset(
        data_root=config.data.data_root,
        data_name=config.data.data_name,
        conditioning_parameters=config.data.conditioning_parameters,
        flag='train',
        invert_mask=True,
    )
    val_x, val_cond, val_mask, _ = datasets.read_preprocess_dataset(
        data_root=config.data.data_root,
        data_name=config.data.data_name,
        conditioning_parameters=config.data.conditioning_parameters,
        flag='val',
        invert_mask=True,
        norm_dict=norm_dict,
    )
    train_loader = datasets.create_dataloader(
        (train_x, train_cond, train_mask), batch_size=config.training.batch_size,
        shuffle=True, pin_memory=torch.cuda.is_available())
    val_loader = datasets.create_dataloader(
        (val_x, val_cond, val_mask), batch_size=config.training.batch_size,
        shuffle=False, pin_memory=torch.cuda.is_available())

    # Create the VDM model
    model = vdm.VariationalDiffusionModel(
        d_in=config.vdm.d_in,
        d_cond=config.vdm.d_cond,
        d_context_embedding=config.vdm.d_context_embedding,
        embed_context=config.vdm.embed_context,
        score_model_args=config.vdm.score_model,
        noise_schedule_args=config.vdm.noise_schedule,
        timesteps=config.vdm.timesteps,
        antithetic_time_sampling=config.vdm.antithetic_time_sampling,
        use_encdec=config.vdm.use_encdec,
        training_args=config.training,
        optimizer_args=config.optimizer,
        scheduler_args=config.scheduler,
        norm_dict=norm_dict
    )

    # Create Trainer and start training
    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor='val_loss', patience=config.training.patience,
            mode='min', verbose=True),
        pl.callbacks.ModelCheckpoint(
            filename="{epoch}-{val_loss:.4f}", monitor='val_loss',
            save_top_k=10, mode='min', save_weights_only=False),
        pl.callbacks.LearningRateMonitor("step"),
    ]
    train_logger = pl_loggers.TensorBoardLogger(workdir, version='')

    trainer = pl.Trainer(
        default_root_dir=workdir,
        max_steps=config.training.max_steps,
        accelerator='gpu',
        callbacks=callbacks,
        logger=train_logger,
        enable_progress_bar=True,
        inference_mode=False,
        gradient_clip_val=config.optimizer.grad_clip,
        val_check_interval=config.training.val_check_interval,
        check_val_every_n_epoch=None,
        log_every_n_steps=config.training.log_every_n_steps,
    )

    # train the model
    trainer.fit(
        model, train_loader, val_loader,
        ckpt_path=checkpoint_path
    )


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file(
        "config",
        None,
        "File path to the training or sampling hyperparameter configuration.",
        lock_config=True,
    )
    # Parse flags
    FLAGS(sys.argv)

    # Start training run
    train(config=FLAGS.config, workdir=FLAGS.config.workdir)
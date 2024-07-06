
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
from models import flows, utils

logging.set_verbosity(logging.INFO)

def read_flows_dataset(config):
    """ Hacky way to read flows data using datasets. Not optimized but works. """

    parameters = config.data.target_parameters + config.data.conditioning_parameters
    num_target = len(config.data.target_parameters)

    # Read and preprocess the dataset
    _, train_params, _, norm_dict = datasets.read_preprocess_dataset(
        data_root=config.data.data_root,
        data_name=config.data.data_name,
        conditioning_parameters=parameters,
        flag='train',
    )
    _, val_params, _, _ = datasets.read_preprocess_dataset(
        data_root=config.data.data_root,
        data_name=config.data.data_name,
        conditioning_parameters=parameters,
        flag='val',
        norm_dict=norm_dict,
    )

    # Separate the target and conditioning parameters
    train_target = train_params[:, :num_target]
    train_cond = train_params[:, num_target:]
    val_target = val_params[:, :num_target]
    val_cond = val_params[:, num_target:]

    # Create the dataloaders
    train_loader = datasets.create_dataloader(
        (train_target, train_cond), batch_size=config.training.batch_size,
        shuffle=True, pin_memory=torch.cuda.is_available())
    val_loader = datasets.create_dataloader(
        (val_target, val_cond), batch_size=config.training.batch_size,
        shuffle=False, pin_memory=torch.cuda.is_available())

    # Restructure norm_dict
    new_norm_dict = {
        'target_mean': norm_dict['cond_mean'][:num_target],
        'target_std': norm_dict['cond_std'][:num_target],
        'cond_mean': norm_dict['cond_mean'][num_target:],
        'cond_std': norm_dict['cond_std'][num_target:],
    }

    return train_loader, val_loader, new_norm_dict

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
    train_loader, val_loader, norm_dict = read_flows_dataset(config)

    # Create the flows model
    model = flows.NPE(
        in_dim=config.flows.in_dim,
        context_dim=config.flows.context_dim,
        hidden_dims=config.flows.hidden_dims,
        projection_dims=config.flows.projection_dims,
        dropout=config.flows.dropout,
        num_transforms=config.flows.num_transforms,
        optimizer_args=config.optimizer,
        scheduler_args=config.scheduler,
        norm_dict=norm_dict,
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

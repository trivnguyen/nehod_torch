
import torch
import torch.nn as nn
import pytorch_lightning as pl
import zuko

from nehod import train_utils

class NPE(pl.LightningModule):
    def __init__(
        self, in_dim, context_dim, hidden_dims, projection_dims, num_transforms=None,
        dropout=0.1, optimizer_args=None, scheduler_args=None, norm_dict=None
    ):
        super().__init__()
        self.in_dim = in_dim
        self.context_dim = context_dim
        self.hidden_dims = hidden_dims
        self.projection_dims = projection_dims
        self.num_transforms = num_transforms
        self.dropout = nn.Dropout(dropout)
        self.optimizer_args = optimizer_args
        self.scheduler_args = scheduler_args
        self.norm_dict = norm_dict
        self.save_hyperparameters()

        self.train_losses = []
        self.train_steps = []
        self.val_losses = []
        self.val_steps = []

        # backward compatibility
        if num_transforms is None:
            num_transforms = len(hidden_dims)

        if projection_dims is None:
            self.lin_proj_layers = nn.Identity()
            self.flow = zuko.flows.NSF(
                in_dim, context_dim, transforms=num_transforms,
                hidden_features=hidden_dims, randperm=True
            )
        else:
            self.lin_proj_layers = nn.ModuleList()
            for i in range(len(projection_dims)):
                in_proj_dim = context_dim if i == 0 else projection_dims[i - 1]
                out_proj_dim = projection_dims[i]
                self.lin_proj_layers.append(nn.Linear(in_proj_dim, out_proj_dim))
                self.lin_proj_layers.append(nn.ReLU())
                self.lin_proj_layers.append(nn.BatchNorm1d(out_proj_dim))
                self.lin_proj_layers.append(nn.Dropout(dropout))
            self.lin_proj_layers = nn.Sequential(*self.lin_proj_layers)
            self.flow = zuko.flows.NSF(
                in_dim, projection_dims[-1], transforms=num_transforms,
                hidden_features=hidden_dims, randperm=True
            )

    def forward(self, context):
        # context = self.dropout(context)
        embed_context = self.lin_proj_layers(context)
        return embed_context

    def training_step(self, batch, batch_idx):
        target, context = batch
        embed_context = self.forward(context)
        loss = -self.flow(embed_context).log_prob(target).mean()
        self.train_losses.append(loss.item())
        self.train_steps.append(self.global_step)
        self.log('train_loss', loss, on_step=True, on_epoch=True, batch_size=target.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        target, context = batch
        embed_context = self.forward(context)
        loss = -self.flow(embed_context).log_prob(target).mean()
        self.val_losses.append(loss.item())
        self.val_steps.append(self.global_step)
        self.log('val_loss', loss, on_step=True, on_epoch=True, batch_size=target.size(0))
        return loss

    def configure_optimizers(self):
        return train_utils.configure_optimizers(
            self.parameters, self.optimizer_args, scheduler_args=self.scheduler_args)

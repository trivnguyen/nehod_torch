
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import pytorch_lightning as pl

from models import noise_schedules, scores
from models import diffusion_utils, train_utils, augmentations
from models.diffusion_utils import alpha, sigma2, variance_preserving_map


class VariationalDiffusionModel(pl.LightningModule):
    def __init__(
        self,
        d_in,
        score_model_args,
        noise_schedule_args,
        d_context_embedding=None,
        d_cond=None,
        training_args=None,
        optimizer_args=None,
        scheduler_args=None,
        timesteps=0,
        antithetic_time_sampling=True,
        use_encdec=False,
        embed_context=False,
        norm_dict=None,
    ):

        super().__init__()
        self.d_in = d_in
        self.d_cond = d_cond
        self.d_context_embedding = d_context_embedding
        self.score_model_args = score_model_args
        self.noise_schedule_args = noise_schedule_args
        self.optimizer_args = optimizer_args
        self.scheduler_args = scheduler_args
        self.training_args = training_args
        self.timesteps = timesteps
        self.antithetic_time_sampling = antithetic_time_sampling
        self.use_encdec = use_encdec
        self.embed_context = embed_context
        self.norm_dict = norm_dict
        self.save_hyperparameters()

        self._setup_model()

    def _setup_model(self):

        # create the noise schedule
        if self.noise_schedule_args.name == 'fixed_linear':
            self.gamma = noise_schedules.NoiseScheduleFixedLinear(
                gamma_min=self.noise_schedule_args.gamma_min,
                gamma_max=self.noise_schedule_args.gamma_max,
            )
        elif self.noise_schedule_args.name == 'learned_linear':
            self.gamma = noise_schedules.NoiseScheduleScalar(
                gamma_min=self.noise_schedule_args.gamma_min,
                gamma_max=self.noise_schedule_args.gamma_max,
            )
        elif self.noise_schedule_args.name == "learned_net":
            self.gamma = noise_schedules.NoiseScheduleNet(
                gamma_min=self.noise_schedule_args.gamma_min,
                gamma_max=self.noise_schedule_args.gamma_max,
                scale_non_linear_init=self.noise_schedule_args.scale_non_linear_init,
            )
        else:
            raise NotImplementedError(f"Unknown noise schedule {self.noise_schedule_args.name}")

        # create the score model
        if self.score_model_args.name == 'transformer':
            self.score_model = scores.TransformerScoreModel(
                self.d_in,
                d_t_embedding=self.score_model_args.d_t_embedding,
                d_cond=self.score_model_args.d_cond,
                score_dict=self.score_model_args,
            )
        else:
            raise NotImplementedError(
                f"Unknown score model {self.score_model_args.name}")

        # create the encoder and decoder
        if self.use_encdec:
            self.encoder = None
            self.decoder = None
            raise NotImplementedError("Encoder-decoder not implemented yet")
        else:
            self.encoder = None
            self.decoder = None

        # create context embedding
        if self.embed_context:
            self.embedding_context = nn.Linear(self.d_cond, self.d_context_embedding)

    def _sample_timesteps(self, batch_size):
        if self.antithetic_time_sampling:
            t0 = torch.rand(batch_size, device=self.device)
            t_n = torch.remainder(
                t0 + torch.arange(0.0, 1.0, step=1.0 / batch_size, device=self.device), 1.0)
        else:
            t_n = torch.rand(batch_size)

        # discretize time steps if we're working with discrete time
        if self.timesteps > 0:
            t_n = torch.ceil(t_n * self.timesteps) / self.timesteps
        # t_n = t_n.to(self.device)

        return t_n

    def encode(self, x, conditioning=None, mask=None):
        """ Encode an input x into a latent distribution. """
        if self.encoder is not None:
            return self.encoder(x, conditioning, mask)
        else:
            return x

    def decode(self, z0, conditioning=None, mask=None):
        """ Decode a latent sample z0. """
        if self.use_encdec:
            return self.decoder(z0, conditioning, mask)
        else:
            return dist.Normal(loc=z0, scale=self.training_args.noise_scale)

    def embed(self, conditioning):
        """ Embed the conditioning vector, optionally including embedding a class
        assumed to be the first element of the context vector. """
        if not self.embed_context:
            return conditioning
        else:
            if conditioning is not None:
                context_embedding = self.embedding_context(conditioning)
                return context_embedding
            else:
                return None

    def recon_mass_loss(self, x, z0, mask=None):
        """ Additional term to the recon_loss that enforces mass conservation. """
        # get the mass features of each particle
        i_start = self.training_args.i_mass_start
        i_stop = self.training_args.i_mass_stop
        logm_scale = torch.tensor(
            self.norm_dict['x_std'][i_start:i_stop], dtype=x.dtype, device=x.device)
        logm_loc = torch.tensor(
            self.norm_dict['x_mean'][i_start:i_stop], dtype=x.dtype, device=x.device)
        logm_x = x[..., i_start:i_stop] * logm_scale + logm_loc
        logm_z0 = z0[..., i_start:i_stop] * logm_scale + logm_loc

        # reconstruction loss over the total mass
        # TODO: numerical stability; use logsumexp
        if mask is not None:
            logm_x = torch.log10(
                torch.sum(torch.where(~mask[..., None], 10**logm_x, 0), axis=-2))
            logm_z0 = torch.log10(
                torch.sum(torch.where(~mask[..., None], 10**logm_z0, 0), axis=-2))
        else:
            logm_x = torch.log10(torch.sum(10**logm_x, axis=-2))
            logm_z0 = torch.log10(torch.sum(10**logm_z0, axis=-2))
        logm_x_rescaled = (logm_x - logm_loc) / logm_scale
        logm_z0_rescaled = (logm_z0 - logm_loc) / logm_scale
        loss_recon_mass = -dist.Normal(
            loc=logm_z0_rescaled, scale=self.training_args.noise_scale).log_prob(logm_x_rescaled)

        return loss_recon_mass


    def recon_loss(self, x, f, cond=None, mask=None):
        """ Compute the reconstruction loss. Defined as the negative log-likelihood
        of the data under the model.

        Parameters
        ----------
        x : torch.Tensor
            The data.
        f: torch.Tensor
            The encoded features.
        """
        g_0 = self.gamma(torch.tensor([0.0], device=self.device))
        eps_0 = torch.randn_like(f)
        z_0 = variance_preserving_map(f, g_0, eps_0)
        z_0_rescaled = z_0 / alpha(g_0)
        loss_recon = -self.decode(z_0_rescaled, cond, mask).log_prob(x)

        if self.training_args.add_mass_recon_loss:
            loss_recon_mass = self.recon_mass_loss(x, z_0_rescaled, mask)
        else:
            loss_recon_mass = None
        return loss_recon, loss_recon_mass

    def latent_loss(self, f):
        """ KL divergence between posterior and prior. Prior matching term.

        Parameters
        ----------
        f : torch.Tensor
            The encoded features.
        """
        g_1 = self.gamma(torch.tensor([1.0], device=self.device))
        var_1 = sigma2(g_1)
        mean1_sqr = (1 - var_1) * torch.square(f)
        loss_klz = 0.5 * (mean1_sqr + var_1 - torch.log(var_1) - 1.0)
        return loss_klz

    def diffusion_loss(self, t, f, cond=None, mask=None, position_enc=None):
        """ Compute the diffusion loss.

        Parameters
        ----------
        t_n : torch.Tensor
            The time steps.
        f : torch.Tensor
            The encoded features.
        cond : torch.Tensor. Default is None.
            The conditioning vector.
        mask : torch.Tensor
            Attention mask. Default is None.
        position_enc : torch.Tensor
            Positional encoding. Default is None.
        """
        # Sample z_t
        g_t = self.gamma(t)
        eps = torch.randn_like(f)
        z_t = variance_preserving_map(f, g_t[:, None], eps)

        # Compute predicted noise and the MSE
        eps_hat = self.score_model(
            z_t,
            g_t,
            cond,
            mask,
            position_enc
        )
        deps = eps - eps_hat
        loss_diff_mse = torch.square(deps)  # Compute MSE of predicted noise

        T = self.timesteps
        if T == 0:
            # Loss for infinite depth T, i.e. continuous time
            g_t_grad = torch.autograd.grad(
                g_t, t,
                grad_outputs=torch.ones_like(g_t),
                create_graph=True,
                retain_graph=True,
            )[0]
            loss_diff = -0.5 * g_t_grad[:, None, None] * loss_diff_mse
        else:
            # Loss for finite depth T, i.e. discrete time
            s = t - (1.0 / T)
            g_s = self.gamma(s)
            loss_diff = 0.5 * T * torch.expm1(g_s - g_t)[:, None, None] * loss_diff_mse
        return loss_diff

    def vdm_loss(self, x, conditioning=None, mask=None, position_encoding=None):
        """ Compute the loss for a VDM model, sum of diffusion, latent,
        and reconstruction losses, appropriately masked.
        """
        loss_diff, loss_klz, loss_recon, loss_recon_mass = self.forward(
            x, conditioning, mask, position_encoding)
        beta = self.training_args.beta

        if mask is None:
            mask = torch.ones(x.shape[:-1])
        else:
            # reverse mask because of torch convention
            mask = torch.logical_not(mask).type_as(x)

        loss_batch = (
            ((loss_diff + loss_klz) * mask[:, :, None]).sum((-1, -2)) / beta +
            (loss_recon * mask[:, :, None]).sum((-1, -2))
        ) / mask.sum(-1)
        if loss_recon_mass is not None:
            loss_batch += (loss_recon_mass).sum(-1) / mask.sum(-1)
        return loss_batch.mean()

    def forward(self, x, conditioning=None, mask=None, position_encoding=None):
        batch_size = x.shape[0]

        # 1. Reconstruction loss
        f = self.encode(x, conditioning, mask)
        loss_recon, loss_recon_mass = self.recon_loss(x, f, conditioning, mask)

        # 2. Latent loss
        # KL z1 with N(0,1) prior
        loss_klz = self.latent_loss(f)

        # 3. Diffusion loss
        t = self._sample_timesteps(batch_size).requires_grad_(True)
        cond = self.embed(conditioning)
        loss_diff = self.diffusion_loss(t, f, cond, mask, position_encoding)

        return (loss_diff, loss_klz, loss_recon, loss_recon_mass)

    def prepare_batch(self, batch):
        x, conditioning, mask = batch

        # apply data augmentation
        if self.training_args.rotation_augmentation:
            x = augmentations.augment_with_symmetries(
                x, self.training_args.n_pos_dim, self.training_args.n_vel_dim,
                device=self.device)

        return {
            'x': x,
            'conditioning': conditioning,
            'mask': mask,
            'batch_size': x.shape[0],
        }

    def training_step(self, batch, batch_idx):
        batch_dict = self.prepare_batch(batch)
        loss = self.vdm_loss(
            batch_dict['x'], batch_dict['conditioning'], batch_dict['mask'])
        self.log(
            'train_loss', loss, on_step=True, on_epoch=True,
            batch_size=batch_dict['batch_size'])
        return loss

    def validation_step(self, batch, batch_idx):
        batch_dict = self.prepare_batch(batch)
        with torch.enable_grad():
            # need to enable grad for the score model
            loss = self.vdm_loss(
                batch_dict['x'], batch_dict['conditioning'], batch_dict['mask'])
        self.log(
            'val_loss', loss, on_step=True, on_epoch=True,
            batch_size=batch_dict['batch_size'])
        return loss

    def configure_optimizers(self):
        return train_utils.configure_optimizers(
            self.parameters, self.optimizer_args, self.scheduler_args)

    @torch.no_grad()
    def sample_step(self, z_t, i, T, conditioning=None, mask=None):
        """ Sample a step of the diffusion process.
        Parameters
        ----------
        z_t : torch.Tensor
            Latent state at time t, where t = (T - i) / T.
        i : int
            Current time step.
        T : int
            Total number of time steps.
        conditioning : torch.Tensor, optional
            Conditioning information.
        mask : torch.Tensor, optional
            Mask for the diffusion process.
        Returns
        -------
        z_s : torch.Tensor
            Latent state at time s, where s = t - 1.
        """
        eps = torch.randn_like(z_t)
        t = (T - i) / T
        s = (T - i - 1) / T
        g_t = self.gamma(t)
        g_s = self.gamma(s)
        cond = self.embed(conditioning)
        eps_hat_cond = self.score_model(
            z_t,
            g_t * torch.ones(z_t.shape[0], dtype=z_t.dtype, device=z_t.device),
            cond,
            mask
        )
        a = F.sigmoid(g_s)
        b = F.sigmoid(g_t)
        c = -torch.expm1(g_t - g_s)
        sigma_t = torch.sqrt(diffusion_utils.sigma2(g_t))
        z_s = (
            torch.sqrt(a / b) * (z_t - sigma_t * c * eps_hat_cond)
            + torch.sqrt((1.0 - a) * c) * eps
        )
        return z_s
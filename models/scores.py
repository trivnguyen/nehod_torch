
import copy

import numpy as np
import torch
import torch.nn as nn

from models import diffusion_utils
from models.models import MLP
from models.transformer import Transformer


class TransformerScoreModel(nn.Module):
    """ Transformer score network. """

    def __init__(
        self, d_in, d_t_embedding=32, score_dict=None, d_cond=None):
        """
        Parameters
        ----------
        d_in : int
            Input dimension
        d_t_embedd : int
            Dimension of the timestep embedding
        score_dict : dict
            Dictionary of parameters for the Transformer score model
        d_cond : int
            Dimension of the conditioning context. Must be provided if conditioning is True
        """

        super().__init__()
        self.d_in = d_in
        self.d_t_embedding = d_t_embedding
        self.d_cond = d_cond
        if score_dict is None:
            score_dict = {
                "d_model": 256,
                "d_mlp": 512,
                "n_layers": 4,
                "n_heads": 4,
                "use_pos_enc": False,
            }
        self.score_dict = copy.deepcopy(score_dict)
        self._setup()

    def _setup(self):
        # create MLP layers
        if self.d_cond is not None:
            d_cond = self.d_cond + self.d_t_embedding
        else:
            d_cond = self.d_t_embedding
        self.mlp = MLP(d_cond, [d_cond * 4, d_cond * 4, d_cond])

        # create Transformer
        score_dict = copy.deepcopy(self.score_dict).to_dict()
        score_dict["d_cond"] = d_cond
        score_dict.pop("d_t_embedding", None)
        score_dict.pop("name", None)
        self.transformer = Transformer(self.d_in, **score_dict)

    def forward(self, z, t, conditioning=None, mask=None, position_encoding=None):
        # assert np.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1
        # t = t * torch.ones(z.shape[0], device=z.device)

        # timestep embeddings
        t_embedding = diffusion_utils.get_timestep_embedding(
            t, self.d_t_embedding, device=z.device)

        # Concatenate with conditioning context
        if conditioning is not None:
            cond = torch.cat([t_embedding, conditioning], dim=1)
        else:
            cond = t_embedding

        # Pass context through a 2-layer MLP before passing into Transformer
        # I'm not sure this is really necessary
        cond = self.mlp(cond)

        h = self.transformer(z, cond, mask, position_encoding)
        return z + h

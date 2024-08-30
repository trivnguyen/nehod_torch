
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttentionBlock(nn.Module):
    """Multi-head attention. Uses pre-LN configuration (LN within residual stream),
    which seems to work much better than post-LN."""

    def __init__(self, d_in, d_model, d_mlp, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.ln_attn1 = nn.LayerNorm(d_in)
        self.ln_attn2 = nn.LayerNorm(d_in)
        self.ln_mlp = nn.LayerNorm(d_in)
        self.attention = nn.MultiheadAttention(d_in, n_heads, batch_first=True)
        self.mlp1 = nn.Linear(d_in, d_mlp)
        self.mlp2 = nn.Linear(d_mlp, d_in)
        self.activation = nn.GELU()

        # initialize weights to xavier and bias to zeros of attention
        nn.init.xavier_uniform_(self.attention.in_proj_weight)
        nn.init.zeros_(self.attention.in_proj_bias)

    def forward(self, x, y, mask=None, conditioning=None):
        # Multi-head attention
        if x is y:  # Self-attention
            x_sa = self.ln_attn1(x)
            x_sa = self.attention(
                x_sa, x_sa, x_sa, key_padding_mask=mask, need_weights=False)[0]
        else:  # Cross-attention
            x_sa = self.ln_attn1(x)
            y_sa = self.ln_attn2(y)
            x_sa = self.attention(
                x_sa, y_sa, y_sa, key_padding_mask=mask, need_weights=False)[0]

        # Add into residual stream
        x = x + x_sa

        # MLP
        x_mlp = self.ln_mlp(x)  # pre-LN
        x_mlp = self.activation(self.mlp1(x_mlp))
        x_mlp = self.mlp2(x_mlp)

        # Add into residual stream
        x = x + x_mlp
        return x


class Transformer(nn.Module):
    """Simple decoder-only transformer for set modeling.
    Attributes:
      n_input: The number of input (and output) features.
      d_model: The dimension of the model embedding space.
      d_mlp: The dimension of the multi-layer perceptron (MLP) used in the feed-forward network.
      n_layers: Number of transformer layers.
      n_heads: The number of attention heads.
      concat_conditioning: Whether to concatenate conditioning to the input.
      use_pos_enc: Whether to use positional encoding.
    """

    def __init__(
        self, d_in, d_model=128, d_mlp=512, n_layers=4, n_heads=4,
        d_pos=None, d_cond=None, concat_conditioning=False, use_pos_enc=False):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.d_pos = d_pos
        self.d_cond = d_cond
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.use_pos_enc = use_pos_enc
        self.concat_conditioning = concat_conditioning

        self._setup_model()

    def _setup_model(self):

        # create embedding layers
        self.input_embed = nn.Linear(self.d_in, self.d_model)

        if self.use_pos_enc and self.d_pos is not None:
            self.pos_encoding_layer = nn.Linear(self.d_pos, self.d_model)
        else:
            self.pos_encoding_layer = None

        if self.d_cond is not None:
            self.conditioning_embed = nn.Linear(self.d_cond, self.d_model)
            self.conditioning_concat_embed = nn.Linear(
                self.d_model + self.d_in, self.d_model)
        else:
            self.conditioning_embed = None
            self.conditioning_concat_embed = None

        # create transformer layers
        self.transformer_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.transformer_layers.append(
                MultiHeadAttentionBlock(self.d_model, self.d_model, self.d_mlp, self.n_heads)
            )

        # create final layer norm and unembedding layer
        self.final_ln = nn.LayerNorm(self.d_model)
        self.unembed = nn.Linear(self.d_model, self.d_in) # unembedding layer
        torch.nn.init.zeros_(self.unembed.weight)

    def forward(self, x, conditioning=None, mask=None, pos_enc=None):
        # Input embedding
        x = self.input_embed(x)  # (batch, seq_len, d_model)

        # Positional encoding
        if pos_enc is not None and self.use_pos_enc:
            pos_enc = self.pos_encoding_layer(pos_enc)  # (batch, seq_len, d_model)
            if mask is not None:
                pos_enc = torch.where(~mask[:, :, None], pos_enc, 0)
            x = x + pos_enc

        # Add conditioning
        if conditioning is not None:
            conditioning = self.conditioning_embed(conditioning)  # (batch, d_model)
            if self.concat_conditioning:
                conditioning = conditioning[:, None, :].repeat(1, x.size(1), 1)
                x = torch.cat([x, conditioning], dim=-1)
                x = self.conditioning_concat_embed(x)

        # Transformer layers
        for i in range(self.n_layers):
            if conditioning is not None and not self.concat_conditioning:
                x = x + conditioning[:, None, :]  # (batch, seq_len, d_model)
            x = self.transformer_layers[i](x, x, mask, conditioning)

        # Final LN as in pre-LN configuration
        x = self.final_ln(x)

        # Unembed; zero init kernel to propagate zero residual initially before training
        x = self.unembed(x)

        return x

import math

import lightning as L
import numpy as np
import torch
from torch import Tensor, nn, optim, utils

from fmukf.ML.integrators import correct_positions_with_integration
torch.set_default_dtype(torch.float32)



class ResidualBlock(nn.Module):
    """
    Creates a residual block with skip connections for neural network architectures.
    
    This module implements a residual connection pattern where the input is added to the output
    of a linear transformation stack. If input_dim != output_dim, a linear projection is used
    for the skip connection to match dimensions.
    
    Args:
        input_dim (int):
            Input dimension of the feature vector
        hidden_dim (int):
            Hidden dimension for the intermediate linear layers
        output_dim (int):
            Output dimension of the feature vector
        dropout_rate (float, optional):
            Dropout rate applied after the first linear layer. Defaults to 0.
        use_skip_connection (bool, optional):
            Whether to use skip connections. Defaults to True.
    
    Shape:
        - Input: [batch_size, ..., input_dim]
        - Output: [batch_size, ..., output_dim]
        
    Example:
        >>> block = ResidualBlock(64, 128, 64, dropout_rate=0.1)
        >>> x = torch.randn(32, 64)  # batch_size=32, input_dim=64
        >>> output = block(x)  # shape: [32, 64]
    """
    def __init__(self,
                input_dim: int,
                hidden_dim: int,
                output_dim: int,
                dropout_rate: float = 0,
                use_skip_connection: bool = True,
                ):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        if input_dim == output_dim:
            self.skip_connection: nn.Linear | nn.Identity = nn.Identity()
        else:
            self.skip_connection = nn.Linear(input_dim, output_dim, dtype=torch.float)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the residual block.
        
        Args:
            x (Tensor):
                Input tensor of shape [batch_size, ..., input_dim]
            
        Returns:
            Tensor:
                Output tensor of shape [batch_size, ..., output_dim]
        """
        identity = self.skip_connection(x)
        out = self.linear_relu_stack(x)
        out += identity
        return out


class PositionalEncoding(nn.Module):
    """
    Standard positional encoding for transformer architectures, adapted https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    
    This module adds positional information to input embeddings using sinusoidal functions.
    The positional encoding is computed using sine and cosine functions with different frequencies,
    allowing the model to learn relative positions in the sequence.
    
    Implementation based on the PyTorch transformer tutorial.
    
    Args:
        d_model (int):
            Embedding dimension (input and output size)
        dropout (float, optional):
            Dropout rate applied after positional encoding. Defaults to 0.05.
        max_len (int, optional):
            Maximum sequence length for positional encoding. Defaults to 1024.
    
    Shape:
        - Input: [batch_size, seq_len, d_model]
        - Output: [batch_size, seq_len, d_model]
        
    Note:
        Uses batch-first format for input tensors.
    """

    def __init__(self,
                 d_model: int,
                 dropout: float = 0.05,
                 max_len: int = 1024,
                 ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x (Tensor):
                Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Tensor:
                Output tensor with positional encoding added, shape [batch_size, seq_len, d_model]
        """
        x = x.transpose(0, 1)         # Correcting for batch_first
        x = x + self.pe[: x.size(0)]
        self.dropout(x)
        return x.transpose(0, 1)

    
class MyTransformerDecoderLayer(nn.Module):
    """
    A single transformer decoder layer with self-attention and feed-forward networks.
    
    This module implements a transformer decoder layer following the standard architecture:
    1. Multi-head self-attention with residual connection and layer normalization
    2. Feed-forward network with residual connection and layer normalization
    
    The layer uses causal masking to ensure autoregressive behavior (each position can only
    attend to previous positions).
    
    Args:
        embedding_dim (int):
            Dimension of input embeddings
        num_heads (int):
            Number of attention heads in multi-head attention
        dim_feedforward (int):
            Dimension of the feed-forward network
        dropout (float):
            Dropout rate applied throughout the layer
    
    Shape:
        - Input: [batch_size, seq_len, embedding_dim]
        - Output: [batch_size, seq_len, embedding_dim]
    """
    
    def __init__(self,
                embedding_dim: int,
                num_heads: int,
                dim_feedforward: int,
                dropout: float,
                ):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.multiheadSelfAttention = nn.MultiheadAttention(
            embed_dim   = embedding_dim,
            num_heads   = num_heads,
            dropout     = dropout,
            bias        = True,
            batch_first = True,
        )
        self.layernorm2 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.MLP = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=dim_feedforward, bias=True),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(in_features=dim_feedforward, out_features=embedding_dim, bias=True),
        )
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the transformer decoder layer.
        
        Args:
            x (Tensor):
                Input tensor of shape [batch_size, seq_len, embedding_dim]
            
        Returns:
            Tensor:
                Output tensor of shape [batch_size, seq_len, embedding_dim]
        """
        # Assume [batch_size, num_time_steps, embedding_dim]

        # Self attention with skip connection
        skip1 = x
        x     = self.layernorm1(x)

        attn_mask = torch.nn.Transformer.generate_square_subsequent_mask(x.size(1))
        x, scores    = self.multiheadSelfAttention.forward(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask,
            is_causal=True,
            need_weights=False,
        )
        x     = x + skip1

        # MLP with skip connection
        skip2 = x 
        x     = self.layernorm2(x)
        x     = self.MLP(x)
        x     = x + skip2
        return x


class MyDecoderOnlyTransformer(nn.Module):
    """
    A decoder-only transformer architecture for sequence modeling.
    
    This module implements a complete transformer decoder stack with input/output projections.
    It consists of:
    1. Input projection layer (embedder) using ResidualBlock
    2. Optional positional encoding
    3. Stack of transformer decoder layers
    4. Output projection layer (debedder) using ResidualBlock
    
    The model is designed for autoregressive sequence modeling tasks where the output
    at each position depends only on previous positions.
    
    Args:
        input_dim (int):
            Input dimension of the feature vector
        output_dim (int):
            Output dimension of the feature vector
        embedding_dim (int, optional):
            Hidden dimension for transformer layers. 
            If None, defaults to input_dim. Defaults to 16.
        residual_hidden_dim (int, optional):
            Hidden dimension for residual blocks.
            If None, defaults to embedding_dim. Defaults to None.
        residual_dropout (float, optional):
            Dropout rate for residual blocks. Defaults to 0.0.
        num_layers (int, optional):
            Number of transformer decoder layers. Defaults to 128.
        num_heads (int, optional):
            Number of attention heads. If None, defaults to embedding_dim//2.
            Defaults to None.
        transformer_hidden_dim (int, optional):
            Dimension of feed-forward networks in transformer layers.
            If None, defaults to embedding_dim * 8. Defaults to None.
        attention_dropout (float, optional):
            Dropout rate for attention mechanisms. Defaults to 0.01.
        use_posenc (bool, optional):
            Whether to use positional encoding. Defaults to True.
        posenc_dropout (float, optional):
            Dropout rate for positional encoding. Defaults to 0.01.
        posenc_max_len (int, optional):
            Maximum sequence length for positional encoding. Defaults to 1024.
    
    Shape:
        - Input: [batch_size, seq_len, input_dim]
        - Output: [batch_size, seq_len, output_dim]
    """
    
    def __init__(self,
                input_dim: int,
                output_dim: int,
                embedding_dim: int = 16, # Default: input_dim OR ACTUALLY SET TO 64s
                residual_hidden_dim: int | None = None, # Default: embedding_dim
                residual_dropout: float = 0.0,  
                num_layers: int = 128,   
                num_heads: int | None = None,    # Default: embedding_dim/2 (!NOTE: Check if heuristics is maybe something like embedding_dim/4)
                transformer_hidden_dim: int | None = None, # Default: embedding_dim * 8
                attention_dropout: float = 0.01,
                use_posenc: bool = True,
                posenc_dropout: float = 0.01,
                posenc_max_len: int = 1024,
                ):
        super().__init__()
        if embedding_dim is None:
            embedding_dim = input_dim
    
        if residual_hidden_dim is None:
            residual_hidden_dim = embedding_dim

        if num_heads is None:
            assert embedding_dim % 2 == 0
            num_heads = embedding_dim // 2
        
        if transformer_hidden_dim is None:
            transformer_hidden_dim = embedding_dim * 8

        # Input Projection layer (the "embedder") 
        self.embedder = ResidualBlock(input_dim, residual_hidden_dim, embedding_dim, dropout_rate=residual_dropout)
        
        # Positional Encoding
        if use_posenc:
            self.pos_encoder = PositionalEncoding(embedding_dim, dropout=posenc_dropout, max_len=posenc_max_len)
        self.use_positional_encoding = use_posenc

        # Transformer decoder layers
        self.decoder_layers = nn.Sequential(*[  
            MyTransformerDecoderLayer(
                    embedding_dim   = embedding_dim,
                    num_heads       = num_heads,
                    dim_feedforward = transformer_hidden_dim,
                    dropout         = attention_dropout)
            for _ in range(num_layers) ])
        
        # Output Projection layer (the "debedder")
        self.debedder = ResidualBlock(embedding_dim, residual_hidden_dim, output_dim, dropout_rate=residual_dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the decoder-only transformer.
        
        Args:
            x (Tensor):
                Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Tensor:
                Output tensor of shape [batch_size, seq_len, output_dim]
        """
        # Input Projection 
        x = self.embedder(x)                 # [b, l, input_dim]     -> [b, l, embedding_dim]
        
        # Positional Encoding
        if self.use_positional_encoding:
            x = self.pos_encoder(x)          # [b, l, embedding_dim] -> [b, l, embedding_dim]
        
        # Transformer Decoder only stack
        x = self.decoder_layers(x)           # [b, l, embedding_dim] -> [b, l, embedding_dim]
        
        # Output Projection 
        x = self.debedder(x)                 # [b, l, embedding_dim] -> [b, l, output_dim]
        return x





class MyTimesSeriesTransformer(L.LightningModule):
    """
    A transformer-based model for time series prediction with specialized preprocessing and training logic.
    
    This model learns the transformation from a sequence of state-control pairs to the next state in each pass:
    ((x_0, u_0), ..., (x_l, u_l)) -> ((x_0, u_0), ..., (x_l+1, u_l+1))
    
    Key Features:
    1. **State Encoding**: The 10-dimensional state vector is transformed to 11 dimensions by converting
       the heading angle psi to (sin(psi), cos(psi)) to avoid discontinuity at 0°/360° boundary and
       optional standard scaling of all features.
    2. **Input Patching**: Multiple (p) time steps are concatenated together to form a single input sequence
       to the transformer, then separated again for output.
    3. **Masked Long Horizon Prediction (MLHP)**: During training, only a random context length (L_context) is used,
       with remaining positions masked to zero, encouraging the model to learn deep transformations long-term
    4. **Normalization**: Both state and control vectors are normalized using learned statistics.
    5. **Noise Injection**: Training noise is added to improve robustness.
    
    Args:
        patch_size (int):
            Number of time steps to concatenate together as a single input
        transformer_kwargs (dict):
            Arguments passed to MyDecoderOnlyTransformer (see MyDecoderOnlyTransformer for details)
        datatype (str, optional):
            Data type for computations. Defaults to "float32".
        h (float, optional):
            Time step in seconds. Defaults to 1.0.
        normalization_params_x (dict, optional):
            Normalization parameters for state vector.
            Format: {"var_name": (mean, std), ...}. Defaults to None.
        normalization_params_u (dict, optional):
            Normalization parameters for control vector.
            Format: {"var_name": (mean, std), ...}. Defaults to None.
        noise_stds_x (dict, optional):
            Noise standard deviations for state variables during training.
            Format: {"var_name": std, ...}. Defaults to None.
        noise_stds_u (dict, optional):
            Noise standard deviations for control variables during training.
            Format: {"var_name": std, ...}. Defaults to None.
        min_std_multiplier (float, optional):
            Minimum multiplier for noise std during training. Defaults to None.
        max_std_multiplier (float, optional):
            Maximum multiplier for noise std during training. Defaults to None.
        masking_min_context (int, optional):
            Minimum context length for MLHP masking. Defaults to 64.
        masking_max_context (int, optional):
            Maximum context length for MLHP masking. Defaults to 65.
        optimizer (str, optional):
            Optimizer type. Defaults to 'Adam'.
        learning_rate (float, optional):
            Learning rate. Defaults to 1e-3.
        lr_schedule_params (dict, optional):
            Learning rate schedule parameters.
            Format: {"warmup": int, "decay": float, "step": int}. Defaults to None.
        loss_type (str, optional):
            Loss function type. Defaults to 'Huber'.

    """

    def __init__(self,
                 patch_size: int,               # Input patch size
                 transformer_kwargs: dict,
                 datatype: str = "float32",
                 h: float      = 1.0,
                 #  dtype_data_hidden: torch.float32,

                 # Normalization parameters
                 normalization_params_x: dict | None = None, # {"u": (mean, std), "v": (mean, std),...}
                 normalization_params_u: dict | None = None, # {"n": (mean, std), "delta": (mean, std)}

                 # Noise parameters during training
                 noise_stds_x: dict | None = None, # {"u": std, "v": std,...}
                 noise_stds_u: dict | None = None, # {"n": std, "delta": std)}
                 
                 min_std_multiplier: float | None = None, 
                 max_std_multiplier: float | None = None,

                 # Masking parameters:
                 masking_min_context: int = 64,    # Minimum context length (L_context)
                 masking_max_context: int = 65,    # Maximum context length

                 # Optimizer and Learning rate
                 optimizer: str = 'Adam', # 'MSE' or 'Adam'
                 learning_rate: float = 1e-3,
                 lr_schedule_params: dict | None = None, # {warmup:: 40, decay:0.774, step:60}

                 loss_type: str = 'Huber' # Function used for computing the loss via Weighted
                 ):
        super().__init__()
        self.save_hyperparameters()  # This saves patch_size, transformer_kwargs, etc.

        self.p = patch_size
        self.transformer_kwargs = transformer_kwargs
        self.h = h
        self.loss_type = loss_type if loss_type in ['Huber', 'MSE'] else 'Huber'

        # Convert data to the correct type
        self.datatype = datatype
        if datatype == "float32":
            self.dtype_ = torch.float32
        elif datatype == "float64":
            self.dtype_ = torch.float64
        else:
            raise ValueError(f"Data type {datatype} not supported")
        
        # IO Transformation parameters
        self.init_normalization(normalization_params_x, normalization_params_u)
        self.init_noise_matrices(noise_stds_x, noise_stds_u, min_std_multiplier, max_std_multiplier)

        # Masking parameters
        if masking_min_context is not None:
            assert masking_min_context < masking_max_context
        self.masking_min_context = masking_min_context
        self.masking_max_context = masking_max_context

        # Main Transformer Parameters
        if transformer_kwargs is None:
            transformer_kwargs = {}

        # Add input and output dimensions
        # dim_x = 11
        transformer_kwargs["input_dim"] = (11 + 2) * self.p # dim_x + dim_u (after encoding) * patch_size
        transformer_kwargs["output_dim"] = 11
        self.transformer = MyDecoderOnlyTransformer(**transformer_kwargs).to(self.dtype_)

        # Optimizer and Learning rate
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.lr_schedule_params = lr_schedule_params

    def init_normalization(self, normalization_params_x: dict | None, normalization_params_u: dict | None) -> None:
        """
        Initialize normalization parameters for state and control vectors.
        
        Sets up mean and standard deviation tensors for normalizing input data.
        State vector order: ["u", "v", "r", "x", "y", "p", "phi", "delta", "n", "psi_sin", "psi_cos"]
        Control vector order: ["delta", "n"]
        
        Args:
            normalization_params_x (dict | None):
                Normalization parameters for state variables.
                Format: {"var_name": (mean, std), ...}
            normalization_params_u (dict | None):
                Normalization parameters for control variables.
                Format: {"var_name": (mean, std), ...}
        """
        state_vec_order = ["u", "v", "r", "x", "y", "p", "phi", "delta", "n", "psi_sin", "psi_cos"]
        mean_x = torch.zeros(11, dtype=self.dtype_).to(self.device)
        std_x = torch.ones(11, dtype=self.dtype_).to(self.device)
        if normalization_params_x is not None:
            for var_name, (mean, std) in normalization_params_x.items():
                assert var_name in state_vec_order
                mean_x[state_vec_order.index(var_name)] = mean
                std_x[state_vec_order.index(var_name)]  = std
        self.register_buffer("mean_x", mean_x)
        self.register_buffer("std_x", std_x)

        input_vec_order = ["delta", "n"]
        mean_u = torch.zeros(2, dtype=self.dtype_).to(self.device)
        std_u = torch.ones(2, dtype=self.dtype_).to(self.device)
        if normalization_params_u is not None:
            for var_name, (mean, std) in normalization_params_u.items():
                assert var_name in input_vec_order
                mean_u[input_vec_order.index(var_name)] = mean
                std_u[input_vec_order.index(var_name)]  = std
        self.register_buffer("mean_u", mean_u)
        self.register_buffer("std_u", std_u)

    def init_noise_matrices(self,
                            noise_params_x: dict | None,
                            noise_params_u: dict | None,
                            min_std_multiplier: float | None,
                            max_std_multiplier: float | None,
                            ) -> None:
        """
        Initialize noise parameters for training data augmentation.
        
        Sets up noise standard deviations for both state and control variables during training.
        State vector order: ["u", "v", "r", "x", "y", "psi", "p", "phi", "delta", "n"]
        Control vector order: ["n", "delta"]
        
        Args:
            noise_params_x (dict | None):
                Noise std for state variables. Format: {"var_name": std, ...}
            noise_params_u (dict | None):
                Noise std for control variables. Format: {"var_name": std, ...}
            min_std_multiplier (float | None):
                Minimum multiplier for noise std. Defaults to 1.0.
            max_std_multiplier (float | None):
                Maximum multiplier for noise std. Defaults to 1.0001.
        """
        state_vec_order = ["u", "v", "r", "x", "y", "psi", "p", "phi", "delta", "n"] 
        noise_std_x = torch.zeros(10)
        if noise_params_x is not None:
            for var_name, std in noise_params_x.items():
                assert var_name in state_vec_order
                noise_std_x[state_vec_order.index(var_name)] = std
        self.register_buffer("noise_std_x", noise_std_x)

        input_vec_order = ["n", "delta"]
        noise_std_u = torch.zeros(2)
        if noise_params_u is not None:
            for var_name, std in noise_params_u.items():
                assert var_name in input_vec_order
                noise_std_u[input_vec_order.index(var_name)] = std
        self.register_buffer("noise_std_u", noise_std_u)

        # Std multipliers
        if min_std_multiplier is None: min_std_multiplier = 1.0
        if max_std_multiplier is None: max_std_multiplier = 1.0001
        self.register_buffer("min_std_multiplier", torch.tensor(min_std_multiplier, dtype=self.dtype_))
        self.register_buffer("max_std_multiplier", torch.tensor(max_std_multiplier, dtype=self.dtype_))

    def encode_x(self, x: Tensor) -> Tensor:
        """
        Encode the state vector by standard scaling and converting heading angle to sin/cos.
        
        This transformation:
        1. Converts the heading angle psi (at dimension 5) to (sin(psi), cos(psi)) effectively increasing the dimension from 10 to 11
        2. Applies standard normalization to all features
        
        New state vector order: ["u", "v", "r", "x", "y", "p", "phi", "delta", "n", "psi_sin", "psi_cos"]
       
        The sin/cos transformation is used to avoid discontinuity at the 0°/360° boundary
        and make the loss function continuous.
        
        Args:
            x (Tensor):
                Input state tensor of shape [..., 10]
            
        Returns:
            Tensor:
                Encoded state tensor of shape [..., 11]
        """
        # Encode heading angle psi (at dim=5) to sin and cos
        dim = 5 
        angle = x[..., dim] 
        angle = (angle % 360) * torch.pi / 180
        x = torch.cat([x[...,:dim], x[...,dim+1:], torch.sin(angle).unsqueeze(-1), torch.cos(angle).unsqueeze(-1)], dim=-1)
        
        # Normalize
        x = (x - self.mean_x.to(x)) / self.std_x.to(x)
        return x

    def decode_x(self, x: Tensor) -> Tensor:
        """
        Decode the state vector by unnormalizing and converting sin/cos back to heading angle.
        
        This transformation:
        1. Applies inverse normalization to all features
        2. Converts (sin(psi), cos(psi)) back to heading angle psi
        
        
        Args:
            x (Tensor):
                Encoded state tensor of shape [..., 11]
            
        Returns:
            Tensor:
                Decoded state tensor of shape [..., 10]
        """
        # Unnormalize
        x = x * self.std_x.to(x) + self.mean_x.to(x)

        dim = 5
        angle = torch.atan2(x[..., -2], x[..., -1]) * 180 / torch.pi
        angle = angle % 360
        x = torch.cat([x[..., :dim], angle.unsqueeze(-1), x[..., dim:-2]], dim=-1)
        return x
    
    def encode_u(self, u: Tensor) -> Tensor:
        """
        Normalize control input vector via standard scaling
        
        Args:
            u (Tensor):
                Input control tensor of shape [..., 2]
            
        Returns:
            Tensor:
                Normalized control tensor of shape [..., 2]
        """
        u = (u - self.mean_u.to(u)) / self.std_u.to(u)
        return u
    
    def decode_u(self, u: Tensor) -> Tensor:
        """
        Unnormalize control input vector.
        
        Args:
            u (Tensor):
                Normalized control tensor of shape [..., 2]
            
        Returns:
            Tensor:
                Unnormalized control tensor of shape [..., 2]
        """
        u = u * self.std_u.to(u) + self.mean_u.to(u)
        return u
    
    def prepare_input_target(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        """
        Prepare input and target sequences for training.
        
        This method:
        1. Applies small amount of training noise to both state and control inputs
        2. Encodes state and control vectors (normalization + heading angle transformation)
        3. Applies Masked Long Horizon Prediction (MLHP) masking to input states x
        4. Creates input sequence A and target sequence B with time shifts
        
        Args:
            batch (tuple[Tensor, Tensor]):
                Tuple of (x, u) where:
                - x: State tensor of shape  [batch_size, seq_len, 10]
                - u: Control tensor of shape [batch_size, seq_len, 2]
                
        Returns:
             (A, B) tuple[Tensor, Tensor]:
                where
                - A: Input sequence of shape [batch_size, seq_len-1, 13]
                - B: Target sequence of shape [batch_size, seq_len-1, 11]


        Note:
            The seq_len is reduced by 1 because the model is trai
        """
        # Assume x has shape (b, l, 10) where b is batch size, l is sequence length
        # Assume u has shape (b, l, 2) where b is batch size, l is sequence length
        
        x, u = batch
        b, l, d = x.shape
        x_input  = x.clone()        
        x_target = x.clone()

        # Apply training noise (gaussian with random std in range [min_std_multiplier, max_std_multiplier])
        std_multiplier = torch.distributions.uniform.Uniform(self.min_std_multiplier, self.max_std_multiplier).sample((b, 1, 1))
        x_input += torch.randn_like(x_input) * self.noise_std_x.to(x_input) * std_multiplier.to(x_input.device)
        u += torch.randn_like(u) * self.noise_std_u.to(u)
        
        # Apply encoding (ie standard scaling and converting heading angle to sin and cos)
        u = self.encode_u(u)
        x_input = self.encode_x(x_input)
        x_target = self.encode_x(x_target)

        # Apply Masked Long Horizon Prediction (MLHP) Masking
        if self.masking_min_context is not None:
            # Sample a random context length (L_context)
            L_context = torch.randint(self.masking_min_context, self.masking_max_context, (1,))
            x_input[..., L_context:, :] = 0  # Set all values to zero for k>=L_context,

        # Input and target sequences  A_k--transformer-->B_k 
        A = torch.cat([x_input, u], dim=-1)[..., :-1, :]  # <--- time-shifts! 
        B = x_target[..., 1:, :]                          # <--- time-shifts!
        return A, B

    def forward(self, A: Tensor, random_initial_masking: bool = False) -> Tensor:
        """
        Forward pass through the transformer model.
        
        Args:
            A (Tensor):
                Input sequence tensor of shape [batch_size, seq_len, input_dim]
            random_initial_masking (bool, optional):
                Whether to apply random initial masking during training.
                Defaults to False.
                
        Returns:
            Tensor:
                Output tensor of shape [batch_size, seq_len//patch_size, output_dim]
        """
        # Input Patch A, and time-indexes of model outputs
        if self.p is not None:
            A = self.patch_preprocess(A, random_initial_masking=random_initial_masking)
            assert not torch.isnan(A).any()
        A_ = self.transformer(A)
        assert not torch.isnan(A_).any()
        return A_

    def patch_preprocess(self, A: Tensor, random_initial_masking: bool) -> Tensor:
        """
        Perform zero-padding and patching of input sequence.
        
        This method:
        1. Zero-pads the sequence at the front to make length divisible by patch_size
        2. Optionally applies random initial masking to first p-1 time-steps to make the model
           generalize over different sequence lengths during training
        3. Reshapes the sequence by concatenating patch_size time steps together
        
        Args:
            A (Tensor):
                Input tensor of shape [batch_size, seq_len, input_dim]
            random_initial_masking (bool):
                Whether to apply random initial masking (should be True during training)
            
        Returns:
            Tensor:
                Patched tensor of shape [batch_size, seq_len//patch_size, input_dim*patch_size]
        """
        b, l, d = A.shape  # batch_size, seq_len, input_dim
        
        # Zero pad at the front 
        n  = int(np.ceil(l/self.p)) # Number of patches (after padding)
        l_ = n * self.p             # Seq-len after padding
        A_ = torch.zeros(b, l_, d, dtype=A.dtype).to(A)
        A_[:, -l:, :] = A                 
        assert A_.shape[1] % self.p == 0
        assert not torch.isnan(A_).any()
        # Mask initial r time values as zero, where r=0 (none),1,2...,p-1
        # This is so that the model generalizes over different sequence lengths during training
        if random_initial_masking and self.p > 1:
            r = torch.randint(low=0, high=self.p, size=(1,))
            if r != 0:
                A_[:, :r, :] = 0.

        # Patching
        return A_.view(b, l_//self.p, d*self.p)

    def get_patched_output_indices(self, A: Tensor) -> Tensor:
        """
        Get time-indices of predicted model outputs (ie the time-steps that were not effectively discarded by the masking)
        
        Returns the time indices idxs such that self.forward(A) == B[:, idxs, :]
        where B is the target sequence. (if model were perfect)
        
        Args:
            A (Tensor): Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Tensor: Time indices of shape [seq_len//patch_size]
        """
        if self.p == 1:
            return torch.arange(A.shape[1])

        l = A.shape[1] # so b,l,d = # batch_size, seq_len, output_dim
        n  = int(np.ceil(l/self.p)) # Number of patches (with zero-padding)
        l_ = n * self.p             # Number of patches (with )
        # Get time-idxs of predicted model output Bpred w.r.t to p so the model.forward(A__) == B[:,idxs,:]
        # Okay so think of this, the model output for each patch should be the last time-idx of that patch in B
        # Case = non-padded: [0...p-1],[p...2p-1],...,[(n-1)*p...n*p-1]  <-- ie i*p-1 for i=1...n = range(1,n+1)
        # Case = padded: so the first (l-l_) entries are zero padded ie to remove we need to shift left by (l-l_)
        return (l - l_) - 1 + self.p * torch.arange(1, n+1).to(A.device)

    def loss_fn(self, B: Tensor, Bpred: Tensor, axis: int | None = None) -> Tensor:
        """
        Compute weighted loss between predicted and target sequences.
        
        Uses per-feature normalization with either Huber or MSE loss.
        Dynamic weighting is applied based on trajectory differences.
        
        Args:
            B (Tensor):
                Target sequence of shape [batch_size, seq_len, output_dim]
            Bpred (Tensor):
                Predicted sequence of shape [batch_size, seq_len, output_dim]
            axis (int | None, optional):
                Axes over which to average the loss. Defaults to None.
            
        Returns:
            loss (Tensor):
                Scalar loss value
        """
        # Compute dynamic weighting based on trajectory differences
        W = 1/torch.abs(B[:, :-1, :] - B[:, 1:, :]).mean(axis=1)[:, None, :]
        
        if self.loss_type == 'Huber':
            # Apply weighting to absolute error for Huber loss
            abs_error = W * torch.abs(Bpred - B)
            huber_delta = 1.0
            loss = torch.where(abs_error <= huber_delta,
                                0.5 * abs_error**2,
                                huber_delta * (abs_error - 0.5 * huber_delta))
        elif self.loss_type == 'MSE':
            # Apply weighting to squared error for MSE loss
            weighted_error = W * (Bpred - B)
            loss = weighted_error**2
        else:
            raise ValueError(f"Loss type {self.loss_type} not supported")

        # Return the mean of the weighted loss
        loss = torch.mean(loss, axis=axis)
        return loss

    def train_eval_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        Common training and evaluation step.
        
        Args:
            batch (tuple[Tensor, Tensor]): Tuple of (x, u) tensors
            batch_idx (int): Batch index
            
        Returns:
            loss (Tensor):
                Loss value
        """
        # Sanity Check
        A, B = self.prepare_input_target(batch) # input and target sequences

        # Sanity Check
        b, l, d = A.shape
        assert l % self.p == 0, "During training (and validation_step) expect initial patching to be zero unless it goes wrong otherwise..."
        assert not torch.isnan(A).any()
        assert not torch.isnan(B).any()
        
        # Forward pass
        Bpred = self.forward(A, random_initial_masking=True)   # shape (b, l_//p, d) # where l_ = seq_len after padding
        output_time_idxs = self.get_patched_output_indices(A)
        B = B[:, output_time_idxs, :]                  # shape (b, l, d)
        
        # Compute loss
        loss = self.loss_fn(B, Bpred, axis=None)
        assert not torch.isnan(loss).any()
        return loss

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        Training step for PyTorch Lightning.
        
        Args:
            batch (tuple[Tensor, Tensor]):
                Tuple of (x, u) tensors
            batch_idx (int):
                Batch index
            
        Returns:
            loss (Tensor):
                Training loss
        """
        loss = self.train_eval_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        Validation step for PyTorch Lightning.
        
        Args:
            batch (tuple[Tensor, Tensor]):
                Tuple of (x, u) tensors
            batch_idx (int):
                Batch index
            
        Returns:
            Tensor:
                Validation loss
        """
        loss = self.train_eval_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure optimizer for PyTorch Lightning.
        
        Returns:
            torch.optim.Optimizer:
                Configured optimizer
        """
        if self.optimizer == 'SGD':
            optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Optimizer {self.optimizer} not supported")
        return optimizer
    
    def on_train_epoch_end(self) -> None:
        """
        Learning rate scheduling at the end of each training epoch.
        
        Implements stepwise exponential decay after a warmup period.
        """
        # Stepwise exponential decay
        optimizer = self.trainer.optimizers[0]
        if self.lr_schedule_params is not None:
            # Retrieve the warmup/decay/step parameters
            warmup = self.lr_schedule_params.get("warmup")
            decay = self.lr_schedule_params.get("decay")
            step = self.lr_schedule_params.get("step")

            current_epoch = self.current_epoch  # zero-based epoch index

            # After the warmup period, decay every 'step' epochs
            if current_epoch >= warmup:
                epochs_since_warmup = current_epoch - warmup
                if epochs_since_warmup % step == 0:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] *= decay

        # Log current LR value
        current_lr = optimizer.param_groups[0]["lr"]
        self.log("lr", current_lr)

    def unroll(self,
               x: np.ndarray | Tensor,   # shape (l,10) or (b,l,10)  where b is batch size
               u: np.ndarray | Tensor,   # shape (l-1+k,10) or (1,l-1+k,10)     #<--- k is the number of time-steps to unroll into the future (ie k=1 is the next time-step)
               max_context: int | None = None, # Number of previous time-steps used as context
               integrator_method: str | None = None,  # if none do no              
              ) -> np.ndarray | Tensor:
        """
        Iteratively unroll the model on sequence x until it matches the length of the control sequence u
        ie [x0, ..., x_l] 
           [u0, ..., u_l, ...,u_l+k-1]
           -- unroll k times  -->
           [x0, ..., x_l, ..., x_l+k-1]
        
        This method takes an initial state sequence and control inputs, then iteratively
        predicts future states using the transformer model. The model is unrolled k time steps
        into the future, where k is determined by the length of the control sequence.
        
        Args:
            x (np.ndarray | Tensor):
                Initial state sequence of shape (l, 10) or (batch_size, l, 10)
                ie [x0,...,x_l]
            u (np.ndarray | Tensor):
                Control sequence of shape (l-1+k, 2) or (1, l-1+k, 2)
                ie [u0,...,u_l,....,u_l+k-1]
                where k is the number of future time steps to predict
            max_context (int | None, optional):
                Maximum number of previous time steps to use as context.
                If None, uses masking_min_context. Defaults to None.
            integrator_method (str | None, optional):
                Integration method for position and heading angle correction.
                If None, no integration is applied. Defaults to None.
                
        Returns:
            np.ndarray | Tensor:
                x_l
                Predicted state sequence of same shape and type as input x
            
        Shape Transformations:
            - Input x: [batch_size, l, 10] -> [batch_size, l+k, 10]
            - Input u: [1, l-1+k, 2] -> [batch_size, l-1+k, 2] (expanded to match batch size)

        TODO: ADD EXAMPLE HERE
        """
        if max_context is None:
            max_context = self.masking_min_context
                
        # If x and u are not already torch tensors, convert them
        assert type(x) == type(u), f"Expected x and u to be of same type, but got {type(x)} and {type(u)}"
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=self.dtype_)
            u = torch.tensor(u, dtype=self.dtype_)
            return_type = "np"
        elif isinstance(x, Tensor):
            x_device = x.device
            return_type = "torch"
            if x.dtype != self.dtype_:
                print("Note converting dtypes")
                x = x.to(self.dtype_)
                u = u.to(self.dtype_)
        else:
            raise ValueError(f"Expected x and u to be of type np.ndarray or torch.Tensor, but got {type(x)} and {type(u)}")

        # Add batch dimension if not already present
        ndims = x.ndim
        if ndims == 2:
            x = x.unsqueeze(0)
        elif ndims != 3:
            raise ValueError(f"Invalid shape for x: {x.shape}, expected (l,10) or (b,l,10)")
        if u.ndim == 2:
            u = u.unsqueeze(0)
        elif u.ndim == 3:
            assert u.shape[0] == 1, f" Expected u to have shape (l-1+k,10) or (1,l-1+k,10), but got {u.shape}"
        else:
            raise ValueError(f"Invalid shape for u: {u.shape}, expected (l-1+k,10) or (1,l-1+k,10)")
        
        # Compute time-steps to unroll
        l = x.shape[1]            # Initial sequence length
        k = u.shape[1] - l + 1    # Number of time-steps to unroll into the future
        assert k > 0, f"u must have at least as many time-steps as x, but got u.shape={u.shape} and x.shape={x.shape} ie k={k}"

        # Expand u batch dimension to match x (ie duplicate u for each in batch x)
        u = u.expand(x.shape[0], -1, -1)

        # Convert to device if not already
        x = x.to(self.device)
        u = u.to(self.device)

        # Encode x and u
        x = self.encode_x(x)
        u = self.encode_u(u)

        # ACTUAL UNROLLING
        for i in range(k):
           
            l_ = x.shape[1]
            # Remove initial time-steps if they exceed max_context
            if l_ > max_context:
                l_trim = l_ - max_context
                # x_ = x[:, l_trim:, :]
                x_ = x[:, -max_context:, :]
                u_ = u[:, l_trim:l_, :]
            else:
                l_trim = None
                x_ = x
                u_ = u[:, :l_, :]
            
            # Compute Model input A and prediction
            A = torch.cat([x_, u_], dim=-1)      
            Bpred = self.forward(A, random_initial_masking=False)            # shape (b, l_//p, d) # where l_ = seq_len after padding
            x_new = Bpred[:, -1, :]            # Prediction for next/latest time-step

            if integrator_method is not None:
                # Integrate x_new
                x_prev_ = self.decode_x(x[:, -1, :].detach())
                x_new_  = self.decode_x(x_new.detach())
                x_new__ = correct_positions_with_integration(x_prev_, x_new_, h=self.h, method=integrator_method)
                x_new   = self.encode_x(x_new__)

            # Append to x
            x = torch.cat([x, x_new.unsqueeze(1)], dim=1)

        # Decode x
        x = self.decode_x(x)

        # If ndims == 2, remove batch dimension
        if ndims == 2:
            x = x.squeeze(0)
        
        # If return_type is torch, convert back to numpy
        if return_type == "np":
            x = x.detach().cpu().numpy()
        else:
            x = x.detach().to(x_device)
        return x

    def forward_with_nans(self, A: Tensor) -> Tensor:
        """
        Forward pass that returns predictions with NaN padding for non-predicted time steps.
        
        This method is useful for visualization and debugging. It returns a tensor where
        time steps that are not predicted due to input patching are filled with NaN values.
        
        Args:
            A (Tensor):
                Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Tensor:
                Output tensor of shape [batch_size, seq_len+1, output_dim] with NaN padding
        """
        self.eval()

        k = 11 # output_dim of actual transformer
        b,l,_ = A.shape # batch_size, seq_len, itorchut_dim                      [...,p-1],[p...2p-1],...,[(n-1)*p...n*p-1 = l-1]
        Bpred = self.forward(A.to(self.device), random_initial_masking=False)  # [p,2p,3p,...,l] 
        output_time_idxs  = 1 + self.get_patched_output_indices(A)             #<---- Add nan for initial time-step       
        Bout = torch.full((b,l+1,k), torch.nan).to(Bpred)   
        Bout[:, output_time_idxs, :] = Bpred
        assert not torch.isnan(Bout[:,-1,:]).any()
        return Bout

    def predict_with_nans(self,
                        x: np.ndarray | Tensor,   # shape (l,10) or (b,l,10)  where b is batch size
                        u: np.ndarray | Tensor,   # shape (l-1+k,10) or (1,l-1+k,10)     #<--- k is the number of time-steps to unroll into the future (ie k=1 is the next time-step)
                        max_context: int | None = None, # Number of previous time-steps used as context
                        ) -> np.ndarray | Tensor:
        """
        
        Predict future states with NaN padding for non-predicted time steps.
        ie [x0, ..., x_l]
           [u0, ..., u_l]
           -- unroll once  -->
           [x0, ..., x_l, x_l+1]

           where 
        
        This method is similar to unroll but returns predictions with NaN values
        for time steps that are not predicted due to patching. Useful for inference
        and visualization.
        
        Args:
            x (np.ndarray | Tensor):
                Initial state sequence of shape (l, 10) or (batch_size, l, 10)
            u (np.ndarray | Tensor):
                Control sequence of shape (l-1+k, 2) or (1, l-1+k, 2) where k > 0
            max_context (int | None, optional):
                Maximum context length. If None, uses masking_min_context.
                Defaults to None.
                
        Returns:
            np.ndarray | Tensor:
                Predicted state sequence with NaN padding

        Note:
            To illustrate why the forward_with_nans method is useful, consider that time-steps
            [ 0, ..., p-1,   p, ..., 2p-1,   2p, ...,  n*p-1]                               # l = n*p tokens
            get patched together into n = l//p tokens (ignoring padding in case l%p != 0)       
            [(0, ..., p-1), (p, ..., 2p-1), (2p, ..., 3p-1), ..., ((n-1)*p,...,n*p-1)]       # n = l//p tokens
            for which the transformer predicts states x at the following time-steps
            [            p,             2p,              3p, ...,              n*p]        # n = l//p tokens
            which makes it annoying if we want to visualize or compare the output to the input because 
            now the sequences are not aligned (+1 shift), and have fundamentally different shapes. Plotting
            with matplotlib would be annoying because now we need to supply the time stamps on the x-axis as well.
            Luckily matplotlib allows Nan values (which it simply ignores in plotting). With the forward_with_nans
            method we simply get the following ouput and nicely encapsulate the padding logic in the model:
            [nan, ..., nan, p, nan, ..., nan, 2p, nan, ..., nan, 3p, nan, ..., nan, ..., n*p]
            Ie we can easily know which time-steps the transformer has actually made a prediction, and which are patched
        """
        if max_context is None:
            max_context = self.masking_min_context
                
        # If x and u are not already torch tensors, convert them
        assert type(x) == type(u), f"Expected x and u to be of same type, but got {type(x)} and {type(u)}"
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=self.dtype_)
            u = torch.tensor(u, dtype=self.dtype_)
            return_type = "np"
        elif isinstance(x, Tensor):
            x_device = x.device
            return_type = "torch"
            if x.dtype != self.dtype_:
                print("Note converting dtypes")
                x = x.to(self.dtype_)
                u = u.to(self.dtype_)
        else:
            raise ValueError(f"Expected x and u to be of type np.ndarray or torch.Tensor, but got {type(x)} and {type(u)}")

        # If x and u are 2D, add batch dimension (ie (l,10) -> (1,l,10))
        ndims = x.ndim
        if ndims == 2:
            x = x.unsqueeze(0)
        elif ndims != 3:
            raise ValueError(f"Invalid shape for x: {x.shape}, expected (l,10) or (b,l,10)")

        if u.ndim == 2:
            u = u.unsqueeze(0)
        elif u.ndim == 3:
            assert u.shape[0] == 1, f" Expected u to have shape (l-1+k,10) or (1,l-1+k,10), but got {u.shape}"
        else:
            raise ValueError(f"Invalid shape for u: {u.shape}, expected (l-1+k,10) or (1,l-1+k,10)")

        # Compute time-steps to unroll
        l = x.shape[1]
        k = u.shape[1] - l + 1
        assert k > 0, "u must have at least as many time-steps as x"
        l_trim = max(l - max_context, 0)
        
        # Expand u batch dimension to match x (ie duplicate u for each in batch x)
        u = u.expand(x.shape[0], -1, -1)

        # Convert to device if not already
        x = x.to(self.device)
        u = u.to(self.device)

        # Encode x and u
        x = self.encode_x(x)
        u = self.encode_u(u)

        # Pad x along time-dimension with zeros to match u
        x_ = torch.cat([x, torch.zeros((x.shape[0], u.shape[1]-x.shape[1], x.shape[2]), dtype=x.dtype).to(x)], axis=1)

        A = torch.cat([x_, u], dim=-1)
        Bpred = self.forward_with_nans(A)   # shape (b, l_//p, d) # where l_ = seq_len after padding
        xpred = Bpred[0, :, :]              # shape (l, d)
        xpred = self.decode_x(xpred)      

        # If ndims == 2, remove batch dimension
        if ndims == 2:
            xpred = xpred.squeeze(0)

        # If return_type is torch, convert back to numpy
        if return_type == "np":
            xpred = xpred.detach().cpu().numpy()
        else:
            xpred = xpred.detach().to(x_device)
        return xpred
    
    @classmethod
    def load_from_comet(cls,
                ckpt_path: str,
                experiment_name: str,
                project_name: str,
                work_space: str,
                model_name: str,
                api_key: str,
                force_download: bool = False) -> 'MyTimesSeriesTransformer':
        """
        Download a model from Comet experiment and load it.
        
        This method downloads a model checkpoint from a Comet ML experiment and loads it
        using PyTorch Lightning's load_from_checkpoint method.
        
        Args:
            ckpt_path (str):
                Local path where checkpoint will be saved
            experiment_name (str | None):
                Name of the Comet experiment. Required if force_download=True
            project_name (str):
                Comet project name.
            work_space (str):
                Comet workspace name. 
            model_name (str):
                Name of the model in the experiment. 
            api_key (str):
                Comet API key.
            force_download (bool):
                If True, download even if checkpoint exists locally. Defaults to False.
            
        Returns:
            MyTimesSeriesTransformer:
                Loaded model instance
            
        Note:
            This method contains hardcoded API keys and workspace information that should
            be moved to configuration files in production.
        """

        import os
        assert ckpt_path.endswith(".ckpt"), f"Expected ckpt_path to end with .ckpt, but got {ckpt_path}"
        if os.path.exists(ckpt_path):
            if force_download:
                os.remove(ckpt_path)
            else:
                return cls.load_from_checkpoint(ckpt_path, strict=False)
        
        assert experiment_name is not None, "Expected experiment_name to be provided (since not yet downloaded)"
        import comet_ml
        import secrets
        
        temp_dir = os.path.join(os.path.dirname(ckpt_path), secrets.token_urlsafe(16))
        os.makedirs(temp_dir, exist_ok=False)
        comet_ml.login(api_key)
        api = comet_ml.API()
        experiment = api.get_experiment(work_space, project_name, experiment_name)
        assert experiment, f"Failed to load experiment {experiment_name} from project {project_name} in workspace {work_space} (may not exist?)"

        experiment.download_model(model_name, temp_dir, expand=True)
        # Grab the checkpoint file (should be only file in temp_dir and should end with .ckpt)
        temp_ckpt_files = [f for f in os.listdir(temp_dir) if f.endswith(".ckpt")]
        assert len(temp_ckpt_files) == 1, f"Expected one .ckpt file in temp_dir, but got {len(temp_ckpt_files)} files"
        temp_ckpt_path = os.path.join(temp_dir, temp_ckpt_files[0])
        
        # Move and rename the temp file to the desired ckpt_path (#Should overwrite if already exists), and delete the temp_dir
        os.rename(temp_ckpt_path, ckpt_path)
        os.rmdir(temp_dir)

        return cls.load_from_checkpoint(ckpt_path, strict=False)
    
"""
Temporal cnn based on darts implementation:
https://unit8co.github.io/darts/_modules/darts/models/forecasting/tcn_model.html#TCNModel
"""
from typing import Optional
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from portfolio_management_rl.utils.logger import get_logger
import math

logger = get_logger(__file__)


class _ResidualBlock(nn.Module):
    def __init__(
        self,
        num_filters: int,
        kernel_size: int,
        dilation_base: int,
        dropout: float,
        weight_norm: bool,
        nr_blocks_below: int,
        num_layers: int,
        input_size: int,
        target_size: int,
        activation: callable = F.silu,  # swish activation
    ):
        """PyTorch module implementing a residual block module used in `_TCNModule`.

        Parameters
        ----------
        num_filters
            The number of filters in a convolutional layer of the TCN.
        kernel_size
            The size of every kernel in a convolutional layer.
        dilation_base
            The base of the exponent that will determine the dilation on every level.
        dropout_fn
            The dropout function to be applied to every convolutional layer.
        weight_norm
            Boolean value indicating whether to use weight normalization.
        nr_blocks_below
            The number of residual blocks before the current one.
        num_layers
            The number of convolutional layers.
        input_size
            The dimensionality of the input time series of the whole network.
        target_size
            The dimensionality of the output time series of the whole network.

        Inputs
        ------
        x of shape `(batch_size, in_dimension, input_chunk_length)`
            Tensor containing the features of the input sequence.
            in_dimension is equal to `input_size` if this is the first residual block,
            in all other cases it is equal to `num_filters`.

        Outputs
        -------
        y of shape `(batch_size, out_dimension, input_chunk_length)`
            Tensor containing the output sequence of the residual block.
            out_dimension is equal to `output_size` if this is the last residual block,
            in all other cases it is equal to `num_filters`.
        """
        super().__init__()

        self.dilation_base = dilation_base
        self.kernel_size = kernel_size
        self.dropout_fn = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.nr_blocks_below = nr_blocks_below

        self.activation = activation
        input_dim = input_size if nr_blocks_below == 0 else num_filters
        output_dim = target_size if nr_blocks_below == num_layers - 1 else num_filters
        self.conv1 = nn.Conv1d(
            input_dim,
            num_filters,
            kernel_size,
            dilation=(dilation_base**nr_blocks_below),
        )
        self.conv2 = nn.Conv1d(
            num_filters,
            output_dim,
            kernel_size,
            dilation=(dilation_base**nr_blocks_below),
        )
        if weight_norm:
            self.conv1 = nn.utils.parametrizations.weight_norm(self.conv1)
            self.conv2 = nn.utils.parametrizations.weight_norm(self.conv2)

        if input_dim != output_dim:
            self.conv3 = nn.Conv1d(input_dim, output_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the residual block to the input tensor.
        """
        residual = x

        # first step
        left_padding = (self.dilation_base**self.nr_blocks_below) * (
            self.kernel_size - 1
        )
        x = F.pad(x, (left_padding, 0))
        x = self.dropout_fn(self.activation(self.conv1(x)))

        # second step
        x = F.pad(x, (left_padding, 0))
        x = self.conv2(x)
        if self.nr_blocks_below < self.num_layers - 1:
            x = self.activation(x)
        x = self.dropout_fn(x)

        # add residual
        if self.conv1.in_channels != self.conv2.out_channels:
            residual = self.conv3(residual)
        x = x + residual

        return x


class TemporalConvNet(nn.Module):
    """
    PyTorch module implementing a TCN

    """

    def __init__(
        self,
        input_size: int,
        kernel_size: int,
        num_filters: int,
        dilation_base: int,
        weight_norm: bool,
        target_size: int,
        target_length: int,
        dropout: float,
        window_size: int,
        num_layers: Optional[int] = None,
        activation: callable = F.silu,
        **kwargs,
    ):
        """PyTorch module implementing a dilated TCN module used in `TCNModel`.


        Parameters
        ----------
        input_size
            The dimensionality of the input time series.
        target_size
            The dimensionality of the output time series.
        nr_params
            The number of parameters of the likelihood (or 1 if no likelihood is used).
        target_length
            Number of time steps the torch module will predict into the future at once.
        kernel_size
            The size of every kernel in a convolutional layer.
        num_filters
            The number of filters in a convolutional layer of the TCN.
        num_layers
            The number of convolutional layers.
        weight_norm
            Boolean value indicating whether to use weight normalization.
        dilation_base
            The base of the exponent that will determine the dilation on every level.
        dropout
            The dropout rate for every convolutional layer.
        **kwargs
            all parameters required for :class:`darts.model.forecasting_models.PLForecastingModule` base class.

        Inputs
        ------
        x of shape `(batch_size, input_chunk_length, input_size)`
            Tensor containing the features of the input sequence.

        Outputs
        -------
        y of shape `(batch_size, input_chunk_length, target_size, nr_params)`
            Tensor containing the predictions of the next 'output_chunk_length' points in the last
            'output_chunk_length' entries of the tensor. The entries before contain the data points
            leading up to the first prediction, all in chronological order.
        """

        super().__init__(**kwargs)

        # Defining parameters
        self.input_size = input_size
        self.n_filters = num_filters
        self.kernel_size = kernel_size
        self.target_length = target_length
        self.target_size = target_size
        self.dilation_base = dilation_base
        self.dropout = dropout
        self.windows_size = window_size
        self.activation = activation

        # If num_layers is not passed, compute number of layers needed for full history coverage
        if num_layers is None and dilation_base > 1:
            # https://unit8.com/wp-content/uploads/2021/07/formulas_Obszar-roboczy-1-kopia-7-1-scaled.jpg
            num_layers = math.ceil(
                math.log(
                    (
                        (self.windows_size - 1)
                        * (dilation_base - 1)
                        / (2 * (kernel_size - 1))
                    )
                    + 1,
                    dilation_base,
                )
            )
            logger.info(f"Number of layers chosen: {num_layers}")
        elif num_layers is None:
            num_layers = math.ceil((self.windows_size - 1) / (kernel_size - 1) / 2)
            logger.info(f"Number of layers chosen: {num_layers}")
        self.num_layers = num_layers

        # Building TCN module
        self.encoder = nn.Sequential(
            *[
                _ResidualBlock(
                    num_filters=num_filters,
                    kernel_size=kernel_size,
                    dilation_base=dilation_base,
                    dropout=self.dropout,
                    weight_norm=weight_norm,
                    nr_blocks_below=i,
                    num_layers=num_layers,
                    input_size=self.input_size,
                    target_size=target_size,
                    activation=activation,
                )
                for i in range(num_layers)
            ]
        )

        # passes seq_len x channels to 1 x channels
        self.decoder = nn.Sequential(
            nn.Conv1d(
                self.windows_size, 1, 1
            ),  # (batch_size, 1, seq_len) pointwise convolution
            nn.Flatten(),  # (batch_size, seq_len)
            nn.Softmax(),  # (batch_size, seq_len)
        )

    def forward(self, x: Tensor):
        """
        Forward pass of the TCN module. The input is a tuple containing the input time series
        and the corresponding time indices.

        Args:
            x: Time series input of shape (batch_size, seq_len, channels)
        """

        x = x.transpose(1, 2)  # (batch_size, channels, seq_len)

        x = self.encoder(x)  # (batch_size, channels, seq_len)

        x = x.transpose(1, 2)  # (batch_size, seq_len, channels)

        x = self.decoder(x)  # (batch_size, seq_len, 1)

        return x

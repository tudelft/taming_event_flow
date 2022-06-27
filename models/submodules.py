import math

import torch
import torch.nn as nn
import torch.nn.functional as f


class ConvLayer(nn.Module):
    """
    Convolutional layer.
    Default: bias, ReLU, no downsampling, no batch norm.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        activation="relu",
        norm=None,
        BN_momentum=0.1,
        w_scale=None,
        padding=None,
        bias=None,
    ):
        super(ConvLayer, self).__init__()

        if padding is None:
            padding = kernel_size // 2
        if bias is None:
            bias = False if norm == "BN" else True
        if w_scale is None:
            w_scale = math.sqrt(1 / in_channels)

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        nn.init.uniform_(self.conv2d.weight, -w_scale, w_scale)
        if bias:
            nn.init.zeros_(self.conv2d.bias)

        if activation is not None:
            if hasattr(torch, activation):
                self.activation = getattr(torch, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.conv2d(x)

        if self.norm in ["BN", "IN"]:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class RecurrentConvLayer(nn.Module):
    """
    Layer comprised of a convolution followed by a recurrent convolutional block.
    Default: bias, ReLU, no downsampling, no batch norm, ConvGRU.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        recurrent_block_type="convgru",
        activation_ff="relu",
        activation_rec=None,
        norm=None,
        BN_momentum=0.1,
    ):
        super(RecurrentConvLayer, self).__init__()

        assert recurrent_block_type in ["convgru"]
        self.recurrent_block_type = recurrent_block_type
        if recurrent_block_type == "convgru":
            RecurrentBlock = ConvGRU
        else:
            raise NotImplementedError

        self.conv = ConvLayer(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            activation_ff,
            norm,
            BN_momentum=BN_momentum,
        )
        self.recurrent_block = RecurrentBlock(
            input_size=out_channels, hidden_size=out_channels, kernel_size=3, activation=activation_rec
        )

    def forward(self, x, prev_state):
        x = self.conv(x)
        x, state = self.recurrent_block(x, prev_state)
        return x, state


class ConvGRU(nn.Module):
    """
    Convolutional GRU cell.
    Adapted from https://github.com/jacobkimmel/pytorch_convgru/blob/master/convgru.py
    """

    def __init__(self, input_size, hidden_size, kernel_size, activation=None):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        assert activation is None, "ConvGRU activation cannot be set (just for compatibility)"

        nn.init.orthogonal_(self.reset_gate.weight)
        nn.init.orthogonal_(self.update_gate.weight)
        nn.init.orthogonal_(self.out_gate.weight)
        nn.init.constant_(self.reset_gate.bias, 0.0)
        nn.init.constant_(self.update_gate.bias, 0.0)
        nn.init.constant_(self.out_gate.bias, 0.0)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = torch.zeros(state_size, dtype=input_.dtype, device=input_.device)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state, new_state


class ResidualBlock(nn.Module):
    """
    Residual block as in "Deep residual learning for image recognition", He et al. 2016.
    Default: bias, ReLU, no downsampling, no batch norm.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        activation="relu",
        downsample=None,
        norm=None,
        BN_momentum=0.1,
    ):
        super(ResidualBlock, self).__init__()
        bias = False if norm == "BN" else True
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=bias,
        )

        if activation is not None:
            if hasattr(torch, activation):
                self.activation = getattr(torch, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == "BN":
            self.bn1 = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
            self.bn2 = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
        elif norm == "IN":
            self.bn1 = nn.InstanceNorm2d(out_channels, track_running_stats=True)
            self.bn2 = nn.InstanceNorm2d(out_channels, track_running_stats=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=bias,
        )
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out1 = self.conv1(x)
        if self.norm in ["BN", "IN"]:
            out1 = self.bn1(out1)

        if self.activation is not None:
            out1 = self.activation(out1)

        out2 = self.conv2(out1)
        if self.norm in ["BN", "IN"]:
            out2 = self.bn2(out2)

        if self.downsample:
            residual = self.downsample(x)

        out2 += residual
        if self.activation is not None:
            out2 = self.activation(out2)

        return out2, out1


class UpsampleConvLayer(nn.Module):
    """
    Upsampling layer (bilinear interpolation + Conv2d) to increase spatial resolution (x2) in a decoder.
    Default: bias, ReLU, no downsampling, no batch norm.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        activation="relu",
        norm=None,
    ):
        super(UpsampleConvLayer, self).__init__()

        bias = False if norm == "BN" else True
        padding = kernel_size // 2
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        if activation is not None:
            if hasattr(torch, activation):
                self.activation = getattr(torch, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        x_upsampled = f.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.conv2d(x_upsampled)

        if self.norm in ["BN", "IN"]:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class TransposedConvLayer(nn.Module):
    """
    Transposed convolutional layer to increase spatial resolution (x2) in a decoder.
    Default: bias, ReLU, no downsampling, no batch norm.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        activation="relu",
        norm=None,
    ):
        super(TransposedConvLayer, self).__init__()

        bias = False if norm == "BN" else True
        padding = kernel_size // 2
        self.transposed_conv2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=2,
            padding=padding,
            output_padding=1,
            bias=bias,
        )

        if activation is not None:
            if hasattr(torch, activation):
                self.activation = getattr(torch, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.transposed_conv2d(x)

        if self.norm in ["BN", "IN"]:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out

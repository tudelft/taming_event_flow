import torch

from .submodules import *


class BaseUNet(nn.Module):
    """
    Base class for conventional UNet architecture.
    Symmetric, skip connections on every encoding layer.
    """

    ff_type = ConvLayer
    res_type = ResidualBlock
    upsample_type = UpsampleConvLayer
    transpose_type = TransposedConvLayer

    def __init__(
        self,
        num_bins,
        base_channels,
        num_encoders,
        num_residual_blocks,
        num_output_channels,
        skip_type,
        norm,
        use_upsample_conv=True,
        kernel_size=3,
        encoder_stride=2,
        channel_multiplier=2,
        activations=["relu", None],
        final_activation=None,
        final_bias=True,
        final_w_scale=None,
        recurrent_block_type=None,
    ):
        super(BaseUNet, self).__init__()
        self.base_channels = base_channels
        self.num_encoders = num_encoders
        self.num_residual_blocks = num_residual_blocks
        self.num_output_channels = num_output_channels
        self.norm = norm
        self.num_bins = num_bins
        self.recurrent_block_type = recurrent_block_type
        self.kernel_size = kernel_size
        self.encoder_stride = encoder_stride
        self.channel_multiplier = channel_multiplier
        self.ff_act, self.rec_act = activations
        self.final_activation = final_activation
        self.final_bias = final_bias
        self.final_w_scale = final_w_scale

        self.skip_type = skip_type
        assert self.skip_type is None or self.skip_type in ["sum", "concat"]

        if use_upsample_conv:
            self.up_type = self.upsample_type
        else:
            self.up_type = self.transpose_type

        self.encoder_input_sizes = [
            int(self.base_channels * pow(self.channel_multiplier, i - 1)) for i in range(self.num_encoders)
        ]
        self.encoder_output_sizes = [
            int(self.base_channels * pow(self.channel_multiplier, i)) for i in range(self.num_encoders)
        ]

        self.max_num_channels = self.encoder_output_sizes[-1]

    def skip_fn(self, x, y, mode="sum"):
        assert y.shape[2:] <= x.shape[2:]
        if x.shape[2:] > y.shape[2:]:
            print("Warning: skipping row/col in skip_fn() due to odd dimensions throughout the architecture.")
            x = x[:, :, : y.shape[2], : y.shape[3]]  # skip last row/col if necessary

        if mode == "sum":
            assert x.shape[1] == y.shape[1]
            x = x + y
        elif mode == "concat":
            x = torch.cat([x, y], dim=1)
        return x

    def get_axonal_delays(self):
        self.delays = 0

    def build_encoders(self):
        encoders = nn.ModuleList()
        for i, (input_size, output_size) in enumerate(zip(self.encoder_input_sizes, self.encoder_output_sizes)):
            if i == 0:
                input_size = self.num_bins
            encoders.append(
                self.ff_type(
                    input_size,
                    output_size,
                    kernel_size=self.kernel_size,
                    stride=self.encoder_stride,
                    activation=self.ff_act,
                    norm=self.norm,
                )
            )
        return encoders

    def build_recurrent_encoders(self):
        encoders = nn.ModuleList()
        for i, (input_size, output_size) in enumerate(zip(self.encoder_input_sizes, self.encoder_output_sizes)):
            if i == 0:
                input_size = self.num_bins
            encoders.append(
                self.rec_type(
                    input_size,
                    output_size,
                    kernel_size=self.kernel_size,
                    stride=self.encoder_stride,
                    recurrent_block_type=self.recurrent_block_type,
                    activation_ff=self.ff_act,
                    activation_rec=self.rec_act,
                    norm=self.norm,
                )
            )
        return encoders

    def build_resblocks(self):
        resblocks = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            resblocks.append(
                self.res_type(
                    self.max_num_channels,
                    self.max_num_channels,
                    activation=self.ff_act,
                    norm=self.norm,
                )
            )
        return resblocks

    def build_decoders(self):
        decoder_input_sizes = reversed(self.encoder_output_sizes)
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        decoders = nn.ModuleList()
        for input_size, output_size in zip(decoder_input_sizes, decoder_output_sizes):
            decoders.append(
                self.up_type(
                    input_size if self.skip_type == "sum" else 2 * input_size,
                    output_size,
                    kernel_size=self.kernel_size,
                    activation=self.ff_act,
                    norm=self.norm,
                )
            )
        return decoders

    def build_multires_prediction_decoders(self):
        decoder_input_sizes = reversed(self.encoder_output_sizes)
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        decoders = nn.ModuleList()
        for i, (input_size, output_size) in enumerate(zip(decoder_input_sizes, decoder_output_sizes)):
            input_size = 2 * input_size if self.skip_type == "concat" else input_size
            prediction_channels = 0 if i == 0 else self.num_output_channels
            decoders.append(
                self.up_type(
                    input_size + prediction_channels,
                    output_size,
                    kernel_size=self.kernel_size,
                    activation=self.ff_act,
                    norm=self.norm,
                )
            )
        return decoders

    def build_prediction_layer(self):
        return self.ff_type(
            2 * self.base_channels if self.skip_type == "concat" else self.base_channels,
            self.num_output_channels,
            kernel_size=1,
            activation=self.final_activation,
            norm=self.norm,
            w_scale=self.final_w_scale,
            bias=self.final_bias,
        )

    def build_multires_prediction_layer(self):
        preds = nn.ModuleList()
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        for output_size in decoder_output_sizes:
            preds.append(
                self.ff_type(
                    output_size,
                    self.num_output_channels,
                    1,
                    activation=self.final_activation,
                    norm=self.norm,
                    w_scale=self.final_w_scale,
                    bias=self.final_bias,
                )
            )
        return preds


class MultiResUNetRecurrent(BaseUNet):
    """
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block.
    Symmetric, skip connections on every encoding layer (concat/sum).
    Predictions at each decoding layer.
    Predictions are added as skip connection (concat) to the input of the subsequent layer.
    """

    rec_type = RecurrentConvLayer

    def __init__(self, kwargs):
        super().__init__(**kwargs)

        self.encoders = self.build_recurrent_encoders()
        self.resblocks = self.build_resblocks()
        self.decoders = self.build_multires_prediction_decoders()
        self.preds = self.build_multires_prediction_layer()
        self.num_states = self.num_encoders
        self.states = [None] * self.num_states

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: [N x num_output_channels x H x W for i in range(self.num_encoders)]
        """

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x, self.states[i] = encoder(x, self.states[i])
            blocks.append(x)

        # residual blocks
        for resblock in self.resblocks:
            x, _ = resblock(x)

        # decoder and multires predictions
        predictions = []
        for i, (decoder, pred) in enumerate(zip(self.decoders, self.preds)):
            x = self.skip_fn(x, blocks[self.num_encoders - i - 1], mode=self.skip_type)
            if i > 0:
                x = self.skip_fn(predictions[-1], x, mode="concat")
            x = decoder(x)
            predictions.append(pred(x))

        return predictions

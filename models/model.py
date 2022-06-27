from .arch import *
from .base import BaseModel
from .model_util import copy_states, ImagePadder


class RecEVFlowNet(BaseModel):
    """
    Recurrent version of the EV-FlowNet model, as described in the paper "Self-Supervised Learning of
    Event-based Optical Flow with Spiking Neural Networks", Hagenaars and Paredes-Vallés et al., NeurIPS 2021.
    """

    net_type = MultiResUNetRecurrent
    recurrent_block_type = "convgru"
    activations = ["relu", None]

    def __init__(self, kwargs, num_bins=2, key="flow", min_size=16):
        super().__init__()
        self.image_padder = ImagePadder(min_size=min_size)

        self.key = key
        arch_kwargs = {
            "num_bins": num_bins,
            "base_channels": 64,
            "num_encoders": 4,
            "num_residual_blocks": 2,
            "num_output_channels": 2,
            "skip_type": "sum",
            "norm": None,
            "use_upsample_conv": True,
            "kernel_size": 3,
            "encoder_stride": 2,
            "channel_multiplier": 2,
            "final_activation": "tanh",
            "activations": self.activations,
            "recurrent_block_type": self.recurrent_block_type,
        }
        arch_kwargs.update(kwargs)  # udpate params with config
        arch_kwargs.pop("name", None)
        self.arch = self.net_type(arch_kwargs)
        self.num_encoders = arch_kwargs["num_encoders"]

    @property
    def states(self):
        return copy_states(self.arch.states)

    @states.setter
    def states(self, states):
        self.arch.states = states

    def detach_states(self):
        detached_states = []
        for state in self.arch.states:
            if type(state) is tuple:
                tmp = []
                for hidden in state:
                    tmp.append(hidden.detach())
                detached_states.append(tuple(tmp))
            else:
                detached_states.append(state.detach())
        self.arch.states = detached_states

    def reset_states(self):
        self.arch.states = [None] * self.arch.num_states

    def forward(self, x):

        # image padding
        x = self.image_padder.pad(x).contiguous()

        # forward pass
        multires_flow = self.arch.forward(x)

        # upsample flow estimates to the original input resolution
        flow_list = []
        for i, flow in enumerate(multires_flow):
            scaling_h = x.shape[2] / flow.shape[2]
            scaling_w = x.shape[3] / flow.shape[3]
            scaling_flow = 2 ** (self.num_encoders - i - 1)
            upflow = scaling_flow * torch.nn.functional.interpolate(
                flow, scale_factor=(scaling_h, scaling_w), mode="bilinear", align_corners=False
            )
            upflow = self.image_padder.unpad(upflow)
            flow_list.append(upflow)

        return {self.key: flow_list}

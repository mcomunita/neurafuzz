import argparse
import json
import math
import os
import pathlib
import torch
import torchaudio
from typing import List, Dict, Optional
from torch import Tensor, nn
from neutone_sdk import NeutoneParameter, WaveformToWaveformBase
from neutone_sdk.utils import save_neutone_model


"""
Temporal Featurewise Linear Modulation
"""

class TFiLM(torch.nn.Module):
    def __init__(self,
                 nchannels,
                 nparams,
                 block_size):
        super(TFiLM, self).__init__()
        self.nchannels = nchannels
        self.nparams = nparams
        self.block_size = block_size
        self.num_layers = 1
        self.first_run = True
        self.hidden_state = (torch.Tensor(0), torch.Tensor(0))  # (hidden_state, cell_state)

        # to downsample input
        self.maxpool = torch.nn.MaxPool1d(kernel_size=block_size,
                                          stride=None,
                                          padding=0,
                                          dilation=1,
                                          return_indices=False,
                                          ceil_mode=False)

        self.lstm = torch.nn.LSTM(input_size=nchannels + nparams,
                                  hidden_size=nchannels,
                                  num_layers=self.num_layers,
                                  batch_first=False,
                                  bidirectional=False)

    def forward(self, x, p: Optional[Tensor] = None):
        # x = [batch, nchannels, length]
        # p = [batch, nparams]
        x_in_shape = x.shape

        # pad input if it's not multiple of tfilm block size
        if (x_in_shape[2] % self.block_size) != 0:
            padding = torch.zeros(x_in_shape[0], x_in_shape[1], self.block_size - (x_in_shape[2] % self.block_size))
            x = torch.cat((x, padding), dim=-1)

        x_shape = x.shape
        nsteps = int(x_shape[-1] / self.block_size)

        # downsample signal [batch, nchannels, nsteps]
        x_down = self.maxpool(x)

        if self.nparams > 0 and p is not None:
            p_up = p.unsqueeze(-1)
            p_up = p_up.repeat(1, 1, nsteps)  # upsample params [batch, nparams, nsteps]
            x_down = torch.cat((x_down, p_up), dim=1)  # concat along channel dim [batch, nchannels+nparams, nsteps]

        # shape for LSTM (length, batch, channels)
        x_down = x_down.permute(2, 0, 1)

        # modulation sequence
        if self.first_run: # state was reset
            x_norm, self.hidden_state = self.lstm(x_down, None)
            self.first_run = False
        else:
            x_norm, self.hidden_state = self.lstm(x_down, self.hidden_state)

        # put shape back (batch, channels, length)
        x_norm = x_norm.permute(1, 2, 0)

        # reshape input and modulation sequence into blocks
        x_in = torch.reshape(
            x, shape=(-1, self.nchannels, nsteps, self.block_size))
        x_norm = torch.reshape(
            x_norm, shape=(-1, self.nchannels, nsteps, 1))

        # multiply
        x_out = x_norm * x_in

        # return to original (padded) shape
        x_out = torch.reshape(x_out, shape=(x_shape))

        # crop to original (input) shape
        x_out = x_out[..., :x_in_shape[2]]

        return x_out

    def reset_state(self):
        self.first_run = True


# def center_crop(x: Tensor, length: int) -> Tensor:
#     if x.size(-1) != length:
#         assert x.size(-1) > length
#         start = (x.size(-1) - length) // 2
#         stop = start + length
#         x = x[..., start:stop]
#     return x


# def causal_crop(x: Tensor, length: int) -> Tensor:
#     if x.size(-1) != length:
#         assert x.size(-1) > length
#         stop = x.size(-1) - 1
#         start = stop - length
#         x = x[..., start:stop]
#     return x


# TODO(cm): optimize for TorchScript
class PaddingCached(nn.Module):
    """Cached padding for cached convolutions."""

    def __init__(self, n_ch: int, padding: int) -> None:
        super().__init__()
        self.n_ch = n_ch
        self.padding = padding
        self.register_buffer("pad_buf", torch.zeros((1, n_ch, padding)))

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 3  # (batch_size, in_ch, samples)
        if self.padding == 0:
            return x
        bs = x.size(0)
        if bs > self.pad_buf.size(0):  # Perform resizing once if batch size is not 1
            self.pad_buf = self.pad_buf.repeat(bs, 1, 1)
        x = torch.cat([self.pad_buf, x], dim=-1)  # concat input signal to the cache
        self.pad_buf = x[:, :, -self.padding:]  # discard old cache
        return x


class Conv1dCached(nn.Module):  # Conv1d with cache
    """Cached causal convolution for streaming."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int = 0,
                 dilation: int = 1,
                 bias: bool = True) -> None:
        super().__init__()
        assert padding == 0  # We include padding in the constructor to match the Conv1d constructor
        padding = (kernel_size - 1) * dilation
        self.pad = PaddingCached(in_channels, padding)
        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              (kernel_size,),
                              (stride,),
                              padding=0,
                              dilation=(dilation,),
                              bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        # n_samples = x.size(-1)
        x = self.pad(x)  # get (cached input + current input)
        x = self.conv(x)
        # x = causal_crop(x, n_samples)
        return x


"""
Gated Conv1d
"""
class GatedConv1d(torch.nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 dilation,
                 kernel_size,
                 nparams,
                 tfilm_block_size):
        super(GatedConv1d, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.dilation = dilation
        self.kernal_size = kernel_size
        self.nparams = nparams
        self.tfilm_block_size = tfilm_block_size

        # Layers: Conv1D -> Activations -> TFiLM -> Mix + Residual

        self.conv = Conv1dCached(in_channels=in_ch,
        # self.conv = nn.Conv1d(in_channels=in_ch,
                                 out_channels=out_ch * 2,
                                 kernel_size=kernel_size,
                                 stride=1,
                                 padding=0,
                                 dilation=dilation)

        self.tfilm = TFiLM(nchannels=out_ch,
                           nparams=nparams,
                           block_size=tfilm_block_size)

        self.mix = nn.Conv1d(in_channels=out_ch,
                             out_channels=out_ch,
                             kernel_size=1,
                             stride=1,
                             padding=0)

    def forward(self, x, p: Optional[Tensor] = None):
        residual = x

        # dilated conv
        y = self.conv(x)

        # gated activation
        z = torch.tanh(y[:, :self.out_ch, :]) * \
            torch.sigmoid(y[:, self.out_ch:, :])

        # zero pad on the left side, so that z is the same length as x
        z = torch.cat((torch.zeros(residual.shape[0],
                                   self.out_ch,
                                   residual.shape[2] - z.shape[2]),
                       z),
                      dim=2)

        # modulation
        z = self.tfilm(z, p)

        x = self.mix(z)
        x = x + residual

        return x, z


""" 
GCN Block
"""
class GCNBlock(torch.nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 nlayers,
                 kernel_size,
                 dilation_growth,
                 nparams,
                 tfilm_block_size):
        super(GCNBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.nlayers = nlayers
        self.kernel_size = kernel_size
        self.dilation_growth = dilation_growth
        self.nparams = nparams
        self.tfilm_block_size = tfilm_block_size

        dilations = [dilation_growth ** l for l in range(nlayers)]

        self.layers = torch.nn.ModuleList()

        for d in dilations:
            self.layers.append(GatedConv1d(in_ch=in_ch,
                                           out_ch=out_ch,
                                           dilation=d,
                                           kernel_size=kernel_size,
                                           nparams=nparams,
                                           tfilm_block_size=tfilm_block_size))
            in_ch = out_ch

    def forward(self, x, p: Optional[Tensor] = None):
        # [batch, channels, length]
        z = torch.empty([x.shape[0],
                         self.nlayers * self.out_ch,
                         x.shape[2]])

        for n, layer in enumerate(self.layers):
            x, zn = layer(x, p)
            z[:, n * self.out_ch: (n + 1) * self.out_ch, :] = zn

        return x, z


""" 
Gated Convolution Network with Temporal FiLM layers
"""
class GCNTF(torch.nn.Module):
    def __init__(self,
                 nparams=0,
                 nblocks=2,
                 nlayers=9,
                 nchannels=8,
                 kernel_size=3,
                 dilation_growth=2,
                 tfilm_block_size=128,
                 device="cpu",
                 **kwargs):
        super(GCNTF, self).__init__()
        self.nparams = nparams
        self.nblocks = nblocks
        self.nlayers = nlayers
        self.nchannels = nchannels
        self.kernel_size = kernel_size
        self.dilation_growth = dilation_growth
        self.tfilm_block_size = tfilm_block_size
        self.device = device

        self.blocks = torch.nn.ModuleList()
        for b in range(nblocks):
            self.blocks.append(GCNBlock(in_ch=1 if b == 0 else nchannels,
                                        out_ch=nchannels,
                                        nlayers=nlayers,
                                        kernel_size=kernel_size,
                                        dilation_growth=dilation_growth,
                                        nparams=nparams,
                                        tfilm_block_size=tfilm_block_size))

        # output mixing layer
        self.blocks.append(
            torch.nn.Conv1d(in_channels=nchannels * nlayers * nblocks,
                            out_channels=1,
                            kernel_size=1,
                            stride=1,
                            padding=0))

    def forward(self, x, p: Optional[Tensor] = None):
        # x.shape = [length, batch, channels]
        # x = x.permute(1, 2, 0)  # change to [batch, channels, length]
        z = torch.empty([x.shape[0], self.blocks[-1].in_channels, x.shape[2]])

        block = self.blocks[0]
        for n, b in enumerate(self.blocks[:-1]):
            block = b
            x, zn = block(x, p)
            z[:,
                n * self.nchannels * self.nlayers:
                (n + 1) * self.nchannels * self.nlayers,
            :] = zn

        # back to [length, batch, channels]
        # return self.blocks[-1](z).permute(2, 0, 1)
        return self.blocks[-1](z)

    # reset state for all TFiLM layers
    def reset_states(self):
        for layer in self.modules():
            if isinstance(layer, TFiLM):
                layer.reset_state()


"""
NEUTONE WRAPPER
"""
class ModelWrapper(WaveformToWaveformBase):
    def __init__(self, model) -> None:
        super().__init__(model)
        self.SR = int(48000)
        self.BIAS = float(1.0) # hardwire bias value the model was trained on
        self.SENS = float(1.0) # hardwire sensitivity value the model was trained on
        self.phi0 = float(0.0) # oscillator initial phase
        self.MIN_FREQ = float(20.0)    # min oscillator frequency
        self.MAX_FREQ = float(500.0) # max oscill. frequency

    def get_model_name(self) -> str:
        return "NeuraFuzz"

    def get_model_authors(self) -> List[str]:
        return ["Marco Comunità"]

    def get_model_short_description(self) -> str:
        return "Neural fuzz + Ring modulator"

    def get_model_long_description(self) -> str:
        return "Neural fuzz distortion trained on an analogue fuzz circuit I designed + DSP ring modulator to add extra oomph"

    def get_technical_description(self) -> str:
        return "This neural fuzz was designed firstly as an analogue circuit. Training data was generated from Spice circuit simulation and used to train a gated convolution network with temporal film modulation. Designed mainly for guitar"

    def get_technical_links(self) -> Dict[str, str]:
        return {
            "Paper": "https://arxiv.org/abs/2211.00497",
            "Code": "https://github.com/mcomunita/gcn-tfilm",
            "Website": "https://mcomunita.github.io/gcn-tfilm_page"
        }

    def get_citation(self) -> str:
        return "Comunit{\`a}, M. et al (2022). Modelling black-box audio effects with time-varying feature modulation. arXiv preprint arXiv:2211.00497."

    def get_tags(self) -> List[str]:
        return ["fuzz", "distortion", "overdrive"]

    def get_model_version(self) -> str:
        return "1.0.0"

    def is_experimental(self) -> bool:
        return False

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return [
            NeutoneParameter("GAIN", "gain", default_value=0.1),
            NeutoneParameter("FUZZ", "fuzz", default_value=0.5),
            NeutoneParameter("RING", "fuzz/ring mod mix", default_value=0.0),
            NeutoneParameter("FREQ", "ring modulator frequency", default_value=0.0),
        ]

    @torch.jit.export
    def is_input_mono(self) -> bool:
        return True

    @torch.jit.export
    def is_output_mono(self) -> bool:
        return True

    @torch.jit.export
    def get_native_sample_rates(self) -> List[int]:
        return [48000]

    @torch.jit.export
    def get_native_buffer_sizes(self) -> List[int]:
        return []  # Supports all buffer sizes

    @torch.jit.export
    def denormalise_param(self, 
                          norm_val: Tensor, 
                          min_val: float, 
                          max_val: float
    ) -> Tensor:
        return (norm_val * (max_val - min_val)) + min_val
    
    @torch.jit.export
    def ring_modulation(self, 
                        nsamples: int, 
                        amplitude: Tensor, 
                        frequency: Tensor, 
                        phi0: float, 
                        sample_rate: int
    ) -> Tensor:
        time = torch.arange(nsamples)/sample_rate
        phases = 2 * math.pi * frequency * time + phi0
        self.phi0 = float(phases[-1]) # save last phase for next buffer
        # mod = amplitude * torch.sin(phases)
        mod = torch.sin(phases)
        return mod.unsqueeze(0) # modulation sequence

    def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        gain, fuzz, ring, freq = params["GAIN"], params["FUZZ"], params["RING"], params["FREQ"]

        # FUZZ
        # fuzz model params order: [bias, gain, sens, fuzz]
        bias = torch.ones_like(params["GAIN"]) * self.BIAS
        sens = torch.ones_like(params["GAIN"]) * self.SENS

        params = torch.stack([bias, gain, sens, fuzz], dim=-1)
        
        x_fuzz = x.unsqueeze(0) # (1, 1, buffer_size)
        x_fuzz = self.model.forward(x_fuzz, params)
        x_fuzz = x_fuzz.squeeze(0)
        
        # RING MODULATOR (apply to clean signal - sounds better)
        buffer_size = x.shape[-1]
        freq = self.denormalise_param(freq, self.MIN_FREQ, self.MAX_FREQ)
        mod = self.ring_modulation(buffer_size, ring, freq, self.phi0, self.SR)
        x_mod = x * mod
        
        # mix fuzz and ring (gains to compensate loudness difference)
        return 0.5 * (1 - ring) * x_fuzz + 2.0 * ring * x_mod

    def reset_model(self) -> bool:
        self.model.reset_states() # reset all TFiLM layers
        return True


if __name__ == "__main__":
    # init model with hyperparams used for training
    model = GCNTF(
        nparams=4,
        nblocks=1,
        nlayers=10,
        nchannels=16,
        kernel_size=3,
        dilation_growth=2,
        tfilm_block_size=128,
        device="cpu"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="results/1-GCNTF3-fuzz__1-10-16-3-2-128__prefilt-None-bs6/model_best.json")
    parser.add_argument("-o", "--output", default="export_model")
    args = parser.parse_args()
    
    model_path = args.model
    root_dir = pathlib.Path(args.output)
    
    with open(model_path, "r") as in_f:
        model_data = json.load(in_f)

    state_dict = model.state_dict()
    for each in model_data['state_dict']:
        new_each = each
        if "conv.weight" in each:
            new_each = each.replace("conv.weight", "conv.conv.weight")
        if "conv.bias" in each:
            new_each = each.replace("conv.bias", "conv.conv.bias")
        state_dict[new_each] = torch.tensor(model_data['state_dict'][each])
    model.load_state_dict(state_dict)

    wrapper = ModelWrapper(model)

    save_neutone_model(
        wrapper, root_dir, freeze=False, dump_samples=False, submission=True
    )

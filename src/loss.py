import torch
import auraloss

class PreEmph(torch.nn.Module):
    def __init__(self, filter_cfs, low_pass=0):
        super(PreEmph, self).__init__()
        self.epsilon = 0.00001
        self.zPad = len(filter_cfs) - 1

        self.conv_filter = torch.nn.Conv1d(1, 1, 2, bias=False)
        self.conv_filter.weight.data = torch.tensor([[filter_cfs]], requires_grad=False)

        self.low_pass = low_pass
        if self.low_pass:
            self.lp_filter = torch.nn.Conv1d(1, 1, 2, bias=False)
            self.lp_filter.weight.data = torch.tensor([[[0.85, 1]]], requires_grad=False)

    def forward(self, output, target):
        # zero pad the input/target so the filtered signal is the same length
        output = torch.cat((torch.zeros(self.zPad, output.shape[1], 1), output))
        target = torch.cat((torch.zeros(self.zPad, target.shape[1], 1), target))
        # Apply pre-emph filter, permute because the dimension order is different for RNNs and Convs in pytorch...
        output = self.conv_filter(output.permute(1, 2, 0))
        target = self.conv_filter(target.permute(1, 2, 0))

        if self.low_pass:
            output = self.lp_filter(output)
            target = self.lp_filter(target)

        return output.permute(2, 0, 1), target.permute(2, 0, 1)


class LossWrapper(torch.nn.Module):
    def __init__(self, losses, pre_filt=None):
        super(LossWrapper, self).__init__()
        self.losses = losses
        self.loss_dict = {
            'ESR': auraloss.time.ESRLoss(),
            'DC': auraloss.time.DCLoss(),
            'L1': torch.nn.L1Loss(),
            'STFT': auraloss.freq.STFTLoss(),
            'MSTFT': auraloss.freq.MultiResolutionSTFTLoss()
        }
        if pre_filt:
            pre_filt = PreEmph(pre_filt)
            self.loss_dict['ESRPre'] = lambda output, target: self.loss_dict['ESR'].forward(*pre_filt(output, target))
        loss_functions = [[self.loss_dict[key], value] for key, value in losses.items()]

        self.loss_functions = tuple([items[0] for items in loss_functions])
        try:
            self.loss_factors = tuple(torch.Tensor([items[1] for items in loss_functions]))
        except IndexError:
            self.loss_factors = torch.ones(len(self.loss_functions))

    def forward(self, output, target):
        all_losses = {}
        for i, loss in enumerate(self.losses):
            # original shape: length x batch x 1
            # auraloss needs: batch x 1 x length
            loss_fcn = self.loss_functions[i]
            loss_factor = self.loss_factors[i]
            # if isinstance(loss_fcn, auraloss.freq.STFTLoss) or isinstance(loss_fcn, auraloss.freq.MultiResolutionSTFTLoss):
            #     output = torch.permute(output, (1, 2, 0))
            #     target = torch.permute(target, (1, 2, 0))
            all_losses[loss] = torch.mul(loss_fcn(output, target), loss_factor)
        return all_losses
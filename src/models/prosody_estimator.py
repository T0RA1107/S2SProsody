import torch
import torch.nn as nn
import torch.nn.functional as F
import functools


class ProsodyEstimator2D(nn.Module):

    def __init__(self, input_nc, output_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """ ProsodyEstimator2D

        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int)  -- the number of channels in output images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(ProsodyEstimator2D, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, output_nc, kernel_size=kw, stride=1, padding=padw),
            nn.AdaptiveMaxPool2d(1)
            ]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        output = self.model(input)
        return output.squeeze()


class ProsodyEstimator1D(nn.Module):

    def __init__(self, input_nc, output_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm1d):
        """ ProsodyEstimator1D

        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int)  -- the number of channels in output images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(ProsodyEstimator1D, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm1d
        else:
            use_bias = norm_layer == nn.InstanceNorm1d

        kw = 4
        padw = 1
        sequence = [nn.Conv1d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv1d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv1d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv1d(ndf * nf_mult, output_nc, kernel_size=kw, stride=1, padding=padw),
            nn.AdaptiveMaxPool1d(1)
            ]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        output = self.model(input)
        return output.squeeze()


class ProsodyDistEstimator1D(nn.Module):

    def __init__(self, input_nc, output_nc, n_dist, ndf=64, n_layers=3, norm_layer=nn.BatchNorm1d):
        """ ProsodyDistEstimator1D

        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int)  -- the number of channels in output images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(ProsodyDistEstimator1D, self).__init__()
        self.n_dist = n_dist
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm1d
        else:
            use_bias = norm_layer == nn.InstanceNorm1d

        kw = 4
        padw = 1
        sequence = [nn.Conv1d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv1d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv1d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # sequence += [
        #     nn.Conv1d(ndf * nf_mult, output_nc, kernel_size=kw, stride=1, padding=padw),
        #     nn.AdaptiveMaxPool1d(1)
        #     ]
        self.model = nn.Sequential(*sequence)
        self.task_heads = nn.ModuleList([nn.Sequential(
            nn.Conv1d(ndf * nf_mult, output_nc, kernel_size=kw, stride=1, padding=padw),
            nn.AdaptiveMaxPool1d(1)
        ) for _ in range(n_dist)])

    def forward(self, input):
        """Standard forward."""
        output = self.model(input)
        dist_preds = torch.stack([self.task_heads[i](output).squeeze() for i in range(self.n_dist)]).transpose(0, 1)
        return dist_preds


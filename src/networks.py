#  Provides different generator and discriminator network architectures. 
#
#  Copyright (c) 2020 Abdul Fatir Ansari. All rights reserved.
#  This work is licensed under the terms of the MIT license.
#  For a copy, see <https://opensource.org/licenses/MIT>.
#
import torch.nn as nn
import torch
import numpy as np

from resnet import ResidualBlock

# DCGAN-like Discriminator
class Encoder(nn.Module):
    # Source: https://github.com/OctoberChang/MMD-GAN/blob/master/base_module.py
    def __init__(self, isize, nc, k=100, ndf=64, bn=True):
        super(Encoder, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"
        main = nn.Sequential()
        main.add_module('initial-conv-{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial-relu-{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid-{0}-{1}-conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            if bn:
                main.add_module('pyramid-{0}-batchnorm'.format(out_feat),
                                nn.BatchNorm2d(out_feat))
            main.add_module('pyramid-{0}-relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        main.add_module('final-{0}-{1}-conv'.format(cndf, k),
                        nn.Conv2d(cndf, k, 4, 1, 0, bias=False))

        self.main = main

    def forward(self, input, return_layers=False):
        if not return_layers:
            output = self.main(input)
            return output
        else:
            h = [self.main[1](self.main[0](input))]
            i = 2
            while i < len(self.main) - 1:
                hi = self.main[i + 2](self.main[i + 1](self.main[i](h[-1])))
                h.append(hi)
                i += 3
            h.append(self.main[i](h[-1]))
            return h

class DCGANDiscriminator(nn.Module):
    # Image size is fixed to 32 x 32
    def __init__(self, nz=10, ndf=64, nc=3):
        super().__init__()
        self.l1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.l2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.l3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.l4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.l5 = nn.Conv2d(ndf * 8, nz, 4, 2, 1, bias=False)

    def forward(self, x):
        x = nn.LeakyReLU(0.2, inplace=True)(self.l1(x))
        x = nn.LeakyReLU(0.2, inplace=True)(self.l2(x))
        x = nn.LeakyReLU(0.2, inplace=True)(self.l3(x))
        x = nn.LeakyReLU(0.2, inplace=True)(self.l4(x))
        x = self.l5(x)
        return x

class Decoder(nn.Module):
    # Source: https://github.com/OctoberChang/MMD-GAN/blob/master/base_module.py
    def __init__(self, isize, nc, k=100, ngf=64):
        super(Decoder, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        main.add_module('initial-{0}-{1}-convt'.format(k, cngf),
                        nn.ConvTranspose2d(k, cngf, 4, 1, 0, bias=False))
        main.add_module(
            'initial-{0}-batchnorm'.format(cngf), nn.BatchNorm2d(cngf))
        main.add_module('initial-{0}-relu'.format(cngf), nn.ReLU(True))

        csize = 4
        while csize < isize // 2:
            main.add_module('pyramid-{0}-{1}-convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(cngf // 2),
                            nn.BatchNorm2d(cngf // 2))
            main.add_module('pyramid-{0}-relu'.format(cngf // 2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        main.add_module('final-{0}-{1}-convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final-{0}-tanh'.format(nc),
                        nn.Tanh())

        self.main = main

    def forward(self, input):
        output = self.main(input)
        return output

class DCGANGenerator(nn.Module):
    # Image size is fixed to 32 x 32
    def __init__(self, ngf=64, nz=10, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output

class DCGAN5Generator(nn.Module):
    def __init__(self, imsize=128, ngf=64, nz=10, nc=3):
        super().__init__()
        self.ngf = ngf
        self.nz = nz
        s1, s2, s4, s8, s16, self.s32 = conv_sizes(imsize, layers=5, stride=2)
        self.linear1 = nn.Linear(nz, ngf * 16 * self.s32 * self.s32)
        self.relu = nn.ReLU(True)
        self.bn0 = nn.BatchNorm2d(ngf * 16)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(-1, self.nz)
        x = self.linear1(x).view(-1, self.ngf * 16, self.s32, self.s32)
        x = self.relu(self.bn0(x))
        output = self.main(x)
        return output

class DCGAN5Discriminator(nn.Module):
    def __init__(self, imsize=128, nz=10, ndf=64, nc=3):
        super().__init__()
        self.nc = nc
        self.imsize = imsize
        self.ndf = ndf
        self.nz = nz
        tmp = torch.randn(2, nc, imsize, imsize)
        self.l1 = nn.Conv2d(nc, ndf, 4, 2, 0, bias=False)
        self.l2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 0, bias=False)
        self.l3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 0, bias=False)
        self.l4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 0, bias=False)
        self.l5 = nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 0, bias=False)
        with torch.no_grad():
            outshape = self.l5(self.l4(self.l3(self.l2(self.l1(tmp))))).size()
        self.linear = nn.Linear(np.prod(outshape[1:]), nz)

    def forward(self, x):
        x = nn.LeakyReLU(0.2, inplace=True)(self.l1(x))
        x = nn.LeakyReLU(0.2, inplace=True)(self.l2(x))
        x = nn.LeakyReLU(0.2, inplace=True)(self.l3(x))
        x = nn.LeakyReLU(0.2, inplace=True)(self.l4(x))
        x = nn.LeakyReLU(0.2, inplace=True)(self.l5(x))
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        output = x.view(-1, self.nz, 1, 1)
        return output

class ResNetGenerator(nn.Module):
    def __init__(self, imsize=128, ngf=64, nz=10, nc=3):
        super().__init__()
        self.ngf = ngf
        self.nz = nz
        s1, s2, s4, s8, s16, self.s32 = conv_sizes(imsize, layers=5, stride=2)
        self.linear1 = nn.Linear(nz, ngf * 16 * self.s32 * self.s32)
        self.relu = nn.ReLU(True)
        self.bn0 = nn.BatchNorm2d(ngf * 16)
        self.model = nn.Sequential(
            ResidualBlock(ngf * 16, ngf * 8, 3, resample='up'),
            ResidualBlock(ngf * 8, ngf * 4, 3, resample='up'),
            ResidualBlock(ngf * 4, ngf * 2, 3, resample='up'),
            ResidualBlock(ngf * 2, ngf * 1, 3, resample='up'),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf, nc, 4, stride=2, padding=(3-1)//2),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(-1, self.nz)
        x = self.linear1(x).view(-1, self.ngf * 16, self.s32, self.s32)
        output = self.model(x)
        return output

def grad_norm(m, norm_type=2):
    total_norm = 0.0
    for p in m.parameters():
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def conv_sizes(imsize, layers=5, stride=2):
    s = [imsize]
    for i in range(layers):
        s.append(s[-1]//stride)
    return s


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname != 'UpsampleConv':
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0)

def build_networks(gen='flexible-dcgan', disc='flexible-dcgan',
                   ngf=64,
                   ndf=64,
                   imsize=32, # Image dim imsize x imsize
                   nc=3, # Number of channels
                   k=100, # Dim. of noise input to disc
                   z=10, # Out dim. of disc
                   bn=True): # Use batch-norm in disc
    if gen == 'flexible-dcgan':
        generator = Decoder(imsize, nc, k=k, ngf=ngf)
    elif gen == 'dcgan32':
        generator = DCGANGenerator(nc=nc, nz=k, ngf=ngf)
    elif gen == 'dcgan5':
        generator = DCGAN5Generator(imsize=imsize, ngf=ngf, nz=k, nc=nc)
    elif gen == 'resnet':
        generator = ResNetGenerator(imsize=imsize, ngf=ngf, nz=k, nc=nc)
    else:
        raise ValueError('Unknown generator')

    if disc == 'flexible-dcgan':
        discriminator = Encoder(imsize, nc, k=z, ndf=ndf, bn=bn)
    elif disc == 'dcgan32':
        discriminator = DCGANDiscriminator(nz=z, ndf=ndf, nc=nc)
    elif disc == 'dcgan5':
        discriminator = DCGAN5Discriminator(imsize=imsize, nz=z, ndf=ndf, nc=nc)
    else:
        raise ValueError('Unknown discriminator')

    return generator, discriminator

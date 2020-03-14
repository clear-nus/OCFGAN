#  Provides the CFGAN-GP model.
#
#  Copyright (c) 2020 Abdul Fatir Ansari. All rights reserved.
#  This work is licensed under the terms of the MIT license.
#  For a copy, see <https://opensource.org/licenses/MIT>.
#
import torch
import ecfd
import os
from gan import GAN

class CFGANGP(GAN):
    def __init__(self,
                 dset_name,
                 imsize,
                 nc,
                 data_root='./data',
                 results_root='./results',
                 noise_dim=100,
                 dout_dim=1,
                 batch_size=64,
                 max_giters=50000,
                 lr=1e-4,
                 clip_disc=False,
                 disc_size=64,
                 gp_lambda=10.,
                 ecfd_type='gaussian_ecfd',
                 sigmas=[1.0],
                 num_freqs=8,
                 optimize_sigma=False,
                 disc_net='flexible-dcgan',
                 gen_net='flexible-dcgan'):
        """Intializer for a CFGANGP model.
        
        Arguments:
            dset_name {str} -- Name of the dataset.
            imsize {int} -- Size of the image.
            nc {int} -- Number of channels.
        
        Keyword Arguments:
            data_root {str} -- Directory where datasets are stored (default: {'./data'}).
            results_root {str} -- Directory where results will be saved (default: {'./results'}).
            noise_dim {int} -- Dimension of noise input to generator (default: {100}).
            dout_dim {int} -- Dimension of output from discriminator (default: {1}).
            batch_size {int} -- Batch size (default: {64}).
            max_giters {int} -- Maximum number of generator iterations (default: {50000}).
            lr {[type]} -- Learning rate (default: {1e-4}).
            clip_disc {bool} -- Whether to clip the parameters of discriminator in [-0.01, 0.01].
                                This should be True when gradient penalty is not used (default: {True}). 
            disc_size {int} -- Number of filters in the first Conv layer of critic. (default: {64}).
            gp_lambda {float} -- Trade-off for gradient penalty (default: {10.0}).
            ecfd_type {str} -- Weighting distribution for ECFD (default: {'gaussian_ecfd'}).
            sigmas {list} -- A list of sigmas (default: {[1.0]}).
            num_freqs {int} -- Number of random frequencies for ECFD (default: {8}).
            optimize_sigma {bool} -- Whether to optimize sigma (default: {False}).
            disc_net {str} -- Discriminator network type (default: {'flexible-dcgan'}).
            gen_net {str} -- Generator network type (default: {'flexible-dcgan'}).
        """
        GAN.__init__(self, dset_name, imsize, nc,
                     data_root=data_root, results_root=results_root,
                     noise_dim=noise_dim, dout_dim=dout_dim,
                     batch_size=batch_size, clip_disc=gp_lambda == 0.,
                     max_giters=max_giters, lr=lr, disc_size=disc_size,
                     batch_norm=False, gen_net=gen_net, disc_net=disc_net)
        self.ecfd_fn = getattr(ecfd, ecfd_type)
        self.optimize_sigma = optimize_sigma
        self.num_freqs = num_freqs
        self.reg_lambda = 16.0
        cls_name = self.__class__.__name__.lower()
        if optimize_sigma:
            cls_name = 'o' + cls_name
        self.results_root = os.path.join(results_root, dset_name, cls_name)
        if optimize_sigma:
            self.lg_sigmas = torch.zeros((1, dout_dim)).cuda()
            self.lg_sigmas.requires_grad = True
            self.d_optim = torch.optim.RMSprop(
                list(self.discriminator.parameters()) + [self.lg_sigmas], lr=lr)
            self.results_root += '_{:s}'.format(ecfd_type)
        else:
            self.sigmas = sigmas
            self.results_root += '_{:s}_{:s}'.format(
                ecfd_type, '_'.join(map(str, sigmas)))
        self.gp_lambda = gp_lambda
        self.results_root = os.path.join(self.results_root, self.gen_net)
        self.ensure_dirs()

    def _reset_grad(self):
        """Resets the gradient of discriminator and lg_sigmas 
           (if sigma is being optimized) to zero.
        """ 
        super()._reset_grad()
        if self.optimize_sigma and self.lg_sigmas.grad is not None:
            self.lg_sigmas.grad.data.zero_()

    def disc_loss(self, reals, fakes):
        """Computes the discriminator loss -ECFD + GP + OneSideErr.
        
        Arguments:
            reals {torch.Tensor} -- A batch of real images.
            fakes {torch.Tensor} -- A batch of fake images.
        
        Returns:
            torch.Tensor -- The discriminator loss.
        """        
        d_real = self.discriminator(reals)
        d_fake = self.discriminator(fakes)
        if self.optimize_sigma:
            sigmas = torch.exp(self.lg_sigmas)
        else:
            sigmas = self.sigmas
        ecfd_loss = self.ecfd_fn(
            d_real, d_fake, sigmas, num_freqs=self.num_freqs,
            optimize_sigma=self.optimize_sigma)
        if self.gp_lambda > 0.0:
            gp = self.gradient_penalty(reals, fakes)
        else:
            gp = 0.0
        reg = self.one_side_error(d_real, d_fake)
        loss = -torch.sqrt(ecfd_loss) + self.gp_lambda * gp + self.reg_lambda * reg
        return loss

    def gen_loss(self, reals, fakes):
        """Computes the generator loss ECFD - OneSideErr.
        
        Arguments:
            reals {torch.Tensor} -- A batch of real images.
            fakes {torch.Tensor} -- A batch of fake images.
        Returns:
            torch.Tensor -- The generator loss.
        """        
        d_real = self.discriminator(reals)
        d_fake = self.discriminator(fakes)
        if self.optimize_sigma:
            sigmas = torch.exp(self.lg_sigmas)
        else:
            sigmas = self.sigmas
        ecfd_loss = self.ecfd_fn(
            d_real, d_fake, sigmas, num_freqs=self.num_freqs,
            optimize_sigma=self.optimize_sigma)
        reg = self.one_side_error(d_real, d_fake)
        return torch.sqrt(ecfd_loss) - self.reg_lambda * reg

    def one_side_error(self, d_real, d_fake):
        """Computes one sided penalty.
           Adapted from: https://github.com/OctoberChang/MMD-GAN/blob/b15c98/mmd_gan.py#L57 
        
        Arguments:
            reals {torch.Tensor} -- A batch of d(real images).
            fakes {torch.Tensor} -- A batch of d(fake images).
        
        Returns:
            torch.Tensor -- The one sided penalty.
        """        
        diff = d_real.mean(0) - d_fake.mean(0)
        err = torch.relu(-diff)
        return err.mean()

    def gradient_penalty(self, reals, fakes):
        """Computes gradient penalty. 
        
        Arguments:
            reals {torch.Tensor} -- A batch of real images.
            fakes {torch.Tensor} -- A batch of fake images.
        
        Returns:
            torch.Tensor -- The gradient penalty.
        """        
        batch_size = reals.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1).cuda()
        interpolations = alpha * reals + (1 - alpha) * fakes
        interpolations.requires_grad = True
        d_interpolations = self.discriminator(interpolations)
        gradients = torch.autograd.grad(
            d_interpolations, interpolations,
            grad_outputs=torch.ones(d_interpolations.size()).cuda(),
            create_graph=True,
            retain_graph=True)[0]
        gradients = gradients.view(batch_size, -1)
        gradients_norm = torch.sum(gradients ** 2 + 1e-7, 1).sqrt()
        return torch.mean((gradients_norm - 1) ** 2)

#  Abstract class for GAN models.
#
#  Copyright (c) 2020 Abdul Fatir Ansari. All rights reserved.
#  This work is licensed under the terms of the MIT license.
#  For a copy, see <https://opensource.org/licenses/MIT>.
# 
import os
import timeit
import torch
import torch.utils.data as tdata
import torch.backends.cudnn as cudnn
import numpy as np
from abc import ABC, abstractmethod

import networks
from datasets import get_dataset
from util import im2grid


class GAN(ABC):
    def __init__(self,
                 dset_name,
                 imsize,
                 nc,
                 data_root='./data',
                 results_root='./results',
                 noise_dim=100,
                 dout_dim=1,
                 batch_size=64,
                 clip_disc=True,
                 max_giters=50000,
                 lr=1e-4,
                 disc_size=64,
                 batch_norm=True,
                 disc_net='flexible-dcgan',
                 gen_net='flexible-dcgan'):
        """Intializer for base GAN model.
        
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
            clip_disc {bool} -- Whether to clip the parameters of discriminator in [-0.01, 0.01].
                                This should be True when gradient penalty is not used (default: {True}). 
            max_giters {int} -- Maximum number of generator iterations (default: {50000}).
            lr {[type]} -- Learning rate (default: {1e-4}).
            disc_size {int} -- Number of filters in the first Conv layer of critic. (default: {64})
            batch_norm {bool} -- Whether to use batch norm in discriminator. This should be
                                 False when gradient penalty is used (default: {True}).
            disc_net {str} -- Discriminator network type. (default: {'flexible-dcgan'})
            gen_net {str} -- Generator network type. (default: {'flexible-dcgan'})
        """                         
        self.imsize = imsize
        self.nc = nc
        self.noise_dim = noise_dim
        self.dout_dim = dout_dim
        self.disc_size = disc_size
        self.batch_norm = batch_norm
        self.disc_net = disc_net
        self.gen_net = gen_net
        self._build_model()
        self.g_optim = torch.optim.RMSprop(
            self.generator.parameters(), lr=lr)
        self.d_optim = torch.optim.RMSprop(
            self.discriminator.parameters(), lr=lr)
        self.giters = 1
        self.diters = 5
        self.max_giters = max_giters
        self.data_root = data_root
        suffix = self.__class__.__name__.lower()
        suffix += '_' + str(self.disc_size) if self.disc_size != 64 else ''
        self.results_root = os.path.join(
            results_root, dset_name, suffix)
        self.clip_disc = clip_disc
        self.model_save_interval = 1000
        self.fixed_im_interval = 100
        self.fixed_noise = torch.cuda.FloatTensor(
            batch_size, self.noise_dim, 1, 1).normal_(0, 1)
        self.noise_tensor = torch.cuda.FloatTensor(batch_size, self.noise_dim, 1, 1)
        train_dataset = get_dataset(
            dset_name, data_root=self.data_root, imsize=self.imsize)
        self.train_dataloader = tdata.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        self.real_data = self.get_real_batch()

    def ensure_dirs(self):
        """Creates directories for saving results if they don't exist.
        """        
        dirs = ['checkpoints', 'samples']
        for d in dirs:
            path = os.path.join(self.results_root, d)
            print(f'[*] {d} will be saved in {path}')
            if not os.path.exists(path):
                os.makedirs(path)

    def _build_model(self):
        """Initializes the generator and discriminator networks.
        """        
        self.generator, self.discriminator = networks.build_networks(gen=self.gen_net,
                                                                    disc=self.disc_net,
                                                                    ngf=64,
                                                                    ndf=self.disc_size,
                                                                    imsize=self.imsize,
                                                                    nc=self.nc,
                                                                    k=self.noise_dim,
                                                                    z=self.dout_dim,
                                                                    bn=self.batch_norm)
        print('Generator', self.generator)
        print('Discriminator', self.discriminator)
        self.generator.apply(networks.weights_init)
        self.discriminator.apply(networks.weights_init)
        self.generator.cuda()
        self.discriminator.cuda()
        cudnn.benchmark = True

    def get_real_batch(self):
        """Infinite generator for real images.
        
        Yields:
            torch.Tensor -- A batch of real images.
        """        
        while True:
            iterator = iter(self.train_dataloader)
            i = 0
            while i < len(iterator):
                i += 1
                yield next(iterator)

    def save_checkpoint(self, g_iter):
        """Saves the current model state.
        
        Arguments:
            g_iter {int} -- Generator iteration.
        """        
        torch.save(self.generator.state_dict(),
                   os.path.join(self.results_root, 'checkpoints',
                   'netG_iter_{0}.pth'.format(g_iter)))
        torch.save(self.discriminator.state_dict(),
                   os.path.join(self.results_root, 'checkpoints',
                   'netD_iter_{0}.pth'.format(g_iter)))
        if g_iter == self.model_save_interval:
            self.model_save_interval = 10000

    def render_fixed_noise_image(self, g_iter):
        """Save the image generated by the generator for a fixed
           latent vector over training iterations.
        
        Arguments:
            g_iter {int} -- Generator iteration.
        """        
        with torch.no_grad():
            fake_data = self.generator(self.fixed_noise).cpu().numpy()
        path = os.path.join(self.results_root, 'samples',
                            'fixed_{0}.png'.format(g_iter))
        im2grid(fake_data, path, shuffle=False)
        if g_iter == self.fixed_im_interval:
            self.fixed_im_interval = 1000

    @abstractmethod
    def disc_loss(self, reals, fakes):
        """The discriminator loss.
        
        Arguments:
            reals {torch.Tensor} -- A batch of real images.
            fakes {torch.Tensor} -- A batch of fake images.
        """        
        pass

    @abstractmethod
    def gen_loss(self, reals, fakes):
        """The generator loss.
        
        Arguments:
            reals {torch.Tensor} -- A batch of real images.
            fakes {torch.Tensor} -- A batch of fake images.
        """ 
        pass

    def _reset_grad(self):
        """Resets the gradient of discriminator to zero.
        """        
        self.discriminator.zero_grad()

    def _disc_iter(self):
        """An iteration of discriminator update.
        
        Returns:
            torch.Tensor -- Discriminator loss.
        """        
        for p in self.discriminator.parameters():
            p.requires_grad = True
        if self.clip_disc:
            for p in self.discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)
        self._reset_grad()
        real_data, _ = next(self.real_data)
        real_data = real_data.cuda()
        batch_size = real_data.size(0)
        noise = self.noise_tensor.normal_(0, 1)
        with torch.no_grad():
            fake_data = self.generator(noise)
        err_disc = self.disc_loss(real_data, fake_data)
        err_disc.backward()
        self.d_optim.step()
        return err_disc.data

    def _gen_iter(self):
        """An iteration of generator update.
        
        Returns:
            torch.Tensor -- Generator loss.
        """ 
        for p in self.discriminator.parameters():
            p.requires_grad = False
        self.generator.zero_grad()
        real_data, _ = next(self.real_data)
        real_data = real_data.cuda()
        batch_size = real_data.size(0)
        noise = self.noise_tensor.normal_(0, 1)
        fake_data = self.generator(noise)
        err_gen = self.gen_loss(real_data, fake_data)
        err_gen.backward()
        self.g_optim.step()
        return err_gen.data

    def train(self):
        """The training loop. Runs for self.max_giters generator
            iterations.
        """        
        start_time = timeit.default_timer()
        g_iter = 1
        while g_iter <= self.max_giters:
            for i in range(self.diters):
                err_disc = self._disc_iter()
            for j in range(self.giters):
                err_gen = self._gen_iter()
                if g_iter % self.model_save_interval == 0:
                    self.save_checkpoint(g_iter)
                if g_iter % self.fixed_im_interval == 0:
                    self.render_fixed_noise_image(g_iter)
                time_elapsed = (timeit.default_timer() - start_time) / 60
                print('[{:d}] <{:06.2f}m> d_loss: {:.6f}, g_loss: {:.6f}'.format(
                    g_iter, time_elapsed, err_disc, err_gen))
                g_iter += 1
        self.save_checkpoint(self.max_giters)

    def generate_samples(self, num_samples=50000, batch_size=100):
        """Generates random samples from the generator and saves them as a 
           npy file.
        
        Keyword Arguments:
            num_samples {int} -- Number of samples to generate (default: {50000}).
            batch_size {int} -- Batch size (default: {100}).
        """        
        n_batches = num_samples // batch_size
        generated_images = []
        noise = torch.cuda.FloatTensor(batch_size, self.noise_dim, 1, 1)
        for b in range(n_batches):
            noise.normal_(0, 1)
            with torch.no_grad():
                fake_data = self.generator(noise).cpu().numpy()
            generated_images.append(fake_data)
        generated_images = np.vstack(generated_images)
        path = os.path.join(self.results_root, 'samples', 'generated.npy')
        np.save(path, generated_images)

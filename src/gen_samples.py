#  Generate samples from trained models using this script.
#
#  Copyright (c) 2020 Abdul Fatir Ansari. All rights reserved.
#  This work is licensed under the terms of the MIT license.
#
import argparse
import torch
import os
import random
import string
import numpy as np
from tqdm import tqdm

import networks
from util import im2grid

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt', type=str,
                        help='Generator weights checkpoint.')
    parser.add_argument('--gen', type=str, default='flexible-dcgan',
                        choices=['flexible-dcgan', 'dcgan32', 'dcgan5', 'resnet'],
                        help='Generator network type.')
    parser.add_argument('--n', '--num_samples', type=int,
                        default=50000, help='Number of samples to generate.')
    parser.add_argument('--k', '--noise_dim', type=int,
                        default=32, help='Dimension of noise input to generator.')
    parser.add_argument('--imsize', type=int,
                        default=32, help='Size of the image.')
    parser.add_argument('--png', action='store_true', help='Whether to generate a png '
                        '(overrides num_samples and generates a 8x8 grid of images).')
    parser.add_argument('--o', '--out_dir', type=str,
                        default='./', help='Output directory.')

    args = parser.parse_args()

    generator, _ = networks.build_networks(gen=args.gen, imsize=args.imsize, k=args.k)
    generator.load_state_dict(torch.load(args.ckpt))
    print('[*] Generator loaded')
    if not os.path.exists(args.o):
        os.makedirs(args.o)
    generator.cuda()
    batch_size = 64 if args.png else 256
    noise = torch.cuda.FloatTensor(batch_size, args.k, 1, 1)
    n_loops = 1 if args.png else args.n // batch_size + 1
    samples = []
    for i in tqdm(range(n_loops)):
        noise.normal_(0, 1)
        with torch.no_grad():
            images = generator(noise).detach().cpu().numpy()
        if args.png: # Generate 8x8 if png
            path = os.path.join(args.o, 'samples.png')
            im2grid(images, path, shuffle=False)
        else:
            samples.append(images)
    if not args.png: # Save images to samples.npy
        samples = np.vstack(samples)[:args.n]
        np.save('samples.npy', samples)

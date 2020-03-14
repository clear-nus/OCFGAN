#  Train CFGAN-GP models using this script.
#
#  Copyright (c) 2020 Abdul Fatir Ansari. All rights reserved.
#  This work is licensed under the terms of the MIT license.
#  For a copy, see <https://opensource.org/licenses/MIT>.
#
import argparse
import torch


def set_args(parser):
    parser.add_argument('--dataset', required=True,
                        choices=['mnist', 'cifar10', 'celeba', 'stl10', 'celeba128'],
                        help='Dataset name.')
    parser.add_argument('--model', required=True,
                        choices=['cfgangp'], help='GAN Model')
    parser.add_argument('--dataroot', default='./data', help='Path to dataset.')
    parser.add_argument('--disc', default='flexible-dcgan',
                        choices=['flexible-dcgan', 'dcgan32', 'dcgan5'],
                        help='Discriminator network type.')
    parser.add_argument('--gen', default='flexible-dcgan',
                        choices=['flexible-dcgan', 'dcgan32', 'dcgan5', 'resnet'],
                        help='Generator network type.')
    parser.add_argument('--resultsroot', default='./results',
                        help='Path where results will be saved.')
    parser.add_argument('--batch_size', type=int,
                        default=64, help='Input batch size.')
    parser.add_argument('--image_size', type=int, default=32,
                        help='The size of the input image to network.')
    parser.add_argument('--nc', type=int, default=3, help='Number of channels.')
    parser.add_argument('--dout_dim', type=int, default=1,
                        help='Output dim of discriminator.')
    parser.add_argument('--noise_dim', type=int, default=100,
                        help='Dim of noise input to generator.')
    parser.add_argument('--disc_size', default=64, type=int,
                        help='Number of filters in first Conv layer of critic.')
    parser.add_argument('--max_giters', type=int, default=50000,
                        help='Number of generator iterations to train for.')
    parser.add_argument('--lr', type=float, default=0.00005,
                        help='Learning rate, default=0.00005.')
    parser.add_argument('--gpu_device', type=int,
                        default=0, help='GPU device id.')
    # CFGAN specific arguments
    parser.add_argument('--num_freqs', type=int, default=8,
                        help='Number of random frequencies.')
    parser.add_argument('--sigmas', type=float, nargs='+',
                        help='Value of sigma for gaussian, student-t weight,\
                        0 means that sigma will be optimized.')
    parser.add_argument('--weight', default='gaussian_ecfd',
                        type=str, choices=['gaussian_ecfd', 'studentT_ecfd',
                        'laplace_ecfd', 'uniform_ecfd'], help='Weighting distribution'
                        ' for ECFD.')
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    set_args(parser)
    args = parser.parse_args()
    assert torch.cuda.is_available(), 'Error: Requires CUDA'
    torch.cuda.set_device(args.gpu_device)
    print("Using GPU device", torch.cuda.current_device())
    if args.model == 'cfgangp':
        from cfgan import CFGANGP
        gan = CFGANGP(args.dataset, args.image_size, args.nc,
                         data_root=args.dataroot,
                         results_root=args.resultsroot,
                         dout_dim=args.dout_dim,
                         noise_dim=args.noise_dim,
                         batch_size=args.batch_size,
                         max_giters=args.max_giters,
                         lr=args.lr,
                         ecfd_type=args.weight,
                         num_freqs=args.num_freqs,
                         sigmas=args.sigmas,
                         optimize_sigma=args.sigmas[0] == 0.,
                         disc_size=args.disc_size,
			             disc_net=args.disc,
                         gen_net=args.gen)

    gan.train()
    gan.generate_samples()

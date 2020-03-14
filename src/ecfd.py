#  Provides functions to compute 
#  Empirical Characteristic Function Distance (ECFD)
#  with different weighting distributions.
#
#  Copyright (c) 2020 Abdul Fatir Ansari. All rights reserved.
#  This work is licensed under the terms of the MIT license.
#  For a copy, see <https://opensource.org/licenses/MIT>.
#
import torch
import numpy as np

def gaussian_ecfd(X, Y, sigmas, num_freqs=8, optimize_sigma=False):
    """Computes ECFD with Gaussian weighting distribution.
    
    Arguments:
        X {torch.Tensor} -- Samples from distribution P of shape [B x D].
        Y {torch.Tensor} -- Samples from distribution Q of shape [B x D].
        sigmas {list} or {torch.Tensor} -- A list of floats or a torch Tensor of
                                           shape [1 x D] if optimize_sigma is True.
    
    Keyword Arguments:
        num_freqs {int} -- Number of random frequencies to use (default: {8}).
        optimize_sigma {bool} -- Whether to optimize sigma (default: {False}).
    
    Returns:
        torch.Tensor -- The ECFD.
    """    
    total_loss = 0.0
    if not optimize_sigma:
        for sigma in sigmas:
            batch_loss = _gaussian_ecfd(X, Y, sigma, num_freqs=num_freqs)
            total_loss += batch_loss
    else:
        batch_loss = _gaussian_ecfd(X, Y, sigmas, num_freqs=num_freqs)
        total_loss += batch_loss / torch.norm(sigmas, p=2)
    return total_loss


def _gaussian_ecfd(X, Y, sigma, num_freqs=8):
    wX, wY = 1.0, 1.0
    X, Y = X.view(X.size(0), -1), Y.view(Y.size(0), -1)
    batch_size, dim = X.size()
    t = torch.randn((num_freqs, dim)).cuda() * sigma
    X_reshaped = X.view((batch_size, dim))
    tX = torch.matmul(t, X_reshaped.t())
    cos_tX = (torch.cos(tX) * wX).mean(1)
    sin_tX = (torch.sin(tX) * wX).mean(1)
    Y_reshaped = Y.view((batch_size, dim))
    tY = torch.matmul(t, Y_reshaped.t())
    cos_tY = (torch.cos(tY) * wY).mean(1)
    sin_tY = (torch.sin(tY) * wY).mean(1)
    loss = (cos_tX - cos_tY) ** 2 + (sin_tX - sin_tY) ** 2
    return loss.mean()


def laplace_ecfd(X, Y, sigmas, num_freqs=8, optimize_sigma=False):
    """Computes ECFD with Laplace weighting distribution.
    
    Arguments:
        X {torch.Tensor} -- Samples from distribution P of shape [B x D].
        Y {torch.Tensor} -- Samples from distribution Q of shape [B x D].
        sigmas {list} or {torch.Tensor} -- A list of floats or a torch Tensor of
                                           shape [1 x D] if optimize_sigma is True.
    
    Keyword Arguments:
        num_freqs {int} -- Number of random frequencies to use (default: {8}).
        optimize_sigma {bool} -- Whether to optimize sigma (default: {False}).
    
    Returns:
        torch.Tensor -- The ECFD.
    """  
    total_loss = 0.0
    if not optimize_sigma:
        for sigma in sigmas:
            batch_loss = _laplace_ecfd(X, Y, sigma, num_freqs=num_freqs)
            total_loss += batch_loss
    else:
        batch_loss = _laplace_ecfd(X, Y, sigmas, num_freqs=num_freqs)
        total_loss += batch_loss / torch.norm(sigmas, p=2)
    return total_loss


def _laplace_ecfd(X, Y, sigma, num_freqs=8):
    X, Y = X.view(X.size(0), -1), Y.view(Y.size(0), -1)
    batch_size, dim = X.size()
    t = torch.cuda.FloatTensor(
        np.random.laplace(size=(num_freqs, dim))) * sigma
    X_reshaped = X.view((batch_size, dim))
    tX = torch.matmul(t, X_reshaped.t())
    cos_tX = torch.cos(tX).mean(1)
    sin_tX = torch.sin(tX).mean(1)
    Y_reshaped = Y.view((batch_size, dim))
    tY = torch.matmul(t, Y_reshaped.t())
    cos_tY = torch.cos(tY).mean(1)
    sin_tY = torch.sin(tY).mean(1)
    loss = (cos_tX - cos_tY) ** 2 + (sin_tX - sin_tY) ** 2
    return loss.mean()

def studentT_ecfd(X, Y, sigmas, num_freqs=8, optimize_sigma=False, dof=2.0):
    """Computes ECFD with Student's-t weighting distribution with dof = 2.
    
    Arguments:
        X {torch.Tensor} -- Samples from distribution P of shape [B x D].
        Y {torch.Tensor} -- Samples from distribution Q of shape [B x D].
        sigmas {list} or {torch.Tensor} -- A list of floats or a torch Tensor of
                                           shape [1 x D] if optimize_sigma is True.
    
    Keyword Arguments:
        num_freqs {int} -- Number of random frequencies to use (default: {8}).
        optimize_sigma {bool} -- Whether to optimize sigma (default: {False}).
        dof {float} -- Degrees of freedom.
    
    Returns:
        torch.Tensor -- The ECFD.
    """  
    total_loss = 0.0
    if not optimize_sigma:
        for sigma in sigmas:
            batch_loss = _studentT_ecfd(
                X, Y, sigma, num_freqs=num_freqs, dof=dof)
            total_loss += batch_loss
    else:
        batch_loss = _studentT_ecfd(X, Y, sigmas, num_freqs=num_freqs, dof=dof)
        total_loss += batch_loss / torch.norm(sigmas, p=2)
    return total_loss


def _studentT_ecfd(X, Y, sigma, num_freqs=8, dof=2.0):
    X, Y = X.view(X.size(0), -1), Y.view(Y.size(0), -1)
    batch_size, dim = X.size()
    t = torch.cuda.FloatTensor(
        np.random.standard_t(dof, (num_freqs, dim))) * sigma
    X_reshaped = X.view((batch_size, dim))
    tX = torch.matmul(t, X_reshaped.t())
    cos_tX = torch.cos(tX).mean(1)
    sin_tX = torch.sin(tX).mean(1)
    Y_reshaped = Y.view((batch_size, dim))
    tY = torch.matmul(t, Y_reshaped.t())
    cos_tY = torch.cos(tY).mean(1)
    sin_tY = torch.sin(tY).mean(1)
    loss = (cos_tX - cos_tY) ** 2 + (sin_tX - sin_tY) ** 2
    return loss.mean()


def uniform_ecfd(X, Y, sigmas, num_freqs=8, optimize_sigma=False):
    """Computes ECFD with Uniform weighting distribution [-sigma, sigma].
    
    Arguments:
        X {torch.Tensor} -- Samples from distribution P of shape [B x D].
        Y {torch.Tensor} -- Samples from distribution Q of shape [B x D].
        sigmas {list} or {torch.Tensor} -- A list of floats or a torch Tensor of
                                           shape [1 x D] if optimize_sigma is True.
    
    Keyword Arguments:
        num_freqs {int} -- Number of random frequencies to use (default: {8}).
        optimize_sigma {bool} -- Whether to optimize sigma (default: {False}).
    
    Returns:
        torch.Tensor -- The ECFD.
    """  
    total_loss = 0.0
    if not optimize_sigma:
        for sigma in sigmas:
            batch_loss = _uniform_ecfd(X, Y, sigma, num_freqs=num_freqs)
            total_loss += batch_loss
    else:
        batch_loss = _uniform_ecfd(X, Y, sigmas, num_freqs=num_freqs)
        total_loss += batch_loss / torch.norm(sigmas, p=2)
    return total_loss


def _uniform_ecfd(X, Y, sigma, num_freqs=8):
    X, Y = X.view(X.size(0), -1), Y.view(Y.size(0), -1)
    batch_size, dim = X.size()
    t = (2 * torch.rand((num_freqs, dim)).cuda() - 1.0) * sigma
    X_reshaped = X.view((batch_size, dim))
    tX = torch.matmul(t, X_reshaped.t())
    cos_tX = torch.cos(tX).mean(1)
    sin_tX = torch.sin(tX).mean(1)
    Y_reshaped = Y.view((batch_size, dim))
    tY = torch.matmul(t, Y_reshaped.t())
    cos_tY = torch.cos(tY).mean(1)
    sin_tY = torch.sin(tY).mean(1)
    loss = (cos_tX - cos_tY) ** 2 + (sin_tX - sin_tY) ** 2
    return loss.mean()

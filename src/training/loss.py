"""
Loss functions for training the models

The losses used to trained disentangled models are here. Most of the ones
analyzed in Locatello et al, 2020 are included, except DIP-VAE I and II.
Metrics for evaluation the models are also included.

Note: This might not be the best way to implement this. I have tried to keep
architectures, loss functions etc. as separated as possible. However, this
means that controlling the behaviour of some functions during training is
harder to achieve.
"""


import torch
import torch.nn as nn
from functools import partial
from torch.nn.functional import binary_cross_entropy_with_logits as logits_bce
from torch.nn.functional import mse_loss, cross_entropy
from torch.nn.modules.loss import _Loss

from .optimizer import init_optimizer
from .math import gauss2standard_kl, inv_multiquad_sum, min_mean_discrepancy, \
                  permute_dims, mmd_idxs


class AELoss(_Loss):
    """
    Base autoencoder loss
    """
    def __init__(self, reconstruction_loss='bce'):
        super().__init__(reduction='batchmean')
        if reconstruction_loss == 'bce':
            recons_loss = logits_bce
        elif reconstruction_loss == 'mse':
            recons_loss = mse_loss
        elif not callable(reconstruction_loss):
            raise ValueError('Unrecognized reconstruction'
                             'loss {}'.format(reconstruction_loss))
        else:
            recons_loss = reconstruction_loss

        self.recons_loss = recons_loss

    def forward(self, input, target):
        reconstruction, *latent_terms = input
        # target = target.flatten(start_dim=1)

        recons_loss = self.recons_loss(reconstruction, target, reduction='sum')
        recons_loss /= target.size(0)

        latent_term = self.latent_term(*latent_terms)

        return recons_loss + latent_term

    def latent_term(self):
        raise NotImplementedError()


class GaussianVAELoss(AELoss):
    """
    This class implements the Variational Autoencoder loss with Multivariate
    Gaussian latent variables. With defualt parameters it is the one described
    in "Autoencoding Variational Bayes", Kingma & Welling (2014)
    [https://arxiv.org/abs/1312.6114].

    When $\beta>1$ this is the the loss described in $\beta$-VAE: Learning
    Basic Visual Concepts with a Constrained Variational Framework",
    Higgins et al., (2017) [https://openreview.net/forum?id=Sy2fzU9gl]
    """
    def __init__(self, reconstruction_loss='bce', beta=1.0,
                 beta_schedule=None):
        super().__init__(reconstruction_loss)
        self.beta = beta
        self.beta_schedule = beta_schedule
        self.anneal = 1.0

    def latent_term(self, z_sample, z_params):
        mu, logvar = z_params

        kl_div = gauss2standard_kl(mu, logvar).sum()
        kl_div /= z_sample.size(0)
        return self.anneal * self.beta * kl_div

    def update_parameters(self, step):
        if self.beta_schedule is not None:
            steps, schedule_type, min_anneal = self.beta_schedule
            delta = 1 / steps

            if schedule_type == 'anneal':
                self.anneal = max(1.0 - step * delta, min_anneal)
            elif schedule_type == 'increase':
                self.anneal = min(min_anneal + delta * step, 1.0)


class WAEGAN(AELoss):
    """
    Class that implements the adversarial version of the Wasserstein loss
    as found in "Wasserstein Autoencoders" Tolstikhin et al., 2019
    [https://arxiv.org/pdf/1711.01558.pdf].

    This version uses a trained discriminator to distinguish prior samples
    from posterior samples. The implementation is similar to FactorVAE,
    using a feedforward classifier. This model and the autoencoder are
    trained with conjugate gradient descent.
    """
    def __init__(self, reconstruction_loss='mse', lambda1=10.0, lambda2=0.0,
                 prior_var=1.0, lmbda_schedule=None, disc_args=None,
                 optim_kwargs=None):
        super().__init__(reconstruction_loss)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.prior_var = prior_var
        self.lmbda_schedule = lmbda_schedule
        self.anneal = 1.0

        # if disc_args is None:
        #     disc_args = [('linear', [1000]), ('relu',)] * 6

        default_optim_kwargs = {'optimizer': 'adam', 'lr': 1e-3,
                                'betas': (0.5, 0.9)}
        if optim_kwargs is not None:
            default_optim_kwargs.update(optim_kwargs)

        # disc_args.append(('linear', [2]))
        # self.disc = feedforward.FeedForward(*disc_args)
        self.disc = nn.Sequential(
            nn.Linear(10, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

        self.optim = init_optimizer(params=self.disc.parameters(),
                                    **default_optim_kwargs)
        self._batch_samples = None

    @property
    def disc_device(self):
        return next(self.disc.parameters()).device

    def train(self, mode=True):
        self.disc.train(mode)
        for p in self.disc.parameters():
            p.requires_grad = mode

    def eval(self):
        self.train(False)

    def _set_device(self, input_device):
        if self.disc_device is None or (self.disc_device != input_device):
            self.disc.to(device=input_device)

    def latent_term(self, z, z_params):
        # Hack to set the device
        self._set_device(z.device)
        self.eval()

        self._batch_samples = z.detach()

        log_z_ratio = self.disc(z)
        adv_term = (log_z_ratio[:, 0] - log_z_ratio[:, 1]).mean()

        if self.lambda2 != 0.0:
            logvar_reg = self.lambda2 * z_params[1].abs().sum() / z.size(0)
        else:
            logvar_reg = 0.0

        return self.anneal * self.lambda1 * adv_term + logvar_reg

    def update_parameters(self, step):
        # update anneal value
        if self.lmbda_schedule is not None:
            steps, min_anneal = self.lmbda_schedule
            delta = 1 / steps
            self.anneal = max(1.0 - step * delta, min_anneal)

        if self._batch_samples is None:
            return

        # Train discriminator
        self.train()
        self.optim.zero_grad()

        z, self._batch_samples = self._batch_samples, None

        z_prior = self.prior_var * torch.randn_like(z)

        log_ratio_z = self.disc(z)
        log_ratio_z_prior = self.disc(z_prior)

        ones = z_prior.new_ones(z_prior.size(0), dtype=torch.long)
        zeros = torch.zeros_like(ones)

        disc_loss = 0.5 * (cross_entropy(log_ratio_z, zeros) +
                           cross_entropy(log_ratio_z_prior, ones))

        disc_loss.backward()
        self.optim.step()


class WAEMMD(AELoss):
    """
    Class that implements the Minimum Mean Discrepancy term in the latent space
    as found in "Wasserstein Autoencoders", Tolstikhin et al., (2019)
    [https://arxiv.org/pdf/1711.01558.pdf], with the modifications proposed in
    "Learning disentangled representations with Wasserstein Autoencoders"
    Rubenstein et al., 2018 [https://openreview.net/pdf?id=Hy79-UJPM].

    Unlike the adversarial version, this one relies on kernels to determine the
    distance between the distributions. While we allow any kernel, the default
    is the sum of inverse multiquadratics which has heavier tails than RBF. We
    also add an L1 penalty on the log-variance to prevent the encoders from
    becoming deterministic.
    """
    def __init__(self, reconstruction_loss='mse', lambda1=10, lambda2=1.0,
                 prior_type='norm', prior_var=1.0, kernel=None,
                 lmbda_schedule=None):

        super().__init__(reconstruction_loss)

        if kernel is None:
            kernel = partial(inv_multiquad_sum,
                             base_scale=10.0,
                             scales=torch.tensor([0.1, 0.2, 0.5, 1.0,
                                                  2.0, 5.0, 10.0]))

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.prior_type = prior_type
        self.prior_var = prior_var
        self.lmbda_schedule = lmbda_schedule
        self.kernel = kernel
        self.anneal = 1.0
        # Save the indices of the combinations for reuse
        self._idxs = None

    def latent_term(self, z, z_params):
        if self.prior_type == 'norm':
            z_prior = self.prior_var * torch.randn_like(z)
        elif self.prior_type == 'unif':
            z_prior = self.prior_var * torch.rand_like(z) - 0.5
        else:
            raise ValueError('Unrecognized prior {}'.format(self.prior_type))

        if self._idxs is None or len(self._idxs[1]) != z.size(0) ** 2:
            self._idxs = mmd_idxs(z.size(0))

        adv_term = min_mean_discrepancy(z, z_prior, self.kernel, self._idxs)

        # L1 regularization of log-variance
        if self.lambda2 != 0.0:
            logvar_reg = self.lambda2 * z_params[1].abs().sum() / z.size(0)
        else:
            logvar_reg = 0.0

        return self.anneal * self.lambda1 * adv_term + logvar_reg

    def update_parameters(self, step):
        # update anneal value
        if self.lmbda_schedule is not None:
            steps, min_anneal = self.lmbda_schedule
            delta = 1 / steps
            self.anneal = max(1.0 - step * delta, min_anneal)


# Metrics
class ReconstructionNLL(_Loss):
    """
    Standard reconstruction of images. There are two options, minimize the
    Bernoulli loss (i.e. per pixel binary cross entropy) or MSE (i.e. Gaussian
    likelihoid).
    """
    def __init__(self, loss='bce'):
        super().__init__(reduction='batchmean')
        if loss == 'bce':
            recons_loss = logits_bce
        elif loss == 'mse':
            recons_loss = mse_loss
        elif not callable(loss):
            raise ValueError('Unrecognized reconstruction'
                             'loss {}'.format(loss))
        self.loss = recons_loss

    def forward(self, input, target):
        if isinstance(input, (tuple, list)):
            recons = input[0]
        else:
            recons = input

        return self.loss(recons, target, reduction='sum') / target.size(0)


class GaussianKLDivergence(_Loss):
    """
    Computes the KL divergence between a latent variable and a standard normal
    distribution. Optinally, allows for computing the KL for a single
    dimension. This can be used to see which units are being used by the model
    to solve the task.
    """
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, input, targets):
        _, _, (mu, logvar) = input

        if self.dim >= 0:
            mu, logvar = mu[:, self.dim], logvar[:, self.dim]

        kl = gauss2standard_kl(mu, logvar).sum()
        return kl / targets.size(0)

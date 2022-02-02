"""
Sacred Ingredient for training functions.

The objective functions are defined and added as configureations to the
ingredient for ease of use. This allows chaging the objective function
easily and only needing to specify different parameters.
"""


import sys
import numpy as np
import torch
import torch.nn as nn
import ignite.metrics as M
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split
from sacred import Ingredient
from ignite.engine import Events

if sys.path[0] != '../src':
    sys.path.insert(0, '../src')

import training.loss as L
from training.handlers import EarlyStopping, ModelCheckpoint, Tracer
from training.optimizer import init_optimizer, init_lr_scheduler
# from training.loss import init_metrics as _init_metrics


def thresholded_output_transform(output):
    y_pred, y = output
    y_pred = (y_pred.sigmoid() > 0.5).to(dtype=y.dtype)
    return y_pred, y


bern_recons = {'name': 'recons_nll', 'params': {'loss': 'bce'}}
mse_recons = {'name': 'recons_nll', 'params': {'loss': 'mse'}}
kl_div = {'name': 'kl-div', 'params': {}}

vae_loss = {'name': 'vae', 'params': {'reconstruction_loss': 'bce'}}
bvae_loss = {'name': 'beta-vae', 'params': {'reconstruction_loss': 'bce',
                                            'beta': 4.0}}

mse_loss = {'name': 'mse', 'params': {}}
bxent_loss = {'name': 'bxent', 'params': {}}
xent_loss = {'name': 'xent', 'params': {}}
accuracy = {'name': 'acc',
            'params': {'output_transform': thresholded_output_transform}}


training = Ingredient('training')


init_optimizer = training.capture(init_optimizer)


@training.capture
def init_loader(dataset, batch_size, train_val_split=0.0, **loader_kwargs):
    kwargs = {'shuffle': True, 'pin_memory': True, 'prefetch_factor': 2,
              'num_workers': 4, 'persistent_workers': False}
    kwargs.update(**loader_kwargs)

    num_workers = kwargs['num_workers']

    def wif(pid):
        process_seed = torch.initial_seed()
        base_seed = process_seed - pid

        sequence_seeder = np.random.SeedSequence([pid, base_seed])
        np.random.seed(sequence_seeder.generate_state(4))

    kwargs['pin_memory'] = kwargs['pin_memory'] and torch.cuda.is_available()

    if train_val_split > 0.0:
        val_length = int(len(dataset) * train_val_split)
        lenghts = [len(dataset) - val_length, val_length]

        train_data, val_data = random_split(dataset, lenghts)

        train_loader = DataLoader(train_data, batch_size, **kwargs,
                                  worker_init_fn=(wif if num_workers > 1
                                                  else None))
        val_loader = DataLoader(val_data, batch_size, **kwargs,
                                worker_init_fn=(wif if num_workers > 1
                                                else None))

        return train_loader, val_loader

    loader = DataLoader(dataset, batch_size, **kwargs,
                        worker_init_fn=(wif if num_workers > 1 else None))

    return loader, loader


@training.capture
def init_loss(loss):
    loss_fn, params = loss['name'], loss['params']
    if loss_fn == 'vae':
        return L.GaussianVAELoss(**params, beta=1.0)
    elif loss_fn == 'beta-vae':
        return L.GaussianVAELoss(**params)
    elif loss_fn == 'wae-gan':
        return L.WAEGAN(**params)
    elif loss_fn == 'wae-mmd':
        return L.WAEMMD(**params)
    elif loss_fn == 'recons_nll':
        return L.ReconstructionNLL(**params)
    elif loss_fn == 'bxent':
        return nn.BCEWithLogitsLoss(**params)
    elif loss_fn == 'xent':
        return nn.CrossEntropyLoss(**params)
    elif loss_fn == 'mse':
        return nn.MSELoss(**params)
    else:
        raise ValueError('Unknown loss function {}'.format(loss_fn))


@training.capture
def init_metrics(metrics):
    metrics = list(map(dict.copy, metrics))

    labels = [m.pop('label', m['name']) for m in metrics]
    metrics = {l: get_metric(m) for (l, m) in zip(labels, metrics)}
    return metrics


@training.capture
def get_metric(metric):
    name = metric['name']
    params = metric['params']
    if name == 'mse':
        return M.MeanSquaredError(**params)
    elif name == 'vae':
        return M.Loss(L.GaussianVAELoss(**params))
    elif name == 'kl-div':
        return M.Loss(L.GaussianKLDivergence(**params))
    elif name == '2ndMM':
        return M.Loss(L.ReconstructionNLL(**params))
    elif name == 'bxent':
        return M.Loss(nn.BCEWithLogitsLoss(**params))
    elif name == 'xent':
        return M.Loss(nn.CrossEntropyLoss(**params))
    elif name == 'acc':
        return M.Accuracy(**params)
    raise ValueError('Unrecognized metric {}.'.format(metric))

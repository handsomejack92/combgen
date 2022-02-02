import sys
from itertools import product

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sacred import Ingredient
from ignite.contrib.metrics.regression import R2Score

from torch.utils.data.dataloader import DataLoader
from ignite.engine import create_supervised_evaluator

if '../src' not in sys.path:
    sys.path.append('../src')

from analysis.testing import infer
from analysis.metrics import DAF
from analysis.hinton import hinton


torch.set_grad_enabled(False)
sns.set(color_codes=True)
sns.set_style("white", {'axes.grid': False})
plt.rcParams.update({'font.size': 11})

analysis = Ingredient('analysis')


def create_r_squares(output_names):
    n_out = len(output_names)

    def slice_idx(i):
        def transform(output):
            y_pred, y = output
            return y_pred[:, i], y[:, i]
        return transform

    return {'{}'.format(n): R2Score(slice_idx(i))
            for i, n in enumerate(output_names)}


@analysis.capture
def model_score(model, data, metrics, model_name=None, device=None, *kwargs):
    dataloader_args = {'batch_size': 120, 'num_workers': 4, 'pin_memory': True}
    dataloader_args.update(kwargs)

    loader = DataLoader(data, **dataloader_args)

    if model_name is None:
        model_name = 'model'
    if device is None:
        device = next(model.parameters()).device

    engine = create_supervised_evaluator(model, metrics, device)
    metrics = engine.run(loader).metrics

    index = pd.Index(metrics.keys(), name='Metric')
    scores = pd.Series(metrics.values(), index=index, name=model_name)

    return scores


@analysis.capture
def get_recons(model, data, n_recons=10, loss='bce', **kwargs):
    dataloader_args = {'batch_size': n_recons, 'shuffle': True,
                       'pin_memory': True}
    dataloader_args.update(kwargs)

    inputs, targets = next(iter(DataLoader(data, **dataloader_args)))

    with torch.no_grad():
        model.eval()
        device = next(model.parameters()).device

        recons = model(inputs.to(device=device))
        if isinstance(recons, tuple):
            recons = recons[0]

        if loss == 'bce':
            recons = recons.sigmoid()
        else:
            recons = recons.clamp(0, 1)

        recons = recons.cpu()

        # for x in inputs:
        #     x = x.to(device=device)
        #     r = model(x.unsqueeze_(0))

        #     if isinstance(r, tuple):
        #         r = r[0]

        #     if loss == 'bce':
        #         r = r.sigmoid()
        #     else:
        #         r = r.clamp(0, 1)

        #     recons.append(r.cpu())

    # recons = torch.cat(recons)

    return inputs, recons, targets


@analysis.capture
def get_recons_plot(data, no_recon_labels=False, axes=None):
    inputs, recons = data

    batch_size = len(inputs)

    if axes is None:
        fig, axes = plt.subplots(2, batch_size, figsize=(2 * batch_size, 4))
    else:
        fig = None

    images = np.stack([inputs.numpy(), recons.numpy()])

    for j, (examp_imgs, ylab) in enumerate(zip(images, ['original', 'model'])):
        for i, img in enumerate(examp_imgs):
            if np.prod(img.shape) == 3 * 64 * 64:
                axes[j, i].imshow(img.reshape(3, 64, 64).transpose(1, 2, 0))
            else:
                axes[j, i].imshow(img.reshape(1, 64, 64).transpose(1, 2, 0),
                                  cmap='Greys_r')
            # if i == 0:
            #     axes[j, i].set_ylabel(ylab, fontsize=28)

    for ax in axes.reshape(-1):
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    if not no_recon_labels:
        axes[0, 0].set_ylabel('input', fontsize=20)
        axes[1, 0].set_ylabel('recons', fontsize=20)

    return fig


@analysis.capture
def infer(model, data, **kwargs):
    dataloader_args = {'batch_size': 128, 'num_workers': 4, 'pin_memory': True}
    dataloader_args.update(kwargs)

    loader = DataLoader(data, **dataloader_args)

    with torch.no_grad():
        model.eval()
        device = next(model.parameters()).device

        latents, targets = [], []
        for x, t in loader:
            x = x.to(device=device)
            z = model(x)

            if isinstance(z, tuple):
                z = z[1]

            latents.append(z.cpu())
            targets.append(t)

    latents = torch.cat(latents)
    targets = torch.cat(targets)

    return latents, targets


@analysis.capture
def get_factor_idxs(daf_results, factor1, factor2):
    # Find the corresponding dimension and assign
    f1_idx = daf_results.gt_names.index(factor1)
    f2_idx = daf_results.gt_names.index(factor2)

    dim1 = daf_results.sort_idx[f1_idx]
    dim2 = daf_results.sort_idx[f2_idx]

    return dim1, dim2


@analysis.capture
def latent_rep_plot(model, train_data, dim1, dim2, factor1, factor2,
                    test_data=None, train_proj=None, test_proj=None,
                    joint_palette='dim1'):
    # get encoded values
    if train_proj is not None:
        z, targets = train_proj
    else:
        z, targets = infer(model, train_data)

    # z += 0.01 * np.random.randn(*z.shape)

    if test_proj[0] is not None:
        if test_proj:
            z_test, test_targets = test_proj
        else:
            z_test, test_targets = infer(model, test_data)

        # z_test += 0.01 * np.random.randn(*z_test.shape)

        train_alpha = 0.1
    else:
        train_alpha = 1.0

    # Set plot
    fig = plt.figure(figsize=(10, 5))

    # dim1_ax = fig.add_subplot(3, 5, (5, 10))  # y axis
    # dim2_ax = fig.add_subplot(3, 5, (11, 14))  # x axis
    # joint_ax = fig.add_subplot(3, 5, (1, 9), sharex=dim2_ax)
    joint_ax = fig.add_subplot(111)

    # Remove unnecessary axes ticks and spines
    # plt.setp(joint_ax.get_xticklabels(), visible=False)
    # plt.setp(dim2_ax.get_yticklabels(), visible=False)

    # dim1_ax.set_xticklabels([])
    # dim1_ax.set_yticklabels([])

    # for ax in [dim1_ax, dim2_ax, joint_ax]:
    for ax in [joint_ax]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Set titles
    # dim1_ax.set_title(factor1)
    # dim2_ax.set_xlabel(factor2)
    joint_ax.set_ylabel(factor1.replace("_", " "), fontsize=36)
    joint_ax.set_xlabel(factor2.replace("_", " "), fontsize=36)
    joint_ax.tick_params(axis='both', which='major', labelsize=28)

    # Palettes for the plots
    n_colors = len(train_data.unique_values[factor1])
    dim1_palette = sns.color_palette("tab10", n_colors)

    n_colors = len(train_data.unique_values[factor2])
    dim2_palette = sns.color_palette("tab10", n_colors)

    def plot_marginal(z_proj, gf_codes, unique, palette, ax,
                      flip=False, fill=True):
        for c in unique:
            c_instances = z_proj[gf_codes == c]
            if len(c_instances) > 0:
                if flip:
                    sns.kdeplot(y=c_instances, color=palette[c],
                                ax=ax, fill=fill)
                else:
                    sns.kdeplot(x=c_instances, color=palette[c],
                                ax=ax, fill=fill)

    def plot_joint(z1_proj, z2_proj, gf1_codes, gf2_codes,
                   unique1, unique2, alpha=1.0, is_train=True):
        for c1, c2 in product(unique1, unique2):
            idx = (gf1_codes == c1) & (gf2_codes == c2)
            x_joint = z2_proj[idx]
            y_joint = z1_proj[idx]

            # if joint_palette == 'dim1':
            #     color = dim1_palette[c1]
            # else:
            #     color = dim2_palette[c2]
            if is_train:
                color = 'black'
            else:
                color = (0.85, 0.0, 0.0) # Not so bright red

            if len(x_joint) > 1:
                sns.kdeplot(x=x_joint, y=y_joint, color=color,
                            ax=joint_ax, alpha=alpha)
            elif len(x_joint) == 1:
                joint_ax.scatter(x_joint, y_joint, color=color,
                                alpha=alpha, marker='x')

    # training data projection
    z_dim1, z_dim2 = z[:, dim1], z[:, dim2]

    gt1_idx = train_data.factors.index(factor1)
    gt2_idx = train_data.factors.index(factor2)

    # ground truth codes (i.e. class index for each dimension)
    gf1_codes = train_data.factor_classes[:, gt1_idx]
    gf2_codes = train_data.factor_classes[:, gt2_idx]

    gf1_unique = np.unique(gf1_codes)
    gf2_unique = np.unique(gf2_codes)

    # plot_marginal(z_dim1, gf1_codes, gf1_unique,
    #               dim1_palette, dim1_ax, True, train_alpha == 1)
    # plot_marginal(z_dim2, gf2_codes, gf2_unique,
    #               dim2_palette, dim2_ax, False, train_alpha == 1)

    plot_joint(z_dim1, z_dim2, gf1_codes, gf2_codes,
               gf1_unique, gf2_unique)

    if test_data is not None:
        z_test_dim1, z_test_dim2 = z_test[:, dim1], z_test[:, dim2]

        gf1_codes = test_data.factor_classes[:, gt1_idx]
        gf2_codes = test_data.factor_classes[:, gt2_idx]

        gf1_unique = np.unique(gf1_codes)
        gf2_unique = np.unique(gf2_codes)

        # plot_marginal(z_test_dim1, gf1_codes, gf1_unique,
        #               dim1_palette, dim1_ax, True)
        # plot_marginal(z_test_dim2, gf2_codes, gf2_unique,
        #               dim2_palette, dim2_ax, False)

        plot_joint(z_test_dim1, z_test_dim2, gf1_codes, gf2_codes,
                   gf1_unique, gf2_unique, is_train=False)

    # joint_ax.set_ylim([-0.5, 1.5])
    # joint_ax.set_xlim([-0.1, 1.3])

    # dim1_ax.set_xlabel("")
    # dim2_ax.set_ylabel("")
    # joint_ax.set_xlabel("")
    # joint_ax.set_ylabel("")

    fig.tight_layout()

    return fig


@analysis.capture
def disentanglement_metric(model, train_data, test_data=None, projections=None,
                           method='lasso', assignment='optimal',
                           method_args=None):

    daf = DAF(train_data, test_data=test_data, method=method,
              assignment=assignment, method_kwargs=method_args)

    daf_results = daf(model, projections)

    return daf_results

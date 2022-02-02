import torch
from itertools import product, combinations

def gauss2standard_kl(mean, logvar):
    return -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())


def permute_dims(latent_sample):
    pi = torch.randn_like(latent_sample).argsort(dim=0)
    perm = latent_sample[pi, range(latent_sample.size(1))]
    return perm


def inv_multiquad_sum(x, y, scales, base_scale):
    scales = scales.to(x.device)

    quadratic_term = torch.sum((x - y).pow(2))
    scale_term = scales * base_scale

    return scale_term.div(scale_term + quadratic_term)

    # def kernel(scale):
    #     sxc = scale * base_scale
    #     return sxc / (sxc + quadratic_term)

    # return torch.as_tensor([kernel(s) for s in scales]).sum()


def min_mean_discrepancy(dist1, dist2, kernel, idxs=None):
    intra_dist_idx, cross_dist_idx = idxs if idxs else mmd_idxs(dist1)

    i_idx, j_idx = cross_dist_idx

    dist1_i, dist2_j = dist1[i_idx], dist2[j_idx]

    cross_dist_score = kernel(dist1_i, dist2_j).sum() / len(cross_dist_idx)

    i_idx, j_idx = intra_dist_idx
    dist1_i, dist1_j = dist1[i_idx], dist1[j_idx]
    dist2_i, dist2_j = dist2[i_idx], dist2[j_idx]

    intra_dist_score = (kernel(dist2_i, dist2_j) +
                        kernel(dist1_i, dist1_j)).sum() / len(intra_dist_idx)

    return intra_dist_score - 2 * cross_dist_score


def mmd_idxs(n_samples):
    def zipplist(idxs):
        return tuple(map(list, zip(*idxs)))  # Indices must be in list form

    intra_dist_idx = zipplist(combinations(range(n_samples), r=2))
    cross_dist_idx = zipplist(product(range(n_samples), repeat=2))

    return intra_dist_idx, cross_dist_idx


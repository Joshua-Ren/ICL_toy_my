# -*- coding: utf-8 -*-
"""
Here are some functions for sampling x for both training and ICL-prompting
Modified from https://github.com/dtsip/in-context-learning

1. Gaussain sampler: sample x~N(bias, scale)
2. Grid sample (non-overlap train/test split)


@author: YIREN
"""

import math
import torch


class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError


def get_data_sampler(data_name, n_dims, **kwargs):
    names_to_classes = {
        "gaussian": GaussianSampler,
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError


def sample_transformation(eigenvalues, normalize=False):
    n_dims = len(eigenvalues)
    U, _, _ = torch.linalg.svd(torch.randn(n_dims, n_dims))
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t


# ----------- Sample X from N(bias, scale)
class GaussianSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        # The sampled xs_b has the shape [b_size, n_points, n_dims]
        if seeds is None:
            xs_b = torch.randn(b_size, n_points, self.n_dims)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    skew = torch.diag(torch.tensor([1., 1]))
    skew = torch.tensor([[1,0.5],[0.5,1]])
    data_sampler = get_data_sampler('gaussian',n_dims=2,bias=torch.tensor([1,1]),scale=skew)
    xs = data_sampler.sample_xs(n_points=100, b_size=4)
    xs = xs.reshape(-1,2)
    plt.scatter(xs[:,0],xs[:,1])
    plt.xlim(-4,4)
    plt.ylim(-4,4)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
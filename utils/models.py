# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 22:10:57 2023
Modified from https://github.com/dtsip/in-context-learning
@author: YIREN
"""

import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Lasso
import warnings
from sklearn import tree
import xgboost as xgb



def build_model(args):
    if args.family == "gpt2":
        if args.gpt_type == 'tiny':
            n_embd, n_layer, n_head = 64, 3, 2
        elif args.gpt_type == 'small':
            n_embd, n_layer, n_head = 128, 6, 4
        elif args.gpt_type == 'standard':
            n_embd, n_layer, n_head = 256, 12, 8
        else:
            raise NotImplementedError        
        model = TransformerModel(
            n_dims=args.n_dims,
            n_positions=args.prompt_points,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
        ).to(args.device)
    else:
        raise NotImplementedError

    return model


class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, 1)

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        zs = self._combine(xs, ys)
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        return prediction[:, ::2, 0][:, inds]  # predict only on xs


# xs and ys should be on cpu for this method. Otherwise the output maybe off in case when train_xs is not full rank due to the implementation of torch.linalg.lstsq.
class LeastSquaresModel:
    def __init__(self, driver=None):
        self.driver = driver
        self.name = f"OLS_driver={driver}"

    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            ws, _, _, _ = torch.linalg.lstsq(
                train_xs, train_ys.unsqueeze(2), driver=self.driver
            )

            pred = test_x @ ws
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


if __name__ == "__main__":
    import argparse
    def update_args(args, config):
        for k in config.keys():
            args.__dict__[k] = config[k]
        return args
    parser = argparse.ArgumentParser(description='test')
    config ={"family":"gpt2", "n_positions": 10,"n_dims":20,"n_embd":64, "n_layer":3, "n_head":2}
    conf_args = update_args(parser,config)
    model = build_model(conf_args)
    n_dims = model.n_dims
    
    from samplers import get_data_sampler
    from tasks import get_task_sampler
    data_sampler = get_data_sampler('gaussian',n_dims=n_dims)
    xs = data_sampler.sample_xs(n_points=10, b_size=4)
    task_sampler = get_task_sampler('sparse_linear_regression',n_dims=n_dims, batch_size=4,sparsity=4)
    task = task_sampler()
    ys = task.evaluate(xs)
    
    output = model(xs, ys)
    

























# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 22:07:25 2023
Modified from https://github.com/dtsip/in-context-learning
task options:
    "LR": LinearRegression,
    "LC": LinearClassification,
    "NLR": NoisyLinearRegression,
    "QR": QuadraticRegression,
    "NN": Relu2nnRegression,   
@author: YIREN
"""
import os
from random import randint
import uuid
from tqdm import tqdm
import yaml
import argparse

import torch

from utils.samplers import get_data_sampler
from utils.tasks import get_task_sampler
from utils.models import build_model
from utils.curriculum import Curriculum
from utils.general import *


def get_args_parser():
    # Training settings
    # ========= Usually default settings
    parser = argparse.ArgumentParser(description='Iterated ICL Toy')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')    
    parser.add_argument('--config_file', type=str, default=None,
                        help='the name of the toml configuration file')
    parser.add_argument('--seed', default=0, type=int)
    
    # ========= Model related
    parser.add_argument('--family', default='gpt2', type=str,
                        help='model type of pretraining')   
    parser.add_argument('--n_dims', default=10, type=int,
                        help='embedding dimension used in transformer model')
    parser.add_argument('--gpt_type',default='standard',type=str,
                        help='what type of gpt2 we use, tiny, small, standard')
    
    # ========= Task related    
    parser.add_argument('--prompt_points', default=20, type=int,
                        help='number of examples used in prompt')    
    parser.add_argument('--data_x', default='gaussian', type=str,
                        help='x distribution, gaussian, ')       
    parser.add_argument('--task', default='LR', type=str,
                        help='x distribution, LR, LC, NLR, QR, NN ')      
    
    # ========= Training
        # ---- Curriculum
    parser.add_argument('--curr_interval', default=2000, type=int,
                        help='interval of updating curriculum')
    parser.add_argument('--curr_dim_start', default=5, type=int,
                        help='start dim of the curriculum')
    parser.add_argument('--curr_points_start', default=10, type=int,
                        help='start number of points of the curriculum')
    
    parser.add_argument('--b_size', default=64, type=int,
                        help='batch size used during training')
    parser.add_argument('--train_steps', default=5, type=int,
                        help='number of parameter updates during training')    
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='the learning rate for training')

    
    
    # ===== Wandb and saving results ====
    parser.add_argument('--WD_ID',default='joshuaren', type=str,
                        help='W&D ID, joshuaren or joshua_shawn')
    parser.add_argument('--save_model', default=False, type=eval, 
                        help='Whether save the model in the save-path') 
    parser.add_argument('--save_every_steps', default=10000, type=int,
                        help='gap between different savings')     
    parser.add_argument('--run_name',default='test',type=str)
    parser.add_argument('--proj_name',default='P5_iICL_toy', type=str)
    return parser


def train_step(model, xs, ys, optimizer, loss_func):
    optimizer.zero_grad()
    output = model(xs, ys)
    loss = loss_func(output, ys)
    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach()

   
    
def train(model, optimizer, args):
    model.train()
    # ---------- Continue training if there exist checkpoint in the folder
    curriculum = Curriculum(args)
    starting_step = 0
    ckp_path = os.path.join(args.save_path, "state.pt")
    if os.path.exists(ckp_path):
        state = torch.load(ckp_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state['train_step']+1):
            curriculum.update()

    data_sampler = get_data_sampler(args.data_x, n_dims=args.n_dims)
    task_sampler = get_task_sampler(args.task, args.n_dims, args.b_size)
    pbar = tqdm(range(starting_step, args.train_steps))
    
    for i in pbar:
        xs = data_sampler.sample_xs(n_points=curriculum.n_points, 
                                    b_size=args.b_size,
                                    n_dims_truncated=curriculum.n_dims)
        task = task_sampler()
        ys = task.evaluate(xs)
        xs, ys = xs.to(args.device), ys.to(args.device)
        loss_func = task.get_training_metric()
        loss, output = train_step(model, xs, ys, optimizer, loss_func)
        
        point_wise_tags = list(range(curriculum.n_points))
        point_wise_loss_func = task.get_metric()
        point_wise_loss = point_wise_loss_func(output, ys).mean(dim=0)
        point_wise_dict = dict(zip(point_wise_tags, point_wise_loss.cpu().numpy()))
        wandb.log({'training_loss':loss})
        #wandb.log({'pointwise_loss':point_wise_dict})
        curriculum.update()
        
        if i % args.save_every_steps == 0 or i==args.train_steps-1:
            tmp_name = "state_"+str(i).zfill(6)+".pt"
            save_path = os.path.join(args.save_path, tmp_name)
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, save_path)        
def main(args):
    # ========== Generate seed ==========
    if args.seed==0:
        args.seed = torch.randint(1,10086,(1,)).item()
    rnd_seed(args.seed)    
    
    # ========== Prepare save folder and wandb ==========
    wandb_init(proj_name=args.proj_name, run_name=args.run_name, config_args=args)
    args.save_path = os.path.join('results', args.run_name)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)            
    
    model = build_model(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train(model, optimizer, args)
    if args.save_model:
        ckp_path = os.path.join(args.save_path, '.pt')
        torch.save(model.state_dict(), ckp_path)        
    wandb.finish()
    

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    args.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if args.config_file is not None:
        config = toml.load(os.path.join("configs",args.config_file+".toml"))
        args = update_args(args, config)
    main(args)
    """
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
    """


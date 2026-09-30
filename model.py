"""
Denoising Diffusion (DDPM) from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - linear_beta_schedule
import torch
import torch.nn.functional as F

def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    # TODO: return a linear beta schedule of length T
    return torch.linspace(beta_start, beta_end,T)

# Step 2 - alphas_from_betas
import torch
import torch.nn.functional as F

def alphas_from_betas(betas):
    # TODO: return 1 - betas
    return 1 - betas

# Step 3 - cumprod_alphas
import torch
import torch.nn.functional as F

def cumprod_alphas(alphas):
    # TODO: cumulative product of alphas
    return torch.cumprod(alphas,dim=0)

# Step 4 - extract_into_batch
import torch
import torch.nn.functional as F

def extract_into_batch(a, t, x):
    # TODO: gather a[t] and reshape to (B, 1, 1, 1) for broadcasting with x
    return torch.gather(a, dim=0, index=t).view(-1,1,1,1)

# Step 5 - q_sample
import torch
import torch.nn.functional as F

def q_sample(x0, t, noise, alphas_cumprod):
    # TODO: x_t = sqrt(bar_alpha_t) * x0 + sqrt(1 - bar_alpha_t) * noise
    steps = extract_into_batch(alphas_cumprod, t, x0)
    return x0 * torch.sqrt(steps) + torch.sqrt(1-steps) * noise

# Step 6 - build_diffusion_schedule
import torch
import torch.nn.functional as F

def build_diffusion_schedule(T: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02) -> dict:
    # TODO: build betas, alphas, alphas_cumprod and useful sqrts
    betas = torch.linspace(beta_start,beta_end,T)
    return {
        'betas':betas,
        'alphas':1-betas,
        'alphas_cumprod': torch.cumprod(1-betas,dim=0),
        'sqrt_alphas_cumprod': torch.sqrt(torch.cumprod(1-betas,dim=0)),
        'sqrt_one_minus_alphas_cumprod': torch.sqrt(1-torch.cumprod(1-betas,dim=0)),
        'T':T
    }

# Step 7 - noise_prediction_loss
import torch
import torch.nn.functional as F

def noise_prediction_loss(noise_pred, noise):
    # TODO: MSE between predicted and true noise
    return torch.mean((noise-noise_pred)**2)

# Step 8 - diffusion_training_loss
import torch
import torch.nn.functional as F

def diffusion_training_loss(model, x0, t, noise, alphas_cumprod):
    # TODO: q_sample -> model -> MSE(noise_pred, noise)
    xt = q_sample(x0, t, noise, alphas_cumprod)
    noise_pred = model(xt, t)
    return noise_prediction_loss(noise_pred, noise)

# Step 9 - timestep_embedding
import torch
import torch.nn.functional as F

def timestep_embedding(t, dim: int):
    # TODO: sinusoidal timestep embedding of shape (B, dim)
    B = t.shape[0]
    PE = torch.zeros((B,dim))
    PE[:, 0:(dim//2)] = torch.tensor([10000**(-2*i/dim) for i in range(0,dim//2)])
    PE[:, (dim//2):] = torch.tensor([10000**(-2*i/dim) for i in range(0,dim - dim//2)])
    PE = PE * t.view(B,1)
    PE[:, 0:(dim//2)] = torch.sin(PE[:, 0:(dim//2)])
    PE[:, (dim//2):] = torch.cos(PE[:, (dim//2):])
    return PE

# Step 10 - init_tiny_unet
import torch
import torch.nn.functional as F

def init_tiny_unet(in_ch: int = 1, hidden: int = 16, time_dim: int = 16, seed: int = 0) -> dict:
    # TODO: initialize tiny residual denoiser parameters
    torch.manual_seed(seed)
    return {
        'conv_in_w': (torch.randn((hidden,in_ch,3,3)) * 0.02).requires_grad_(True),
        'conv_in_b': torch.zeros((hidden,),requires_grad=True),
        'time_mlp_w': (torch.randn((hidden,time_dim))* 0.02).requires_grad_(True),
        'time_mlp_b': torch.zeros((hidden,),requires_grad=True),
        'conv_mid_w': (torch.randn((hidden,hidden,3,3))* 0.02).requires_grad_(True),
        'conv_mid_b':torch.zeros((hidden,),requires_grad=True),
        'conv_out_b': torch.zeros((in_ch,),requires_grad=True),
        'conv_out_w':(torch.randn((in_ch,hidden,3,3))* 0.02).requires_grad_(True),
    }

# Step 11 - tiny_unet_forward
import torch
import torch.nn.functional as F

def tiny_unet_forward(x, t, params: dict):
    # TODO: time-conditioned tiny CNN predicting noise
    h = F.conv2d(x, params['conv_in_w'], params['conv_in_b'], padding=1)
    temb = timestep_embedding(t, params['time_mlp_w'].shape[1])
    temb = F.relu(F.linear(temb, params['time_mlp_w'], params['time_mlp_b']))
    h = h + temb[:,:,None,None]
    h = F.relu(h)
    h = F.relu(F.conv2d(h,params['conv_mid_w'],params['conv_mid_b'], padding=1))
    return F.conv2d(h, params['conv_out_w'], params['conv_out_b'],padding=1)

# Step 12 - make_blob_dataset
import torch
import torch.nn.functional as F

def make_blob_dataset(n: int = 128, size: int = 8, seed: int = 0):
    # TODO: n images with a random bright disk on a black background
    torch.manual_seed(seed)
    radius = size//4
    ys, xs = torch.meshgrid(
        torch.arange(size), torch.arange(size), indexing="ij"
    )

    x = torch.zeros(n, 1, size, size)
    for i in range(n):
        cy, cx = torch.randint(radius, size - radius, (2,)).tolist()
        mask = (ys - cy) ** 2 + (xs - cx) ** 2 <= radius ** 2
        x[i, 0][mask] = 1.0
    return x

# Step 13 - ddpm_train_step
import torch
import torch.nn.functional as F

def ddpm_train_step(params: dict, x0, schedule: dict, lr: float = 1e-2, seed: int = 0) -> tuple[dict, float]:
    # TODO: sample t,noise -> loss -> SGD on params
    torch.manual_seed(seed)
    t = torch.randint(low=0,high=schedule['T'],size=(x0.shape[0],) )
    noise = torch.randn_like(x0)
    model = lambda x,t: tiny_unet_forward(x,t,params)
    loss = diffusion_training_loss(model, x0, t, noise, schedule['alphas_cumprod'])
    loss.backward()
    for k,p in params.items():
        if isinstance(p, torch.Tensor) and p.grad is not None:
            params[k] = (p-lr*p.grad).detach().requires_grad_(True)
    return params, loss.item()

# Step 14 - train_ddpm (not yet solved)
# TODO: implement

# Step 15 - predict_x0_from_eps (not yet solved)
# TODO: implement

# Step 16 - ddpm_p_mean_variance (not yet solved)
# TODO: implement

# Step 17 - ddpm_p_sample (not yet solved)
# TODO: implement

# Step 18 - ddpm_sample_loop (not yet solved)
# TODO: implement

# Step 19 - sample_quality_mse (not yet solved)
# TODO: implement

# Step 20 - ddpm_experiment (not yet solved)
# TODO: implement


import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import jax.image
import os
import numpy as np
from PIL import Image
from functools import partial
import pickle
from scripts.data_load import get_dataloader

T = 1000

def make_beta_schedule(n_timesteps, start=1e-4, end=0.02):
    return jnp.linspace(start, end, n_timesteps)

betas = make_beta_schedule(T)
alphas = 1.0 - betas
alphas_cumprod = jnp.cumprod(alphas)
alphas_cumprod_prev = jnp.concatenate([jnp.array([1.0]), alphas_cumprod[:-1]], axis=0)

class PointNet(nn.Module):
    hidden_dim: int = 128
    num_points: int = 256

    @nn.compact
    def __call__(self, x):  # x: [B, num_points, 3]
        # Input Transformation Network
        transform_input = nn.Dense(64)(x)
        transform_input = nn.relu(transform_input)
        transform_input = nn.Dense(128)(transform_input)
        transform_input = nn.relu(transform_input)
        transform_input = nn.Dense(9)(transform_input)  # Learn transformation matrix (3x3)
        transform_input = transform_input.mean(axis=1)  # [B, 9]
        transform_matrix = transform_input.reshape(-1, 3, 3)  # [B, 3, 3]

        # Apply transformation to input
        x = jnp.einsum("bij,bpj->bpi", transform_matrix, x)  # [B, num_points, 3]

        # Feature Extraction
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)  # [B, num_points, hidden_dim]

        # Aggregation (Global Feature Vector)
        x = jnp.mean(x, axis=1)  # [B, hidden_dim]

        return x

class MLP(nn.Module):
    """A very simplified U-Net-like model for DDPM."""
    @nn.compact
    def __call__(self, x, t, branch_pcs, mug_pcs):
        hidden_dim = 256
        def timestep_embedding(t, dim=64):
            half_dim = dim // 2
            freqs = jnp.exp(-jnp.log(10000) * jnp.arange(half_dim) / half_dim)
            freqs = t[:, None]*freqs[None]
            emb = jnp.concatenate([jnp.sin(freqs), jnp.cos(freqs)], axis=-1)
            return emb

        t_emb = timestep_embedding(t, dim=64)
        t_emb = nn.Dense(features=hidden_dim // 2)(t_emb)
        t_emb = jax.nn.silu(t_emb)
        t_emb = nn.Dense(features=hidden_dim)(t_emb)
        t_emb = jax.nn.silu(t_emb) # [B, 256]

        # x: [B, 7]
        x_emb = nn.Dense(features=hidden_dim // 2)(x)
        x_emb = jax.nn.silu(x_emb)
        x_emb = nn.Dense(features=hidden_dim)(x_emb) # [B, 256]

        branch_pcs_emb = PointNet(hidden_dim, 256)(branch_pcs) # [B, 256]
        mug_pcs_emb = PointNet(hidden_dim, 256)(mug_pcs) # [B, 256]
        emb = jnp.concatenate([t_emb, x_emb, branch_pcs_emb, mug_pcs_emb], axis=-1) # [B, 1024]
        emb = nn.Dense(features=hidden_dim)(emb)
        emb = jax.nn.silu(emb)
        emb = nn.Dense(features=hidden_dim // 2)(emb)
        emb = jax.nn.silu(emb)
        emb = nn.Dense(features=7)(emb) # [B, 7]

        return emb

def q_sample(rng, x_0, t):
    # t is a vector of shape (B,)
    a_t = alphas_cumprod[t]
    a_t = a_t[:, None]
    noise = jax.random.normal(rng, x_0.shape)
    return jnp.sqrt(a_t)*x_0 + jnp.sqrt(1 - a_t)*noise

def predict_noise(model, params, rng, x_t, t, branch_pcs, mug_pcs):
    return model.apply(params, x_t, t, branch_pcs, mug_pcs)

def diffusion_loss(model, params, rng, x_0):
    mug_pose_0 = x_0['mug_poses']
    bsz = mug_pose_0.shape[0]
    rng_t, rng_noise = jax.random.split(rng, 2)
    t = jax.random.randint(rng_t, shape=(bsz,), minval=0, maxval=T)
    mug_pose_t = q_sample(rng_noise, mug_pose_0, t)
    a_t = alphas_cumprod[t][:, None]
    noise_actual = (mug_pose_t - jnp.sqrt(a_t)*mug_pose_0) / jnp.sqrt(1 - a_t)
    noise_pred = predict_noise(model, params, rng, mug_pose_t, t, x_0['branch_pcs'], x_0['mug_pcs'])
    loss = jnp.mean((noise_pred - noise_actual)**2)
    return loss

def p_sample_step(model, params, rng, x_t, t, branch_pcs, mug_pcs):
    """
    Sample x_{t-1} given x_t using the model.
    """
    # t should be a scalar or array of shape [B,]
    # Ensure t is a jnp array of shape (B,)
    # If x_t has shape (B, feature_dim), then t should be broadcastable
    bsz = x_t.shape[0]
    t_b = jnp.ones((bsz,), dtype=jnp.int32)*t
    
    # Predict noise
    eps_pred = model.apply(params, x_t, t_b, branch_pcs, mug_pcs)
    
    # Compute x_0 estimate
    alpha_t = alphas[t]
    alpha_bar_t = alphas_cumprod[t]
    x_0_est = (x_t - jnp.sqrt(1.0 - alpha_t)*eps_pred) / jnp.sqrt(alpha_t)
    
    # Compute the mean of q(x_{t-1} | x_t, x_0_est)
    alpha_bar_prev = alphas_cumprod_prev[t]
    beta_t = betas[t]
    # Posterior variance
    # Using DDPM formula: 
    # p(x_{t-1}|x_t) ~ N(x_{t-1}; mean = mu_t, variance = \tilde{\beta}_t)
    # where
    # mu_t = \sqrt{\bar{\alpha}_{t-1}} \frac{\beta_t}{1 - \bar{\alpha}_t} x_0_est 
    #        + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t
    # and
    # \tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t
    coef1 = (jnp.sqrt(alpha_bar_prev) * beta_t) / (1.0 - alpha_bar_t)
    coef2 = (jnp.sqrt(alpha_t) * (1 - alpha_bar_prev)) / (1 - alpha_bar_t)
    mu_t = coef1 * x_0_est + coef2 * x_t
    
    # Compute log variance (we will add noise if t > 0)
    # \tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t
    var_t = (1 - alpha_bar_prev) / (1 - alpha_bar_t) * beta_t
    
    rng, rng_noise = jax.random.split(rng)
    noise = jax.random.normal(rng_noise, x_t.shape)
    
    # If t > 0, add noise, else no noise
    x_t_minus_1 = mu_t + jnp.sqrt(var_t)*noise if t > 0 else mu_t
    return x_t_minus_1, rng

def p_sample_loop(model, params, rng, shape, branch_pcs, mug_pcs):
    """
    Run the full reverse diffusion loop to sample from the model.
    shape: (B, 7) for mug_poses if that's what we're sampling.
    branch_pcs: (B, num_points, 3)
    mug_pcs: (B, num_points, 3)
    
    Returns x_0 samples.
    """
    # Start from random noise
    rng, rng_init = jax.random.split(rng)
    x_T = jax.random.normal(rng_init, shape)  # x_T ~ N(0, I)
    x_t = x_T
    for i in reversed(range(T)):
        x_t, rng = p_sample_step(model, params, rng, x_t, i, branch_pcs, mug_pcs)
    return x_t


if __name__ == "__main__":
    batch_size = 128
    train_dataloader, val_dataloader = get_dataloader("dataset", batch_size, 256)

    model = MLP()
    rng = jax.random.PRNGKey(0)
    diffusion_loss = partial(diffusion_loss, model)
    loss_grad = jax.grad(diffusion_loss, has_aux=False)

    batch = next(iter(train_dataloader))
    sample_t = jnp.array([0], dtype=jnp.int32)
    mug_pose = batch['mug_poses'][0:1]
    branch_pcs = batch['branch_pcs'][0:1]
    mug_pcs = batch['mug_pcs'][0:1]

    params = model.init(rng, mug_pose, sample_t, branch_pcs, mug_pcs)
    optimizer = optax.adam(1e-4)
    opt_state = optimizer.init(params)

    def train_step(params, opt_state, rng, x):
        grads = loss_grad(params, rng, x)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    train_step_jit = jax.jit(train_step)
    diffusion_loss_jit = jax.jit(diffusion_loss)
    # train_step_jit = train_step

    num_steps = 1000  # Adjust as needed

    for step in range(num_steps):
        for batch in train_dataloader:
            rng, step_rng = jax.random.split(rng)
            params, opt_state = train_step_jit(params, opt_state, step_rng, batch)

        loss_val = 0
        for batch in val_dataloader:
            loss_val += diffusion_loss_jit(params, step_rng, batch)
        loss_val /= len(val_dataloader)
        print(f"Step {step}, loss: {loss_val.item()}")

        # save model
        if step % 100 == 0:
            save_dict = {
                "params": params,
                "opt_state": opt_state,
                "step": step,
            }
            with open(os.path.join("checkpoint", f'save_dict_{step}.pkl'), 'wb') as f:
                pickle.dump(save_dict, f)

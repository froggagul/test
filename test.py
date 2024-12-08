import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import tensorflow_datasets as tfds
import jax.image
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from functools import partial

# ==========================
# Diffusion Hyperparameters
# ==========================
T = 1000

def make_beta_schedule(n_timesteps, start=1e-4, end=0.02):
    return jnp.linspace(start, end, n_timesteps)

betas = make_beta_schedule(T)
alphas = 1.0 - betas
alphas_cumprod = jnp.cumprod(alphas)
alphas_cumprod_prev = jnp.concatenate([jnp.array([1.0]), alphas_cumprod[:-1]], axis=0)

# ==========================
# Data Loading (CIFAR-10)
# ==========================
def load_cifar10(batch_size, split='train'):
    ds = tfds.load('cifar10', split=split, as_supervised=True)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y))
    ds = ds.shuffle(10_000).batch(batch_size).repeat()
    return ds

batch_size = 64
train_ds = load_cifar10(batch_size)
train_iter = iter(tfds.as_numpy(train_ds))

# ==========================
# Simple U-Net-like Model
# ==========================
class DownBlock(nn.Module):
    features: int
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.features, kernel_size=(3,3), strides=(2,2), padding='SAME')(x)
        x = nn.GroupNorm()(x)
        x = jax.nn.silu(x)
        return x

class UpBlock(nn.Module):
    features: int
    @nn.compact
    def __call__(self, x):
        # Using jax.image.resize for simplicity
        x = jax.image.resize(x, shape=(x.shape[0], x.shape[1]*2, x.shape[2]*2, x.shape[3]), method='nearest')
        x = nn.Conv(self.features, kernel_size=(3,3), strides=(1,1), padding='SAME')(x)
        x = nn.GroupNorm()(x)
        x = jax.nn.silu(x)
        return x

class UNet(nn.Module):
    """A very simplified U-Net-like model for DDPM."""
    @nn.compact
    def __call__(self, x, t):
        def timestep_embedding(t, dim=64):
            half_dim = dim // 2
            freqs = jnp.exp(-jnp.log(10000) * jnp.arange(half_dim) / half_dim)
            freqs = t[:, None]*freqs[None]
            emb = jnp.concatenate([jnp.sin(freqs), jnp.cos(freqs)], axis=-1)
            return emb

        emb = timestep_embedding(t, dim=64)
        emb = nn.Dense(features=128)(emb)
        emb = jax.nn.silu(emb)
        emb = nn.Dense(features=256)(emb)
        emb = jax.nn.silu(emb)  # additional nonlinearity

        # Initial Conv
        h = nn.Conv(64, (3,3), padding='SAME')(x)
        h = jax.nn.silu(h)

        # Downsampling
        h_down1 = DownBlock(128)(h)
        h_down2 = DownBlock(256)(h_down1)

        # Bottleneck
        # Add time embedding
        h_mid = nn.Conv(256, (3,3), padding='SAME')(h_down2)
        h_mid = nn.GroupNorm()(h_mid)
        # Broadcast emb over spatial dims
        emb_b = emb[:, None, None, :]
        h_mid = h_mid + emb_b
        h_mid = jax.nn.silu(h_mid)

        # Upsampling
        h_up1 = UpBlock(128)(h_mid)
        # Concatenate skip connection
        h_up1 = jnp.concatenate([h_up1, h_down1], axis=-1)
        h_up1 = nn.Conv(128, (3,3), padding='SAME')(h_up1)
        h_up1 = nn.GroupNorm()(h_up1)
        h_up1 = jax.nn.silu(h_up1)

        h_up2 = UpBlock(64)(h_up1)
        h_up2 = jnp.concatenate([h_up2, h], axis=-1)
        h_up2 = nn.Conv(64, (3,3), padding='SAME')(h_up2)
        h_up2 = nn.GroupNorm()(h_up2)
        h_up2 = jax.nn.silu(h_up2)

        # Final
        h_out = nn.Conv(3, (3,3), padding='SAME')(h_up2)
        return h_out


# ==========================
# Diffusion Utilities
# ==========================
def q_sample(rng, x_0, t):
    # t is a vector of shape (B,)
    a_t = alphas_cumprod[t]
    a_t = a_t[:, None, None, None]
    noise = jax.random.normal(rng, x_0.shape)
    return jnp.sqrt(a_t)*x_0 + jnp.sqrt(1 - a_t)*noise

def predict_noise(model, params, rng, x_t, t):
    return model.apply(params, x_t, t)

def diffusion_loss(model, params, rng, x_0):
    bsz = x_0.shape[0]
    rng_t, rng_noise = jax.random.split(rng, 2)
    t = jax.random.randint(rng_t, shape=(bsz,), minval=0, maxval=T)
    x_t = q_sample(rng_noise, x_0, t)
    a_t = alphas_cumprod[t][:, None, None, None]
    noise_actual = (x_t - jnp.sqrt(a_t)*x_0) / jnp.sqrt(1 - a_t)
    noise_pred = predict_noise(model, params, rng, x_t, t)
    loss = jnp.mean((noise_pred - noise_actual)**2)
    return loss

# ==========================
# Training Step
# ==========================

def save_timesteps_concatenated(all_images, filepath):
    """
    Concatenate multiple timesteps of images into one large image.
    all_images: (T, B, H, W, C), where T is number of timesteps,
                B is batch size, and each image is HxW with C channels.
    """
    # Convert to uint8
    all_images = np.array(all_images * 255, dtype=np.uint8)
    
    rows = []
    for t in range(all_images.shape[0]):
        # Concatenate this timestep's images horizontally
        row = np.concatenate([all_images[t, i] for i in range(all_images.shape[1])], axis=1)
        rows.append(row)
    
    # Concatenate all rows vertically
    combined = np.concatenate(rows, axis=0)
    im = Image.fromarray(combined)
    im.save(filepath)
    print(f"Saved visualization to {filepath}")

def p_sample_step(model, params, rng, t_idx, x_t):
    # p_sample_step with t_idx as a static argument number if you want.
    # However, t_idx changes each iteration, so we might remove static_argnums for t_idx.
    # Just remove static_argnums if it causes issues:
    beta_t = betas[t_idx]
    alpha_t = alphas[t_idx]
    bar_alpha_t = alphas_cumprod[t_idx]
    bar_alpha_prev = alphas_cumprod_prev[t_idx]

    bsz = x_t.shape[0]
    t = jnp.ones((bsz,), dtype=jnp.int32)*t_idx
    eps = model.apply(params, x_t, t)

    x_0 = (x_t - jnp.sqrt(1 - bar_alpha_t)*eps) / jnp.sqrt(bar_alpha_t)

    mu_t = (jnp.sqrt(bar_alpha_prev)*beta_t/(1 - bar_alpha_t))*x_0 + \
           (jnp.sqrt(alpha_t)*(1 - bar_alpha_prev)/(1 - bar_alpha_t))*x_t

    rng, step_rng = jax.random.split(rng)
    x_prev = jnp.where(t_idx > 0, mu_t + jnp.sqrt(beta_t)*jax.random.normal(step_rng, x_t.shape), mu_t)
    return x_prev, rng

def run_reverse_diffusion(p_sample_step, params, rng, x_init):
    """
    Run the entire reverse diffusion from T to 0 inside a JIT-compiled function.
    We'll store all intermediate steps in a big array of shape (T+1, B, H, W, C).
    """
    B, H, W, C = x_init.shape

    # We will create an array to store x at all timesteps
    def body_fun(i, carry):
        # i goes from 0 to T-1
        # The current timestep index in reverse diffusion = T-1 - i
        # This means when i=0, t_idx = T-1
        # when i=1, t_idx = T-2, etc.
        x, rng, all_x = carry
        t_idx = T-1 - i
        x, rng = p_sample_step(params, rng, t_idx, x)
        all_x = all_x.at[i+1].set(x) # store the new x
        return (x, rng, all_x)

    # all_x[0] = x at T
    all_x = jnp.zeros((T+1, B, H, W, C), dtype=jnp.float32)
    all_x = all_x.at[0].set(x_init)

    # fori_loop: loop from i=0 to T-1
    x_final, rng_final, all_x = jax.lax.fori_loop(0, T, body_fun, (x_init, rng, all_x))
    return all_x


    # indices = [T - t for t in timesteps_to_save]
    # selected = all_x[indices]

    # # Clip before saving
    # selected = jnp.clip(selected, 0.0, 1.0)
    # save_timesteps_concatenated(selected, output_image)


# ==========================
# Initialization & Training
# ==========================
if __name__ == "__main__":
    # def loss_fn(model, p, rng, x):
    #     return diffusion_loss(model, p, rng, x), {}

    model = UNet()
    rng = jax.random.PRNGKey(0)
    diffusion_loss = partial(diffusion_loss, model)
    loss_grad = jax.grad(diffusion_loss, has_aux=False)

    sample_input = jnp.zeros((1, 32, 32, 3), dtype=jnp.float32)
    sample_t = jnp.array([0], dtype=jnp.int32)

    params = model.init(rng, sample_input, sample_t)
    optimizer = optax.adam(1e-4)
    opt_state = optimizer.init(params)

    def train_step(params, opt_state, rng, x):
        grads = loss_grad(params, rng, x)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    p_sample_step = partial(p_sample_step, model)
    run_reverse_diffusion = partial(run_reverse_diffusion, p_sample_step)
    jit_run_reverse_diffusion = jax.jit(run_reverse_diffusion)

    def visualize_diffusion_process(params, rng, shape=(4, 32, 32, 3), 
                                    timesteps_to_save=[1000, 750, 500, 250, 0], 
                                    output_image="visualization.png"):
        # Start from x_T ~ N(0, I)
        x_init = jax.random.normal(rng, shape)
        # Run the entire process in one JIT-compiled call
        all_x = jit_run_reverse_diffusion(params, rng, x_init)
        indices = T - np.array(timesteps_to_save, dtype=int)
        selected = all_x[indices]

        # Clip before saving
        selected = jnp.clip(selected, 0.0, 1.0)
        save_timesteps_concatenated(selected, output_image)
        return all_x

    train_step_jit = jax.jit(train_step)

    num_steps = 10_000_000  # Adjust as needed


    for step in range(num_steps):
        batch_x, batch_y = next(train_iter)
        batch_x = jnp.array(batch_x)
        rng, step_rng = jax.random.split(rng)
        params, opt_state = train_step_jit(params, opt_state, step_rng, batch_x)

        if step % 100 == 0:
            loss_val = diffusion_loss(params, step_rng, batch_x)
            print(f"Step {step}, loss: {loss_val.item()}")

        if step % 10000 == 0:
            visualize_diffusion_process(params, step_rng, (4, 32, 32, 3), [1000, 750, 500, 250, 0], f"samples/{step}.png")

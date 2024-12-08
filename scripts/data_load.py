import numpy as np
import jax
import jax.numpy as jnp
import torch

import glob

import torch.utils.data.dataset

def get_data(path):
    data_paths = glob.glob(path + "/*.npy")
    datas = []
    for path in data_paths[:1]:
        data = np.load(path, allow_pickle=True)
        datas.append(data.item())
    datas = jax.tree_util.tree_map(lambda *x: jnp.concatenate(x, axis = 0), *datas)
    return datas

def pytree_collate(batch):
  """Simple collation for numpy pytree instances"""
  data = jax.tree_util.tree_map(lambda *x: np.stack(x, 0), *batch)
  return data

def get_dataloader(path, batch_size, val_data_len):
    data = get_data(path)
    dataset = torch.utils.data.StackDataset(
        **data,
    )
    train_data_len = int(len(dataset) - val_data_len)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_data_len, val_data_len])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=pytree_collate)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, collate_fn=pytree_collate)
    return train_dataloader, val_dataloader


# if __name__ == "__main__":
#     # data = get_data("dataset")
#     # dataset = torch.utils.data.StackDataset(
#     #     **data,
#     # )
#     # train_data_len = int(len(dataset) - 1000)
#     # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_data_len, len(dataset) - train_data_len])
#     # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, collate_fn=pytree_collate)
#     # val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, collate_fn=pytree_collate)
#     train_dataloader, val_dataloader = get_dataloader("dataset", 32, 1000)
#     for i in range(10000):
#         batch = next(iter(train_dataloader))
#         print(i, batch['mug_pcs'].shape)


if __name__ == "__main__":
    train_dataloader, val_dataloader = get_dataloader("dataset", 64, 1000)
    for i in range(10000):
        batch = next(iter(train_dataloader))
        print(i, batch['mug_pcs'].shape, batch['mug_pcs'][0][0])
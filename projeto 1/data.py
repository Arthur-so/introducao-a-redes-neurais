import numpy as np
import torch

CSV = "dataset_projeto1.csv"
SEED = 42


def load_splits(csv=CSV, seed=SEED):
    """Divide o dataset em treino (10%), validacao (10%) e teste (80%)."""
    data = np.loadtxt(csv, delimiter=",", skiprows=1)
    idx = np.random.default_rng(seed).permutation(len(data))
    n = int(0.1 * len(data))
    return tuple(data[p] for p in np.split(idx, [n, 2 * n]))


class Scaler:
    """Padroniza x e y usando apenas as estatisticas do treino."""

    def __init__(self, train):
        self.mean = train.mean(axis=0)
        self.std = train.std(axis=0)

    def to_tensors(self, split):
        z = (split - self.mean) / self.std
        x = torch.tensor(z[:, :1], dtype=torch.float32)
        y = torch.tensor(z[:, 1:], dtype=torch.float32)
        return x, y

    def denorm_y(self, y):
        return y * self.std[1] + self.mean[1]

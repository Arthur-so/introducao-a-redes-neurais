import matplotlib.pyplot as plt
import numpy as np
import torch

from baseline import EPOCHS, curvas, fmt, metrics, mlp, train
from data import load_splits, Scaler

SEEDS = (0, 1, 2)

CONFIGS = [
    ("baseline", {}),
    ("L1 1e-5", {"l1": 1e-5}),
    ("L1 1e-4", {"l1": 1e-4}),
    ("L1 1e-3", {"l1": 1e-3}),
    ("L2 1e-5", {"l2": 1e-5}),
    ("L2 1e-4", {"l2": 1e-4}),
    ("L2 1e-3", {"l2": 1e-3}),
    ("dropout 0.1", {"dropout": 0.1}),
    ("dropout 0.2", {"dropout": 0.2}),
    ("dropout 0.3", {"dropout": 0.3}),
    ("momentum 0.3", {"momentum": 0.3}),
    ("momentum 0.5", {"momentum": 0.5}),
    ("momentum 0.7", {"momentum": 0.7}),
    ("momentum 0.9", {"momentum": 0.9}),
]


def roda(kw, train_t, val_t, seed):
    torch.manual_seed(seed)
    net = mlp(dropout=kw.get("dropout", 0.0))
    hist = train(net, train_t, val_t, **{k: v for k, v in kw.items() if k != "dropout"})
    return net, hist


def main():
    tr, va, te = load_splits()
    sc = Scaler(tr)
    train_t, val_t = sc.to_tensors(tr), sc.to_tensors(va)

    resultados, historicos = {}, {}
    for nome, kw in CONFIGS:
        execucoes = [roda(kw, train_t, val_t, s) for s in SEEDS]
        historicos[nome] = execucoes[0][1]
        med = {k: np.mean([metrics(n, te, sc)[k] for n, _ in execucoes])
               for k in ("MAE", "MSE", "RMSE", "R2")}
        med["val"] = np.mean([h[:, 1].min() for _, h in execucoes])
        med["epoca"] = np.mean([h[:, 1].argmin() for _, h in execucoes])
        resultados[nome] = med
        print(fmt(nome, med) + f"  val={med['val']:.4f}  epoca*={med['epoca']:.0f}")

    print("\nmelhor de cada familia por MSE de teste:")
    for fam in ("L1", "L2", "dropout", "momentum"):
        cand = {k: v for k, v in resultados.items() if k.startswith(fam)}
        melhor = min(cand, key=lambda k: cand[k]["MSE"])
        delta = cand[melhor]["MSE"] - resultados["baseline"]["MSE"]
        print(f"  {melhor:<14} MSE={cand[melhor]['MSE']:.4f}  ({delta:+.4f} vs baseline)")

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharey=True)
    mostrar = ["baseline", "L1 1e-3", "L2 1e-3", "dropout 0.1", "momentum 0.7", "momentum 0.9"]
    for ax, nome in zip(axes.ravel(), mostrar):
        curvas(ax, historicos[nome], nome)
    fig.tight_layout()
    fig.savefig("ablacao.png", dpi=130)
    print("\ngrafico salvo em ablacao.png")


if __name__ == "__main__":
    main()

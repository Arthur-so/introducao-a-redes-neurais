import argparse
import copy
import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from data import load_splits, Scaler

WIDTH, DEPTH, LR, BATCH, EPOCHS = 64, 3, 0.01, 8, 10000
ATIVACAO = nn.ReLU


def mlp(width=WIDTH, depth=DEPTH, dropout=0.0, act=ATIVACAO):
    layers, d = [], 1
    for _ in range(depth):
        layers += [nn.Linear(d, width), act()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        d = width
    return nn.Sequential(*layers, nn.Linear(d, 1))


def train(net, train_t, val_t, epochs=EPOCHS, lr=LR, batch=BATCH,
          momentum=0.0, l2=0.0, l1=0.0, restaurar_melhor=False):
    """Treina com SGD e devolve o historico (treino, validacao) por epoca.

    Por padrao os pesos entregues sao os da ultima epoca: o orcamento de epocas e
    fixo e igual para todas as configuracoes, para que a comparacao da ablacao nao
    seja contaminada por early stopping (que e, ele proprio, uma regularizacao).
    Com restaurar_melhor=True, restaura os pesos da melhor epoca de validacao."""
    xtr, ytr = train_t
    xva, yva = val_t
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=l2)
    mse = nn.MSELoss()
    hist, best = [], (float("inf"), None)
    for _ in range(epochs):
        net.train()
        for b in torch.randperm(len(xtr)).split(batch):
            loss = mse(net(xtr[b]), ytr[b])
            if l1:
                loss = loss + l1 * sum(p.abs().sum() for p in net.parameters())
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            hist.append((mse(net(xtr), ytr).item(), mse(net(xva), yva).item()))
        if restaurar_melhor and hist[-1][1] < best[0]:
            best = (hist[-1][1], copy.deepcopy(net.state_dict()))
    if restaurar_melhor:
        net.load_state_dict(best[1])
    return np.array(hist)


def metrics(net, split, scaler):
    x, _ = scaler.to_tensors(split)
    net.eval()
    with torch.no_grad():
        pred = scaler.denorm_y(net(x).numpy().ravel())
    true = split[:, 1]
    err = true - pred
    return {
        "MAE": np.abs(err).mean(),
        "MSE": (err ** 2).mean(),
        "RMSE": np.sqrt((err ** 2).mean()),
        "R2": 1 - (err ** 2).sum() / ((true - true.mean()) ** 2).sum(),
    }


def suave(v, k=101):
    return np.convolve(v, np.ones(k) / k, "valid")


def curvas(ax, hist, titulo, ylim=None):
    for col, cor, nome in ((0, "tab:blue", "treino"), (1, "tab:orange", "validacao")):
        ax.plot(hist[:, col], color=cor, alpha=0.15, lw=0.5)
        ax.plot(np.arange(len(suave(hist[:, col]))) + 50, suave(hist[:, col]), color=cor, label=nome)
    if not np.isfinite(hist).all():
        titulo += " (divergiu)"
    ax.set(xlabel="epoca", ylabel="MSE (normalizado)", yscale="log", title=titulo)
    ax.set_xlim(0, len(hist))
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend()


def fmt(name, m):
    return f"{name:<24} " + "  ".join(f"{k}={m[k]:7.4f}" for k in ("MAE", "MSE", "RMSE", "R2"))


def busca(train_t, val_t, epocas):
    """Busca empirica que definiu o baseline: mediana de 3 seeds por configuracao."""
    print(f"{'val':>7} {'par':>5} {'width':>5} {'depth':>5} {'act':>5} {'lr':>5} {'batch':>5}")
    linhas = []
    for depth, width, act, lr, batch in itertools.product(
            (1, 2, 3), (4, 8, 16), (nn.Tanh, nn.ReLU), (0.01, 0.1, 0.3), (8, 30)):
        vals = []
        for seed in range(3):
            torch.manual_seed(seed)
            layers, d = [], 1
            for _ in range(depth):
                layers += [nn.Linear(d, width), act()]
                d = width
            net = nn.Sequential(*layers, nn.Linear(d, 1))
            n_par = sum(p.numel() for p in net.parameters())
            vals.append(train(net, train_t, val_t, epochs=epocas, lr=lr, batch=batch)[:, 1].min())
        linhas.append((float(np.median(vals)), n_par, width, depth, act.__name__, lr, batch))
    for r in sorted(linhas)[:15]:
        print(f"{r[0]:7.3f} {r[1]:5} {r[2]:5} {r[3]:5} {r[4]:>5} {r[5]:5} {r[6]:5}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--busca", action="store_true", help="roda a busca empirica de arquitetura")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epocas", type=int, default=EPOCHS)
    args = ap.parse_args()

    tr, va, te = load_splits()
    sc = Scaler(tr)
    train_t, val_t = sc.to_tensors(tr), sc.to_tensors(va)

    if args.busca:
        busca(train_t, val_t, args.epocas)
        return

    torch.manual_seed(args.seed)
    net = mlp()
    camadas = "-".join(["1"] + [str(WIDTH)] * DEPTH + ["1"])
    print(f"baseline: MLP {camadas} {ATIVACAO.__name__}, "
          f"{sum(p.numel() for p in net.parameters())} parametros, "
          f"SGD lr={LR} batch={BATCH}, {args.epocas} epocas")
    hist = train(net, train_t, val_t, epochs=args.epocas)
    print(f"pesos da ultima epoca (sem early stopping); "
          f"melhor epoca de validacao teria sido a {np.nanargmin(hist[:, 1])}")
    for nome, split in (("treino", tr), ("validacao", va), ("teste", te)):
        print(fmt(nome, metrics(net, split, sc)))

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    curvas(a, hist, "Evolucao do treinamento")
    a.axvline(hist[:, 1].argmin(), color="k", ls=":", lw=1)

    grid = np.linspace(0, 10, 400)
    gx, _ = sc.to_tensors(np.c_[grid, np.zeros_like(grid)])
    with torch.no_grad():
        b.plot(grid, sc.denorm_y(net(gx).numpy().ravel()), "r", lw=2, label="baseline")
    b.scatter(te[:, 0], te[:, 1], s=8, c="0.7", label="teste")
    b.scatter(tr[:, 0], tr[:, 1], s=30, c="tab:blue", label="treino")
    b.set(xlabel="x", ylabel="y", title="Ajuste do baseline")
    b.legend()
    fig.tight_layout()
    fig.savefig("baseline.png", dpi=130)
    print("grafico salvo em baseline.png")


if __name__ == "__main__":
    main()

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from baseline import DEPTH, EPOCHS, WIDTH, curvas, fmt, metrics, mlp, train
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
    ("L2 1e-2", {"l2": 1e-2}),
    ("dropout 0.1", {"dropout": 0.1}),
    ("dropout 0.2", {"dropout": 0.2}),
    ("dropout 0.3", {"dropout": 0.3}),
    ("momentum 0.3", {"momentum": 0.3}),
    ("momentum 0.5", {"momentum": 0.5}),
    ("momentum 0.7", {"momentum": 0.7}),
    ("momentum 0.9", {"momentum": 0.9}),
]


def roda(kw, train_t, val_t, seed, epocas, width, depth):
    torch.manual_seed(seed)
    net = mlp(width=width, depth=depth, dropout=kw.get("dropout", 0.0))
    hist = train(net, train_t, val_t, epochs=epocas,
                 **{k: v for k, v in kw.items() if k != "dropout"})
    return net, hist


PAINEIS = ["baseline", "L2 1e-4", "L2 1e-2", "L1 1e-3", "dropout 0.1", "momentum 0.9"]


def plota(historicos, width, depth, epocas):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharey=True)
    base = historicos["baseline"]
    ylim = (0.5 * np.nanmin(base), 10 * np.nanmax(base[0]))
    for ax, nome in zip(axes.ravel(), PAINEIS):
        curvas(ax, historicos[nome], nome, ylim)
    fig.tight_layout()
    saida = f"ablacao_{width}x{depth}_{epocas}.png"
    fig.savefig(saida, dpi=130)
    print(f"\ngrafico salvo em {saida}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epocas", type=int, default=EPOCHS)
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--replot", action="store_true", help="regera o grafico a partir dos historicos salvos")
    args = ap.parse_args()
    epocas, width, depth = args.epocas, args.width, args.depth

    tr, va, te = load_splits()
    sc = Scaler(tr)
    train_t, val_t = sc.to_tensors(tr), sc.to_tensors(va)

    print(f"rede {width}x{depth}, {epocas} epocas, {len(SEEDS)} seeds por configuracao\n")
    cache = f"historicos_{width}x{depth}_{epocas}.npz"
    if args.replot:
        historicos = dict(np.load(cache))
        plota(historicos, width, depth, epocas)
        return

    resultados, historicos, todos = {}, {}, {}
    for nome, kw in CONFIGS:
        execucoes = [roda(kw, train_t, val_t, s, epocas, width, depth) for s in SEEDS]
        historicos[nome] = execucoes[0][1]
        for i, (_, h) in enumerate(execucoes):
            todos[f"{nome}|seed{i}"] = h
        # so entram na media as execucoes que nao divergiram
        ok = [(n, h) for n, h in execucoes if np.isfinite(h).all()]
        med = {k: np.mean([metrics(n, te, sc)[k] for n, _ in ok]) if ok else float("nan")
               for k in ("MAE", "MSE", "RMSE", "R2")}
        med["val"] = np.mean([h[-500:, 1].mean() for _, h in ok]) if ok else float("nan")
        med["div"] = len(execucoes) - len(ok)
        resultados[nome] = med
        aviso = f"  DIVERGIU em {med['div']}/{len(SEEDS)} seeds" if med["div"] else ""
        print(fmt(nome, med) + f"  val_final={med['val']:.4f}" + aviso)

    print("\nmelhor de cada familia por MSE de teste:")
    for fam in ("L1", "L2", "dropout", "momentum"):
        cand = {k: v for k, v in resultados.items()
                if k.startswith(fam) and np.isfinite(v["MSE"])}
        if not cand:
            print(f"  {fam:<14} todas as configuracoes divergiram")
            continue
        melhor = min(cand, key=lambda k: cand[k]["MSE"])
        delta = cand[melhor]["MSE"] - resultados["baseline"]["MSE"]
        print(f"  {melhor:<14} MSE={cand[melhor]['MSE']:.4f}  ({delta:+.4f} vs baseline)")

    np.savez_compressed(cache, **historicos)
    np.savez_compressed(cache.replace(".npz", "_seeds.npz"), **todos)
    plota(historicos, width, depth, epocas)


if __name__ == "__main__":
    main()

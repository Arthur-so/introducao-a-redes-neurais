import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from baseline import DEPTH, EPOCHS, WIDTH, curvas, fmt, metrics, mlp, suave, train
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
    ("L1+mom", {"l1": 1e-4, "momentum": 0.3}),
    ("L2+mom", {"l2": 1e-5, "momentum": 0.3}),
    ("L1+L2", {"l1": 1e-4, "l2": 1e-5}),
    ("L1+L2+mom", {"l1": 1e-4, "l2": 1e-5, "momentum": 0.3}),
]

# conjunto menor usado no estudo complementar com a rede 128x3
REDUZIDA = [
    ("baseline", {}),
    ("L1 1e-4", {"l1": 1e-4}),
    ("L1 1e-3", {"l1": 1e-3}),
    ("L2 1e-4", {"l2": 1e-4}),
    ("L2 1e-3", {"l2": 1e-3}),
    ("dropout 0.1", {"dropout": 0.1}),
    ("dropout 0.2", {"dropout": 0.2}),
    ("momentum 0.5", {"momentum": 0.5}),
    ("momentum 0.9", {"momentum": 0.9}),
]


def roda(kw, train_t, val_t, seed, epocas, width, depth):
    torch.manual_seed(seed)
    net = mlp(width=width, depth=depth, dropout=kw.get("dropout", 0.0))
    hist = train(net, train_t, val_t, epochs=epocas,
                 **{k: v for k, v in kw.items() if k != "dropout"})
    return net, hist


def epoca_convergencia(hist, tol=1.02, janela=201):
    """Primeira epoca em que a validacao suavizada entra a 2% do seu minimo."""
    s = np.convolve(hist[:, 1], np.ones(janela) / janela, "valid")
    return int(np.argmax(s <= s.min() * tol)) + janela // 2


COMPARAR = ["baseline", "L1 1e-4", "L2 1e-5", "dropout 0.1", "momentum 0.3", "L1+L2+mom"]

PAINEIS = ["baseline", "L2 1e-4", "L2 1e-2", "L1 1e-3", "dropout 0.1", "momentum 0.9"]

COMPARAR_REDUZIDA = ["baseline", "L1 1e-4", "L2 1e-4", "dropout 0.1", "momentum 0.9"]


def plota(historicos, nome_base, epocas, paineis, comparar):
    linhas = -(-len(paineis) // 3)
    fig, axes = plt.subplots(linhas, 3, figsize=(15, 3.5 * linhas), sharey=True)
    base = historicos["baseline"]
    ylim = (0.5 * np.nanmin(base), 10 * np.nanmax(base[0]))
    for ax, nome in zip(axes.ravel(), paineis):
        curvas(ax, historicos[nome], nome, ylim)
    for ax in axes.ravel()[len(paineis):]:
        ax.axis("off")
    fig.tight_layout()
    saida = f"ablacao_{nome_base}.png"
    fig.savefig(saida, dpi=130)

    # comparacao direta das curvas de validacao num unico eixo
    fig2, ax = plt.subplots(figsize=(8, 5))
    for nome in comparar:
        v = historicos[nome][:, 1]
        ax.plot(np.arange(len(suave(v))) + 50, suave(v), lw=1.5,
                label=nome, ls="-" if nome == "baseline" else "--")
    ax.set(xlabel="epoca", ylabel="MSE de validacao (normalizado)", yscale="log",
           title="Validacao: baseline vs. componentes aditivados")
    ax.set_xlim(0, epocas)
    ax.legend(fontsize=9)
    fig2.tight_layout()
    saida2 = f"ablacao_comparacao_{nome_base}.png"
    fig2.savefig(saida2, dpi=130)
    print(f"\ngraficos salvos em {saida} e {saida2}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epocas", type=int, default=EPOCHS)
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--replot", action="store_true", help="regera o grafico a partir dos historicos salvos")
    ap.add_argument("--reduzida", action="store_true", help="usa o conjunto menor de configuracoes")
    args = ap.parse_args()
    epocas, width, depth = args.epocas, args.width, args.depth
    configs = REDUZIDA if args.reduzida else CONFIGS
    paineis = [n for n, _ in REDUZIDA] if args.reduzida else PAINEIS
    comparar = COMPARAR_REDUZIDA if args.reduzida else COMPARAR
    nome_base = f"{width}x{depth}_{epocas}" + ("_reduzida" if args.reduzida else "")

    tr, va, te = load_splits()
    sc = Scaler(tr)
    train_t, val_t = sc.to_tensors(tr), sc.to_tensors(va)

    print(f"rede {width}x{depth}, {epocas} epocas, {len(SEEDS)} seeds por configuracao\n")
    cache = f"historicos_{nome_base}.npz"
    if args.replot:
        historicos = dict(np.load(cache))
        plota(historicos, nome_base, epocas, paineis, comparar)
        return

    resultados, historicos, todos = {}, {}, {}
    for nome, kw in configs:
        execucoes = [roda(kw, train_t, val_t, s, epocas, width, depth) for s in SEEDS]
        historicos[nome] = execucoes[0][1]
        for i, (_, h) in enumerate(execucoes):
            todos[f"{nome}|seed{i}"] = h
        # so entram na media as execucoes que nao divergiram
        ok = [(n, h) for n, h in execucoes if np.isfinite(h).all()]
        med = {k: np.mean([metrics(n, te, sc)[k] for n, _ in ok]) if ok else float("nan")
               for k in ("MAE", "MSE", "RMSE", "R2")}
        med["val"] = np.mean([h[-500:, 1].mean() for _, h in ok]) if ok else float("nan")
        med["conv"] = np.mean([epoca_convergencia(h) for _, h in ok]) if ok else float("nan")
        med["div"] = len(execucoes) - len(ok)
        resultados[nome] = med
        aviso = f"  DIVERGIU em {med['div']}/{len(SEEDS)} seeds" if med["div"] else ""
        print(fmt(nome, med) + f"  val_final={med['val']:.4f}  conv={med['conv']:.0f}" + aviso)

    print("\nmelhor de cada familia por MSE de teste:")
    for fam in ("L1", "L2", "dropout", "momentum", "+"):
        if not any(fam in k if fam == "+" else k.startswith(fam) for k in resultados):
            continue
        if fam == "+":
            cand = {k: v for k, v in resultados.items() if "+" in k and np.isfinite(v["MSE"])}
        else:
            cand = {k: v for k, v in resultados.items()
                    if k.startswith(fam) and "+" not in k and np.isfinite(v["MSE"])}
        if not cand:
            print(f"  {'combinacoes' if fam == '+' else fam:<14} todas as configuracoes divergiram")
            continue
        melhor = min(cand, key=lambda k: cand[k]["MSE"])
        delta = cand[melhor]["MSE"] - resultados["baseline"]["MSE"]
        print(f"  {melhor:<14} MSE={cand[melhor]['MSE']:.4f}  ({delta:+.4f} vs baseline)")

    np.savez_compressed(cache, **historicos)
    np.savez_compressed(cache.replace(".npz", "_seeds.npz"), **todos)
    plota(historicos, nome_base, epocas, paineis, comparar)


if __name__ == "__main__":
    main()

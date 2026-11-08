"""Figuras do projeto: curvas de custo, espaço latente e grades de imagens."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import NOMES_CLASSES, PASTA_FIGURAS

CORES_CLASSES = plt.get_cmap("tab10").colors


def salvar(figura, nome_arquivo):
    PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    figura.savefig(PASTA_FIGURAS / nome_arquivo, dpi=150, bbox_inches="tight")
    plt.close(figura)


def nome_da_topologia(topologia):
    """Ex.: [512, 256] -> '784-512-256'."""
    return "-".join(str(neuronios) for neuronios in [784] + topologia)


def figura_busca_topologia(historicos, nome_arquivo):
    """Custo na validação por época de cada topologia, um painel por dimensão latente.
    historicos[dimensao][nome_topologia] = lista de dicionários por época."""
    figura, eixos = plt.subplots(1, len(historicos), figsize=(12, 4))
    for eixo, (dimensao, por_topologia) in zip(eixos, historicos.items()):
        for nome, historico in por_topologia.items():
            epocas = [linha["epoca"] for linha in historico]
            custos = [linha["custo_validacao"] for linha in historico]
            eixo.plot(epocas, custos, linewidth=2, label=nome)
        eixo.set_title(f"Espaço latente {dimensao}D")
        eixo.set_xlabel("Época")
        eixo.set_ylabel("Custo na validação")
        eixo.grid(alpha=0.3)
        eixo.legend(title="Codificador")
    salvar(figura, nome_arquivo)


def figura_treino_validacao(historicos_escolhidos, nome_arquivo):
    """Custo de treino e de validação por época dos modelos escolhidos."""
    figura, eixos = plt.subplots(1, len(historicos_escolhidos), figsize=(12, 4))
    for eixo, (titulo, historico) in zip(eixos, historicos_escolhidos.items()):
        epocas = [linha["epoca"] for linha in historico]
        eixo.plot(epocas, [linha["custo_treino"] for linha in historico], linewidth=2, label="Treino")
        eixo.plot(epocas, [linha["custo_validacao"] for linha in historico], linewidth=2,
                  label="Validação")
        eixo.set_title(titulo)
        eixo.set_xlabel("Época")
        eixo.set_ylabel("Custo")
        eixo.grid(alpha=0.3)
        eixo.legend()
    salvar(figura, nome_arquivo)


def figura_espaco_latente(pontos, rotulos, titulo, nomes_eixos, nome_arquivo):
    """Pontos 2D pintados pelo rótulo da classe."""
    figura, eixo = plt.subplots(figsize=(7, 6))
    for classe, nome in enumerate(NOMES_CLASSES):
        da_classe = rotulos == classe
        eixo.scatter(pontos[da_classe, 0], pontos[da_classe, 1], s=3, alpha=0.6,
                     color=CORES_CLASSES[classe], label=nome)
    eixo.set_title(titulo)
    eixo.set_xlabel(nomes_eixos[0])
    eixo.set_ylabel(nomes_eixos[1])
    eixo.legend(markerscale=4, loc="center left", bbox_to_anchor=(1, 0.5))
    salvar(figura, nome_arquivo)


def figura_grade_imagens(linhas_de_imagens, nome_arquivo, colunas=None):
    """Uma linha de imagens 28x28 para cada item de linhas_de_imagens (nome -> imagens)."""
    num_linhas = len(linhas_de_imagens)
    num_colunas = len(next(iter(linhas_de_imagens.values())))
    figura, eixos = plt.subplots(num_linhas, num_colunas,
                                 figsize=(num_colunas * 1.1, num_linhas * 1.2), squeeze=False)
    for indice_linha, (nome, imagens) in enumerate(linhas_de_imagens.items()):
        for indice_coluna, imagem in enumerate(imagens):
            eixo = eixos[indice_linha, indice_coluna]
            eixo.imshow(imagem.reshape(28, 28), cmap="gray_r", vmin=0, vmax=1)
            eixo.set_xticks([])
            eixo.set_yticks([])
            if indice_coluna == 0:
                eixo.set_ylabel(nome, rotation=0, ha="right", va="center")
            if indice_linha == 0 and colunas is not None:
                eixo.set_title(colunas[indice_coluna], fontsize=8)
    salvar(figura, nome_arquivo)


def figura_varredura_latente(imagens, limite, nome_arquivo):
    """Imagens decodificadas de uma grade de pontos do espaço latente 2D.
    imagens tem forma (n, n, 784): linha i, coluna j."""
    tamanho_grade = imagens.shape[0]
    mosaico = imagens.reshape(tamanho_grade, tamanho_grade, 28, 28).transpose(0, 2, 1, 3)
    mosaico = mosaico.reshape(tamanho_grade * 28, tamanho_grade * 28)
    figura, eixo = plt.subplots(figsize=(7, 7))
    eixo.imshow(mosaico, cmap="gray_r", extent=[-limite, limite, -limite, limite])
    eixo.set_xlabel("$z_1$")
    eixo.set_ylabel("$z_2$")
    eixo.set_title("Imagens geradas a partir de uma grade do espaço latente 2D")
    salvar(figura, nome_arquivo)


def figura_variancia_pca(variancia_explicada, nome_arquivo):
    """Fração da variância explicada por cada componente principal."""
    componentes = np.arange(1, len(variancia_explicada) + 1)
    figura, eixo = plt.subplots(figsize=(6, 3.5))
    eixo.bar(componentes, variancia_explicada * 100, color=CORES_CLASSES[0])
    eixo.plot(componentes, np.cumsum(variancia_explicada) * 100, "o-", color="black",
              label="Acumulada")
    eixo.set_xlabel("Componente principal")
    eixo.set_ylabel("Variância explicada (%)")
    eixo.set_xticks(componentes)
    eixo.grid(alpha=0.3, axis="y")
    eixo.legend()
    salvar(figura, nome_arquivo)

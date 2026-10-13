"""Figuras: comparação com paleta, painéis das redes, histogramas e hit maps."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import PASTA_FIGURAS

CORES_TAMANHOS = ["#2a78d6", "#eb6834", "#1baf7a"]


def salvar(figura, nome_arquivo):
    """Salva a figura em results/figures e fecha."""
    PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    figura.savefig(PASTA_FIGURAS / nome_arquivo, dpi=150, bbox_inches="tight")
    plt.close(figura)


def mostrar_imagem(eixo, imagem, titulo):
    """Mostra uma imagem RGB sem eixos."""
    eixo.imshow(np.clip(imagem, 0, 1), interpolation="nearest")
    eixo.set_title(titulo, fontsize=10)
    eixo.axis("off")


def desenhar_paleta(eixo, pesos):
    """Faixa de quadrados com as cores dos neurônios (no máximo 32 por linha)."""
    colunas = min(len(pesos), 32)
    linhas = int(np.ceil(len(pesos) / colunas))
    faixa = np.ones((linhas * colunas, 3))
    faixa[: len(pesos)] = pesos
    eixo.imshow(np.clip(faixa.reshape(linhas, colunas, 3), 0, 1), interpolation="nearest")
    eixo.set_xticks(np.arange(-0.5, colunas), minor=True)
    eixo.set_yticks(np.arange(-0.5, linhas), minor=True)
    eixo.grid(which="minor", color="white", linewidth=1.5)
    eixo.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    eixo.set_title("Paleta extraída (centróides)", fontsize=10)


def figura_comparacao(original, quantizada, pesos, nome_arquivo):
    """Original | Quantizada (K=...) com a paleta de centróides abaixo."""
    figura = plt.figure(figsize=(7, 4.6))
    grade = figura.add_gridspec(2, 2, height_ratios=[4, 1])
    mostrar_imagem(figura.add_subplot(grade[0, 0]), original, "Original")
    mostrar_imagem(figura.add_subplot(grade[0, 1]), quantizada, f"Quantizada (K={len(pesos)})")
    desenhar_paleta(figura.add_subplot(grade[1, :]), pesos)
    salvar(figura, nome_arquivo)


def figura_painel_redes(original, quantizadas, nome_arquivo, recorte=None, pixel_destacado=None):
    """Original e as 6 imagens quantizadas: SOM na 1ª linha, GNG na 2ª.

    recorte = (linhas, colunas) em slices para mostrar só uma região ampliada.
    """
    if recorte is None:
        recorte = (slice(None), slice(None))
    figura, eixos = plt.subplots(2, 4, figsize=(10, 5.4))
    mostrar_imagem(eixos[0, 0], original[recorte], "Original")
    eixos[1, 0].axis("off")
    nomes_som = [nome for nome in quantizadas if nome.startswith("SOM")]
    nomes_gng = [nome for nome in quantizadas if nome.startswith("GNG")]
    for coluna, nome in enumerate(nomes_som, start=1):
        mostrar_imagem(eixos[0, coluna], quantizadas[nome][recorte], nome)
    for coluna, nome in enumerate(nomes_gng, start=1):
        mostrar_imagem(eixos[1, coluna], quantizadas[nome][recorte], nome)
    if pixel_destacado is not None:
        for eixo in eixos.ravel():
            if eixo.images:
                destacar_pixel(eixo, pixel_destacado, recorte)
    figura.tight_layout()
    salvar(figura, nome_arquivo)


def destacar_pixel(eixo, pixel, recorte):
    """Desenha um quadrado vazado em volta de um pixel dentro do recorte."""
    linha = pixel[0] - (recorte[0].start or 0)
    coluna = pixel[1] - (recorte[1].start or 0)
    eixo.add_patch(plt.Rectangle((coluna - 1.5, linha - 1.5), 3, 3,
                                 fill=False, edgecolor="white", linewidth=1.2))


def figura_todas_imagens(originais, quantizadas_por_imagem, nome_arquivo):
    """Uma linha por imagem: original e as 6 quantizações."""
    nomes_redes = list(next(iter(quantizadas_por_imagem.values())))
    figura, eixos = plt.subplots(len(originais), 1 + len(nomes_redes),
                                 figsize=(14, 2.1 * len(originais)))
    for linha, nome_imagem in enumerate(originais):
        imagens_da_linha = [originais[nome_imagem]] + [
            quantizadas_por_imagem[nome_imagem][nome] for nome in nomes_redes]
        for eixo, imagem, titulo in zip(eixos[linha], imagens_da_linha, ["Original"] + nomes_redes):
            mostrar_imagem(eixo, imagem, titulo if linha == 0 else "")
        eixos[linha, 0].text(-0.08, 0.5, f"Imagem {linha + 1}", rotation=90, ha="center",
                             va="center", transform=eixos[linha, 0].transAxes)
    figura.tight_layout()
    salvar(figura, nome_arquivo)


def figura_histogramas(erros_por_imagem, nome_arquivo):
    """Histograma do erro por pixel: uma coluna por imagem, SOM em cima e GNG embaixo."""
    figura, eixos = plt.subplots(2, len(erros_por_imagem), figsize=(15, 5.6), sharey=True)
    for coluna, (nome_imagem, erros_por_rede) in enumerate(erros_por_imagem.items()):
        maior_erro = max(erros.max() for erros in erros_por_rede.values())
        faixas = np.linspace(0, maior_erro, 50)
        for linha, modelo in enumerate(["SOM", "GNG"]):
            eixo = eixos[linha, coluna]
            nomes = [nome for nome in erros_por_rede if nome.startswith(modelo)]
            for cor, nome in zip(CORES_TAMANHOS, nomes):
                eixo.hist(erros_por_rede[nome], bins=faixas, histtype="step",
                          linewidth=1.5, color=cor, label=nome)
            eixo.set_yscale("log")
            eixo.grid(alpha=0.25)
            eixo.spines[["top", "right"]].set_visible(False)
            if coluna == 0:
                eixo.set_ylabel("Pixels (escala log)")
                eixo.legend(frameon=False, fontsize=8)
        eixos[0, coluna].set_title(f"Imagem {coluna + 1}", fontsize=10)
        eixos[1, coluna].set_xlabel("Erro do pixel")
    figura.tight_layout()
    salvar(figura, nome_arquivo)


def cor_do_texto(cor_de_fundo):
    """Preto sobre fundos claros e branco sobre fundos escuros."""
    return "black" if cor_de_fundo.mean() > 0.5 else "white"


def desenhar_hit_map_som(eixo, hits, pesos, grade):
    """Grid do SOM: cada célula tem a cor do neurônio e o número de pixels capturados."""
    linhas, colunas = grade
    eixo.imshow(np.clip(pesos.reshape(linhas, colunas, 3), 0, 1), interpolation="nearest")
    tamanho_fonte = 36 / colunas
    for neuronio, quantidade in enumerate(hits):
        linha, coluna = divmod(neuronio, colunas)
        eixo.text(coluna, linha, str(quantidade), ha="center", va="center",
                  fontsize=tamanho_fonte, color=cor_do_texto(pesos[neuronio]))
    eixo.axis("off")


def desenhar_hits_gng(eixo, hits, pesos):
    """Barras com os hits de cada neurônio da GNG (ordenados), pintadas com a cor do neurônio."""
    ordem = np.argsort(hits)[::-1]
    posicoes = np.arange(len(hits))
    eixo.bar(posicoes, hits[ordem], width=1.0, color=np.clip(pesos[ordem], 0, 1),
             edgecolor="#b0b0b0", linewidth=0.3)
    eixo.set_yscale("log")
    eixo.set_xlabel("Neurônios (ordenados por hits)", fontsize=8)
    eixo.set_xticks([])
    eixo.spines[["top", "right"]].set_visible(False)


def figura_hit_maps(modelos, hits_por_rede, nome_arquivo):
    """Hit maps das 6 redes: grids do SOM na 1ª linha e barras da GNG na 2ª."""
    figura, eixos = plt.subplots(2, 3, figsize=(11, 7.4))
    nomes_som = [nome for nome in modelos if nome.startswith("SOM")]
    nomes_gng = [nome for nome in modelos if nome.startswith("GNG")]
    for eixo, nome in zip(eixos[0], nomes_som):
        desenhar_hit_map_som(eixo, hits_por_rede[nome], modelos[nome]["pesos"],
                             modelos[nome]["grade"])
        eixo.set_title(nome, fontsize=10)
    for eixo, nome in zip(eixos[1], nomes_gng):
        desenhar_hits_gng(eixo, hits_por_rede[nome], modelos[nome]["pesos"])
        eixo.set_title(nome, fontsize=10)
    eixos[1, 0].set_ylabel("Pixels capturados (escala log)")
    figura.tight_layout()
    salvar(figura, nome_arquivo)

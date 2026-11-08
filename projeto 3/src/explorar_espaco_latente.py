"""Explora o espaço latente dos modelos escolhidos: gráficos com os rótulos, PCA do 8D,
reconstruções e imagens geradas. Rodar depois de treinar_modelos.py."""
import numpy as np
import pandas as pd
import torch

import graficos
from config import NOMES_CLASSES, PASTA_TABELAS, SEMENTE
from dados import carregar_dados
from treinar_modelos import caminho_do_modelo
from vae import VAE

NUM_AMOSTRAS_GERADAS = 10
TAMANHO_GRADE_VARREDURA = 15
LIMITE_VARREDURA = 3.0


def carregar_modelo(dimensao_latente):
    salvo = torch.load(caminho_do_modelo(dimensao_latente))
    modelo = VAE(salvo["topologia"], salvo["dimensao_latente"])
    modelo.load_state_dict(salvo["pesos"])
    modelo.eval()
    return modelo


def medias_latentes(modelo, imagens):
    """Representação latente de cada imagem: a média mu dada pelo codificador."""
    with torch.no_grad():
        return modelo.codificar(imagens)[0].numpy()


def decodificar(modelo, pontos_latentes):
    with torch.no_grad():
        return modelo.decodificador(torch.tensor(pontos_latentes, dtype=torch.float32)).numpy()


def uso_das_dimensoes(modelo, imagens):
    """Por dimensão latente: variância de mu entre as imagens e sigma médio.
    Uma dimensão sem uso fica igual a N(0, 1): mu perto de 0 para toda imagem e sigma perto de 1."""
    with torch.no_grad():
        media, log_variancia = modelo.codificar(imagens)
    return pd.DataFrame({"dimensao": np.arange(1, media.shape[1] + 1),
                         "variancia_de_mu": media.var(dim=0).numpy(),
                         "sigma_medio": torch.exp(0.5 * log_variancia).mean(dim=0).numpy()})


def pca(pontos):
    """Projeção nas 2 primeiras componentes principais e fração da variância de cada componente."""
    centralizados = pontos - pontos.mean(axis=0)
    _, valores_singulares, componentes = np.linalg.svd(centralizados, full_matrices=False)
    variancia_explicada = valores_singulares ** 2 / np.sum(valores_singulares ** 2)
    return centralizados @ componentes[:2].T, variancia_explicada


def uma_imagem_por_classe(imagens, rotulos):
    """Primeira imagem de cada classe, na ordem dos rótulos."""
    return imagens[[np.flatnonzero(rotulos == classe)[0] for classe in range(len(NOMES_CLASSES))]]


def grade_latente_2d():
    """Pontos de uma grade quadrada no espaço 2D; a linha 0 é a de z2 mais alto."""
    valores = np.linspace(-LIMITE_VARREDURA, LIMITE_VARREDURA, TAMANHO_GRADE_VARREDURA)
    return np.array([[z1, z2] for z2 in valores[::-1] for z1 in valores])


def main():
    imagens_teste, rotulos_teste = carregar_dados()["teste"]
    modelos = {dimensao: carregar_modelo(dimensao) for dimensao in [2, 8]}

    latente_2d = medias_latentes(modelos[2], imagens_teste)
    graficos.figura_espaco_latente(latente_2d, rotulos_teste, "Espaço latente 2D (conjunto de teste)",
                                   ["$z_1$", "$z_2$"], "espaco_latente_2d.png")

    projecao_8d, variancia_explicada = pca(medias_latentes(modelos[8], imagens_teste))
    graficos.figura_espaco_latente(
        projecao_8d, rotulos_teste, "Espaço latente 8D projetado com PCA (conjunto de teste)",
        [f"CP1 ({variancia_explicada[0]:.1%} da variância)",
         f"CP2 ({variancia_explicada[1]:.1%} da variância)"], "espaco_latente_8d_pca.png")
    graficos.figura_variancia_pca(variancia_explicada, "variancia_pca_8d.png")
    pd.DataFrame({"componente": np.arange(1, 9), "variancia_explicada": variancia_explicada,
                  "variancia_acumulada": np.cumsum(variancia_explicada)}
                 ).to_csv(PASTA_TABELAS / "variancia_pca_8d.csv", index=False)
    uso_das_dimensoes(modelos[8], imagens_teste).to_csv(PASTA_TABELAS / "uso_dimensoes_8d.csv",
                                                        index=False)

    originais = uma_imagem_por_classe(imagens_teste, rotulos_teste)
    graficos.figura_grade_imagens(
        {"Original": originais.numpy(),
         "VAE 2D": decodificar(modelos[2], medias_latentes(modelos[2], originais)),
         "VAE 8D": decodificar(modelos[8], medias_latentes(modelos[8], originais))},
        "reconstrucoes.png", colunas=NOMES_CLASSES)

    gerador = np.random.default_rng(SEMENTE)
    graficos.figura_grade_imagens(
        {f"VAE {dimensao}D": decodificar(modelo, gerador.standard_normal((NUM_AMOSTRAS_GERADAS, dimensao)))
         for dimensao, modelo in modelos.items()},
        "amostras_geradas.png")

    imagens_grade = decodificar(modelos[2], grade_latente_2d())
    graficos.figura_varredura_latente(
        imagens_grade.reshape(TAMANHO_GRADE_VARREDURA, TAMANHO_GRADE_VARREDURA, -1),
        LIMITE_VARREDURA, "varredura_latente_2d.png")


if __name__ == "__main__":
    main()

"""Métricas comuns ao SOM e à GNG: BMUs, erro de quantização, erro topológico e hits."""
import numpy as np

"""
Algumas definições de termos usados nas métricas:
- BMU é o neurônio escolhido para representar o pixel, o mais próximo dele (é só um número, um índice).
- Peso é a cor desse neurônio, 3 números (R, G, B). A cor do BMU é pesos[número do BMU].

- A posição da linha de BMU (o índice da linha) é o número do pixel.
- Os valores guardados dentro de cada linha são números de neurônios: o 1º e o 2º BMU daquele pixel.
"""


def primeiro_e_segundo_bmu(pixels, pesos):
    """Índices do 1º e do 2º neurônio mais próximo de cada pixel (n x 2)."""
    distancias = np.linalg.norm(pixels[:, None, :] - pesos[None, :, :], axis=2)
    return np.argsort(distancias, axis=1)[:, :2]


def erro_por_pixel(pixels, pesos, bmus):
    """Distância euclidiana entre cada pixel e o peso do seu BMU."""
    return np.linalg.norm(pixels - pesos[bmus[:, 0]], axis=1)


def erro_quantizacao(erros_dos_pixels):
    """EQ: média da distância de cada pixel ao seu BMU."""
    return erros_dos_pixels.mean()


def erro_topologico(bmus, vizinhos):
    """ET: fração dos pixels cujo 1º e 2º BMU não são vizinhos."""
    sao_vizinhos = vizinhos[bmus[:, 0], bmus[:, 1]]
    return 1.0 - sao_vizinhos.mean()


def contar_hits(bmus, num_neuronios):
    """Quantos pixels cada neurônio captura como 1º BMU."""
    return np.bincount(bmus[:, 0], minlength=num_neuronios)

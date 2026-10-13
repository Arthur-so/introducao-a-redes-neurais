"""Carregar imagens, extrair pixels, quantizar e reconstruir."""
import numpy as np
from PIL import Image

from config import PASTA_IMAGENS


def carregar_imagem(nome):
    """Imagem H x W x 3 com RGB normalizado para [0, 1]."""
    imagem = Image.open(PASTA_IMAGENS / f"{nome}.png").convert("RGB")
    return np.asarray(imagem, dtype=float) / 255.0


def extrair_pixels(imagem):
    """Transforma a imagem H x W x 3 em uma matriz (H*W) x 3."""
    return imagem.reshape(-1, 3)


def quantizar(pesos, bmus):
    """Troca a cor de cada pixel pelo peso do seu BMU."""
    return pesos[bmus[:, 0]]


def reconstruir_imagem(pixels, formato):
    """Remonta a matriz de pixels no formato H x W x 3."""
    return pixels.reshape(formato)

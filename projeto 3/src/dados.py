"""Download e leitura do Fashion-MNIST, com divisão em treino, validação e teste."""
import gzip
import urllib.request

import numpy as np
import torch

from config import PASTA_DADOS, SEMENTE, TAMANHO_VALIDACAO

URL_BASE = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"


def baixar(nome_arquivo):
    """Baixa o arquivo para a pasta de dados, se ainda não estiver lá."""
    caminho = PASTA_DADOS / nome_arquivo
    if not caminho.exists():
        PASTA_DADOS.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL_BASE + nome_arquivo, caminho)
    return caminho


def ler_imagens(nome_arquivo):
    """Imagens 28x28 achatadas em vetores de 784 pixels, normalizados para [0, 1]."""
    with gzip.open(baixar(nome_arquivo)) as arquivo:
        pixels = np.frombuffer(arquivo.read(), dtype=np.uint8, offset=16)
    return torch.tensor(pixels.reshape(-1, 784) / 255.0, dtype=torch.float32)


def ler_rotulos(nome_arquivo):
    """Rótulos de 0 a 9 (ver NOMES_CLASSES em config.py)."""
    with gzip.open(baixar(nome_arquivo)) as arquivo:
        return np.frombuffer(arquivo.read(), dtype=np.uint8, offset=8).astype(int)


def carregar_dados():
    """Dicionário com (imagens, rótulos) de treino, validação e teste."""
    imagens = ler_imagens("train-images-idx3-ubyte.gz")
    rotulos = ler_rotulos("train-labels-idx1-ubyte.gz")
    ordem = np.random.default_rng(SEMENTE).permutation(len(imagens))
    validacao, treino = ordem[:TAMANHO_VALIDACAO], ordem[TAMANHO_VALIDACAO:]
    return {
        "treino": (imagens[treino], rotulos[treino]),
        "validacao": (imagens[validacao], rotulos[validacao]),
        "teste": (ler_imagens("t10k-images-idx3-ubyte.gz"),
                  ler_rotulos("t10k-labels-idx1-ubyte.gz")),
    }

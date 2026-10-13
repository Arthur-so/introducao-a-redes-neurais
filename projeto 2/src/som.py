"""Treino do SOM com a biblioteca MiniSom."""
import numpy as np
from minisom import MiniSom


def vizinhos_na_grade(lado):
    """Matriz booleana: True se dois neurônios são adjacentes no grid (inclui diagonais)."""
    linhas, colunas = np.divmod(np.arange(lado * lado), lado)
    distancia_linhas = np.abs(linhas[:, None] - linhas[None, :])
    distancia_colunas = np.abs(colunas[:, None] - colunas[None, :])
    return np.maximum(distancia_linhas, distancia_colunas) == 1


def treinar_som(pixels, lado, parametros, num_epocas, semente):
    """Treina um SOM lado x lado e devolve pesos, vizinhos e formato do grid."""
    som = MiniSom(
        lado, lado, 3,
        sigma=parametros["sigma0"],
        learning_rate=parametros["eta0"],
        neighborhood_function=parametros["vizinhanca"],
        random_seed=semente,
    )
    som.random_weights_init(pixels)
    # Cada pixel é apresentado num_epocas vezes, em ordem aleatória.
    som.train(pixels, num_epocas * len(pixels), random_order=True)
    return {
        "pesos": som.get_weights().reshape(-1, 3),
        "vizinhos": vizinhos_na_grade(lado),
        "grade": (lado, lado),
    }

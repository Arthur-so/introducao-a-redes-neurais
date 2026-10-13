"""GNG (Growing Neural Gas) implementada à mão, seguindo a Aula 12 (slides 4 a 9).

A rede é um dicionário com:
  pesos    (N x 3)  cor RGB de cada neurônio
  erros    (N)      erro local acumulado de cada neurônio
  conexoes (N x N)  True se existe aresta entre dois neurônios
  idades   (N x N)  idade de cada aresta
"""
import numpy as np

from metricas import primeiro_e_segundo_bmu


def criar_rede(pixels, gerador):
    """Rede inicial: 2 neurônios com as cores de 2 pixels sorteados e nenhuma conexão.

    Pesos sorteados em todo o cubo RGB deixavam neurônios longe dos dados, quase sem uso.
    """
    return {
        "pesos": pixels[gerador.choice(len(pixels), 2, replace=False)].copy(),
        "erros": np.zeros(2),
        "conexoes": np.zeros((2, 2), dtype=bool),
        "idades": np.zeros((2, 2), dtype=int),
    }


def conectar(rede, neuronio_a, neuronio_b):
    """Cria a aresta a–b com idade 0 (se já existe, a idade volta a 0)."""
    rede["conexoes"][neuronio_a, neuronio_b] = rede["conexoes"][neuronio_b, neuronio_a] = True
    rede["idades"][neuronio_a, neuronio_b] = rede["idades"][neuronio_b, neuronio_a] = 0


def desconectar(rede, neuronio_a, neuronio_b):
    """Remove a aresta a–b (neuronio_b pode ser uma máscara de vários neurônios)."""
    rede["conexoes"][neuronio_a, neuronio_b] = False
    rede["conexoes"][neuronio_b, neuronio_a] = False


def remover_neuronios_isolados(rede):
    """Remove os neurônios que ficaram sem nenhuma conexão."""
    manter = rede["conexoes"].any(axis=1)
    rede["pesos"] = rede["pesos"][manter]
    rede["erros"] = rede["erros"][manter]
    rede["conexoes"] = rede["conexoes"][np.ix_(manter, manter)]
    rede["idades"] = rede["idades"][np.ix_(manter, manter)]


def inserir_neuronio(rede, alfa):
    """Insere r entre q (maior erro) e f (vizinho de q com maior erro)."""
    erros = rede["erros"]
    q = np.argmax(erros)
    vizinhos_de_q = np.flatnonzero(rede["conexoes"][q])
    f = vizinhos_de_q[np.argmax(erros[vizinhos_de_q])]

    erros[q] *= 1 - alfa
    erros[f] *= 1 - alfa
    peso_r = (rede["pesos"][q] + rede["pesos"][f]) / 2
    erro_r = (erros[q] + erros[f]) / 2

    rede["pesos"] = np.vstack([rede["pesos"], peso_r])
    rede["erros"] = np.append(erros, erro_r)
    rede["conexoes"] = np.pad(rede["conexoes"], ((0, 1), (0, 1)))
    rede["idades"] = np.pad(rede["idades"], ((0, 1), (0, 1)))

    r = len(rede["erros"]) - 1
    desconectar(rede, q, f)
    conectar(rede, r, q)
    conectar(rede, r, f)


def adaptar(rede, amostra, parametros):
    """Um passo da GNG para uma amostra: conectar, acumular erro, mover e envelhecer arestas."""
    s1, s2 = primeiro_e_segundo_bmu(amostra[None, :], rede["pesos"])[0]
    conectar(rede, s1, s2)
    rede["erros"][s1] += np.sum((amostra - rede["pesos"][s1]) ** 2)

    vizinhos_de_s1 = rede["conexoes"][s1]
    rede["pesos"][s1] += parametros["eta_b"] * (amostra - rede["pesos"][s1])
    rede["pesos"][vizinhos_de_s1] += parametros["eta_n"] * (amostra - rede["pesos"][vizinhos_de_s1])

    rede["idades"][s1, vizinhos_de_s1] += 1
    rede["idades"][vizinhos_de_s1, s1] += 1
    conexoes_velhas = vizinhos_de_s1 & (rede["idades"][s1] > parametros["idade_max"])
    if conexoes_velhas.any():
        desconectar(rede, s1, conexoes_velhas)
        remover_neuronios_isolados(rede)


def treinar_gng(pixels, num_neuronios, parametros, num_epocas, semente):
    """Treina a GNG por num_epocas passagens pelos pixels.

    Insere neurônios a cada lambda passos até atingir num_neuronios; depois só adapta.
    """
    gerador = np.random.default_rng(semente)
    rede = criar_rede(pixels, gerador)
    passo = 0
    for _ in range(num_epocas):
        for indice in gerador.permutation(len(pixels)):
            adaptar(rede, pixels[indice], parametros)
            passo += 1
            if passo % parametros["lambda"] == 0 and len(rede["erros"]) < num_neuronios:
                inserir_neuronio(rede, parametros["alfa"])
            rede["erros"] *= 1 - parametros["beta"]
    return {"pesos": rede["pesos"], "vizinhos": rede["conexoes"], "grade": None}

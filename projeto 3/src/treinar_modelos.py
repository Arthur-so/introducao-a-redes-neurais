"""Busca da melhor topologia para os espaços latentes 2D e 8D, pelo custo na validação.

Uso:  python treinar_modelos.py            (experimento completo)
      python treinar_modelos.py --rapido   (1 época e 5.000 imagens, só para testar o código)
"""
import argparse

import pandas as pd
import torch

import graficos
from config import (DIMENSOES_LATENTES, NUM_EPOCAS, PASTA_MODELOS, PASTA_TABELAS, SEMENTE,
                    TOPOLOGIAS)
from dados import carregar_dados
from vae import VAE, avaliar, treinar


def caminho_do_modelo(dimensao_latente):
    return PASTA_MODELOS / f"vae_latente_{dimensao_latente}d.pt"


def buscar_topologia(dimensao_latente, imagens_treino, imagens_validacao, num_epocas):
    """Treina todas as topologias e devolve as linhas da tabela, os históricos e o melhor modelo."""
    linhas, historicos = [], {}
    melhor_modelo, melhor_custo = None, float("inf")
    for topologia in TOPOLOGIAS:
        nome = graficos.nome_da_topologia(topologia)
        print(f"Latente {dimensao_latente}D: {nome}", flush=True)
        torch.manual_seed(SEMENTE)
        modelo = VAE(topologia, dimensao_latente)
        historicos[nome] = treinar(modelo, imagens_treino, imagens_validacao, num_epocas)
        melhor_epoca = min(historicos[nome], key=lambda linha: linha["custo_validacao"])
        linhas.append({"dimensao_latente": dimensao_latente, "codificador": nome,
                       "parametros": sum(p.numel() for p in modelo.parameters()),
                       **melhor_epoca})
        if melhor_epoca["custo_validacao"] < melhor_custo:
            melhor_custo = melhor_epoca["custo_validacao"]
            melhor_modelo = {"topologia": topologia, "modelo": modelo}
    return linhas, historicos, melhor_modelo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rapido", action="store_true", help="1 época e 5.000 imagens")
    rapido = parser.parse_args().rapido
    num_epocas = 1 if rapido else NUM_EPOCAS

    dados = carregar_dados()
    imagens_treino = dados["treino"][0][:5000] if rapido else dados["treino"][0]
    imagens_validacao = dados["validacao"][0]
    imagens_teste = dados["teste"][0]

    PASTA_TABELAS.mkdir(parents=True, exist_ok=True)
    PASTA_MODELOS.mkdir(parents=True, exist_ok=True)
    linhas_busca, linhas_escolhidos = [], []
    historicos, historicos_escolhidos = {}, {}
    for dimensao in DIMENSOES_LATENTES:
        linhas, historicos[dimensao], melhor = buscar_topologia(
            dimensao, imagens_treino, imagens_validacao, num_epocas)
        linhas_busca += linhas

        nome = graficos.nome_da_topologia(melhor["topologia"])
        historicos_escolhidos[f"Latente {dimensao}D: {nome}"] = historicos[dimensao][nome]
        teste = avaliar(melhor["modelo"], imagens_teste)
        linhas_escolhidos.append({"dimensao_latente": dimensao, "codificador": nome,
                                  "custo_teste": teste["custo"],
                                  "reconstrucao_teste": teste["reconstrucao"],
                                  "kl_teste": teste["kl"]})
        torch.save({"topologia": melhor["topologia"], "dimensao_latente": dimensao,
                    "pesos": melhor["modelo"].state_dict()}, caminho_do_modelo(dimensao))

    pd.DataFrame(linhas_busca).to_csv(PASTA_TABELAS / "busca_topologia.csv", index=False)
    pd.DataFrame(linhas_escolhidos).to_csv(PASTA_TABELAS / "modelos_escolhidos.csv", index=False)
    graficos.figura_busca_topologia(historicos, "busca_topologia.png")
    graficos.figura_treino_validacao(historicos_escolhidos, "treino_validacao.png")


if __name__ == "__main__":
    main()

"""Roda as 30 simulações base e as variações, salvando tabelas CSV e figuras.

Uso:  python executar_experimentos.py            (experimento completo)
      python executar_experimentos.py --rapido   (1 época, só para testar o código)
"""
import argparse

import numpy as np
import pandas as pd

import graficos
from config import (GNG_BASE, IMAGEM_COR_RARA, IMAGENS, IMAGENS_VARIACOES,
                    LADOS_GRADE_SOM, NUM_EPOCAS, NUM_NEURONIOS_GNG, PASTA_TABELAS,
                    PIXEL_COR_RARA, SEMENTE, SOM_BASE, VARIACOES_GNG, VARIACOES_SOM)
from gng import treinar_gng
from imagens import carregar_imagem, extrair_pixels, quantizar, reconstruir_imagem
from metricas import (contar_hits, erro_por_pixel, erro_quantizacao, erro_topologico,
                      primeiro_e_segundo_bmu)
from som import treinar_som

RAIO_RECORTE_PIXEL_RARO = 12


def nome_da_rede(modelo, tamanho):
    """Ex.: 'SOM 8x8' ou 'GNG 64'."""
    return f"SOM {tamanho}x{tamanho}" if modelo == "SOM" else f"GNG {tamanho}"


def treinar(modelo, tamanho, parametros, pixels, num_epocas):
    """Treina um SOM (tamanho = lado do grid) ou uma GNG (tamanho = nº de neurônios)."""
    if modelo == "SOM":
        return treinar_som(pixels, tamanho, parametros, num_epocas, SEMENTE)
    return treinar_gng(pixels, tamanho, parametros, num_epocas, SEMENTE)


def simular(modelo, tamanho, parametros, pixels, num_epocas):
    """Treina a rede e calcula BMUs, erro por pixel, EQ e ET."""
    rede = treinar(modelo, tamanho, parametros, pixels, num_epocas)
    bmus = primeiro_e_segundo_bmu(pixels, rede["pesos"])
    erros = erro_por_pixel(pixels, rede["pesos"], bmus)
    return {
        "rede": rede,
        "bmus": bmus,
        "erros": erros,
        "EQ": erro_quantizacao(erros),
        "ET": erro_topologico(bmus, rede["vizinhos"]),
    }


def redes_base():
    """As 6 redes base: (modelo, tamanho, parâmetros)."""
    redes_som = [("SOM", lado, SOM_BASE) for lado in LADOS_GRADE_SOM]
    redes_gng = [("GNG", num, GNG_BASE) for num in NUM_NEURONIOS_GNG]
    return redes_som + redes_gng


def indice_do_pixel(pixel, largura):
    """Posição (linha, coluna) do pixel na matriz de pixels achatada."""
    return pixel[0] * largura + pixel[1]


def recorte_em_volta(pixel, raio):
    """Slices de uma janela quadrada centrada no pixel."""
    return (slice(pixel[0] - raio, pixel[0] + raio), slice(pixel[1] - raio, pixel[1] + raio))


def cor_em_texto(cor):
    """Cor RGB em [0, 1] escrita como '129 90 81' (escala 0-255)."""
    return " ".join(str(valor) for valor in np.round(cor * 255).astype(int))


def analisar_pixel_raro(imagem, pixels, resultados, quantizadas):
    """Cor recebida pelo pixel de cor rara em cada rede (tabela e figura ampliada)."""
    indice = indice_do_pixel(PIXEL_COR_RARA, imagem.shape[1])
    linhas = []
    for nome, resultado in resultados.items():
        neuronio = resultado["bmus"][indice, 0]
        linhas.append({
            "rede": nome,
            "cor_original_RGB": cor_em_texto(pixels[indice]),
            "cor_do_BMU_RGB": cor_em_texto(resultado["rede"]["pesos"][neuronio]),
            "erro_do_pixel": resultado["erros"][indice],
            "EQ_da_imagem": resultado["EQ"],
            "hits_do_BMU": resultado["hits"][neuronio],
        })
    pd.DataFrame(linhas).to_csv(PASTA_TABELAS / "pixel_raro.csv", index=False)
    graficos.figura_painel_redes(
        imagem, quantizadas, "pixel_raro_zoom.png",
        recorte=recorte_em_volta(PIXEL_COR_RARA, RAIO_RECORTE_PIXEL_RARO),
        pixel_destacado=PIXEL_COR_RARA)


def simular_imagem(nome_imagem, num_epocas):
    """Roda as 6 redes base numa imagem e salva as figuras dessa imagem."""
    imagem = carregar_imagem(nome_imagem)
    pixels = extrair_pixels(imagem)
    resultados, quantizadas, linhas = {}, {}, []
    for modelo, tamanho, parametros in redes_base():
        nome = nome_da_rede(modelo, tamanho)
        print(f"{nome_imagem}: {nome}", flush=True)
        resultado = simular(modelo, tamanho, parametros, pixels, num_epocas)
        pesos = resultado["rede"]["pesos"]
        resultado["hits"] = contar_hits(resultado["bmus"], len(pesos))
        resultados[nome] = resultado
        quantizadas[nome] = reconstruir_imagem(quantizar(pesos, resultado["bmus"]), imagem.shape)
        linhas.append({"imagem": nome_imagem, "rede": nome, "modelo": modelo,
                       "neuronios": len(pesos), "EQ": resultado["EQ"], "ET": resultado["ET"],
                       "neuronios_sem_pixels": int(np.sum(resultado["hits"] == 0)),
                       "fracao_no_neuronio_mais_usado": resultado["hits"].max() / len(pixels),
                       "semente": SEMENTE})
        graficos.figura_comparacao(imagem, quantizadas[nome], pesos,
                                   f"comparacao_{nome_imagem}_{nome.replace(' ', '_')}.png")

    graficos.figura_painel_redes(imagem, quantizadas, f"painel_{nome_imagem}.png")
    graficos.figura_hit_maps({nome: r["rede"] for nome, r in resultados.items()},
                             {nome: r["hits"] for nome, r in resultados.items()},
                             f"hit_map_{nome_imagem}.png")
    if nome_imagem == IMAGEM_COR_RARA:
        analisar_pixel_raro(imagem, pixels, resultados, quantizadas)
    return imagem, resultados, quantizadas, linhas


def experimentos_base(num_epocas):
    """5 imagens x 6 redes: tabela de EQ/ET, figuras por imagem e figuras com todas as imagens."""
    linhas, originais, quantizadas_por_imagem, erros_por_imagem = [], {}, {}, {}
    for nome_imagem in IMAGENS:
        imagem, resultados, quantizadas, linhas_da_imagem = simular_imagem(nome_imagem, num_epocas)
        linhas += linhas_da_imagem
        originais[nome_imagem] = imagem
        quantizadas_por_imagem[nome_imagem] = quantizadas
        erros_por_imagem[nome_imagem] = {nome: r["erros"] for nome, r in resultados.items()}
    graficos.figura_todas_imagens(originais, quantizadas_por_imagem, "todas_as_imagens.png")
    graficos.figura_histogramas(erros_por_imagem, "histogramas.png")
    return pd.DataFrame(linhas)


def experimentos_variacoes(num_epocas):
    """Variações do SOM (vizinhança, sigma0, eta0) e da GNG (eta_b, eta_n, lambda) nas imagens escolhidas."""
    variacoes = [("SOM", LADOS_GRADE_SOM, SOM_BASE, {"base": {}, **VARIACOES_SOM}),
                 ("GNG", NUM_NEURONIOS_GNG, GNG_BASE, {"base": {}, **VARIACOES_GNG})]
    linhas = []
    for nome_imagem in IMAGENS_VARIACOES:
        pixels = extrair_pixels(carregar_imagem(nome_imagem))
        for modelo, tamanhos, parametros_base, mudancas_por_variacao in variacoes:
            for nome_variacao, mudancas in mudancas_por_variacao.items():
                for tamanho in tamanhos:
                    nome = nome_da_rede(modelo, tamanho)
                    print(f"{nome_imagem}: {nome} ({nome_variacao})", flush=True)
                    resultado = simular(modelo, tamanho, {**parametros_base, **mudancas},
                                        pixels, num_epocas)
                    linhas.append({"imagem": nome_imagem, "rede": nome, "modelo": modelo,
                                   "variacao": nome_variacao, "EQ": resultado["EQ"],
                                   "ET": resultado["ET"], "semente": SEMENTE})
    return pd.DataFrame(linhas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rapido", action="store_true", help="1 época, só para testar o código")
    num_epocas = 1 if parser.parse_args().rapido else NUM_EPOCAS

    PASTA_TABELAS.mkdir(parents=True, exist_ok=True)
    experimentos_base(num_epocas).to_csv(PASTA_TABELAS / "resultados_base.csv", index=False)
    experimentos_variacoes(num_epocas).to_csv(PASTA_TABELAS / "variacoes.csv", index=False)


if __name__ == "__main__":
    main()

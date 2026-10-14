"""Configuração única do projeto: imagens, redes, hiperparâmetros e variações."""
from pathlib import Path

PASTA_PROJETO = Path(__file__).resolve().parent.parent
PASTA_IMAGENS = PASTA_PROJETO / "images"
PASTA_FIGURAS = PASTA_PROJETO / "results" / "figures"
PASTA_TABELAS = PASTA_PROJETO / "results" / "tables"

SEMENTE = 42

IMAGENS = [
    "1_lago_floresta",
    "2_criancas_cachorro",
    "3_cidade_noite",
    "4_por_do_sol_praia",
    "5_flores_porta",
]

# Pixel de cor rara usado para ver se cores pouco frequentes sobrevivem à quantização:
# a pessoa no barco, avermelhada, numa imagem quase toda verde-escura.
IMAGEM_COR_RARA = "1_lago_floresta"
PIXEL_COR_RARA = (185, 100)  # (linha, coluna)

IMAGENS_VARIACOES = ["1_lago_floresta", "4_por_do_sol_praia"]

# Número de passagens por todos os pixels, igual para SOM e GNG (não é explorado).
NUM_EPOCAS = 5

LADOS_GRADE_SOM = [4, 8, 16]
NUM_NEURONIOS_GNG = [16, 64, 256]

# Escolhidos num teste com sigma0 em 1, 2, 4 e eta0 em 0,1 e 0,5.
SOM_BASE = {"sigma0": 2.0, "eta0": 0.1, "vizinhanca": "gaussian"}

# Valores mais utilizados para a GNG.
GNG_BASE = {
    "eta_b": 1e-2,
    "eta_n": 1e-4,
    "beta": 0.005,
    "alfa": 0.5,
    "lambda": 50,
    "idade_max": 90,
}

# Cada variação muda só os parâmetros indicados em relação à configuração base.
VARIACOES_SOM = {
    "bubble": {"vizinhanca": "bubble"},
    "sigma0=1": {"sigma0": 1.0},
    "sigma0=4": {"sigma0": 4.0},
    "eta0=0,01": {"eta0": 0.01},
    "eta0=0,5": {"eta0": 0.5},
}

VARIACOES_GNG = {
    "eta_b=0,005": {"eta_b": 0.005},
    "eta_b=0,05": {"eta_b": 0.05},
    "eta_n=0,001": {"eta_n": 1e-3},
    "lambda=25": {"lambda": 25},
    "lambda=100": {"lambda": 100},
}

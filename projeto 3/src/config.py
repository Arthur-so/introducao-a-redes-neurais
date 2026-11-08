"""Configuração única do projeto: caminhos, dados, topologias e hiperparâmetros."""
from pathlib import Path

PASTA_PROJETO = Path(__file__).resolve().parent.parent
PASTA_DADOS = PASTA_PROJETO / "data"
PASTA_FIGURAS = PASTA_PROJETO / "results" / "figures"
PASTA_TABELAS = PASTA_PROJETO / "results" / "tables"
PASTA_MODELOS = PASTA_PROJETO / "results" / "modelos"

SEMENTE = 42

# Os 60.000 exemplos de treino do Fashion-MNIST são divididos em treino e validação.
TAMANHO_VALIDACAO = 10_000

NOMES_CLASSES = ["Camiseta", "Calça", "Suéter", "Vestido", "Casaco",
                 "Sandália", "Camisa", "Tênis", "Bolsa", "Bota"]

DIMENSOES_LATENTES = [2, 8]

# Camadas ocultas do codificador; o decodificador usa as mesmas camadas na ordem inversa.
TOPOLOGIAS = [
    [256],
    [512],
    [256, 128],
    [512, 256],
    [512, 256, 128],
]

NUM_EPOCAS = 30
TAMANHO_LOTE = 128
TAXA_APRENDIZADO = 1e-3

"""VAE com codificador e decodificador MLP, função de custo e treinamento."""
import copy

import torch
from torch import nn

from config import NUM_EPOCAS, TAMANHO_LOTE, TAXA_APRENDIZADO

NUM_PIXELS = 784


def camadas_densas(tamanhos):
    """Camadas Linear + ReLU ligando os tamanhos dados, em sequência."""
    camadas = []
    for entrada, saida in zip(tamanhos[:-1], tamanhos[1:]):
        camadas += [nn.Linear(entrada, saida), nn.ReLU()]
    return nn.Sequential(*camadas)


class VAE(nn.Module):
    def __init__(self, topologia, dimensao_latente):
        super().__init__()
        self.codificador = camadas_densas([NUM_PIXELS] + topologia)
        self.camada_media = nn.Linear(topologia[-1], dimensao_latente)
        # A rede estima log(sigma²), e não sigma, porque o log pode ser qualquer número real.
        self.camada_log_variancia = nn.Linear(topologia[-1], dimensao_latente)
        self.decodificador = nn.Sequential(
            camadas_densas([dimensao_latente] + topologia[::-1]),
            nn.Linear(topologia[0], NUM_PIXELS),
            nn.Sigmoid(),
        )

    def codificar(self, imagens):
        """Média e log da variância da distribuição latente de cada imagem."""
        saida = self.codificador(imagens)
        return self.camada_media(saida), self.camada_log_variancia(saida)

    def forward(self, imagens):
        media, log_variancia = self.codificar(imagens)
        desvio_padrao = torch.exp(0.5 * log_variancia)
        # Truque de reparametrização: z = mu + sigma * ruído, com ruído ~ N(0, I).
        z = media + desvio_padrao * torch.randn_like(desvio_padrao)
        return self.decodificador(z), media, log_variancia


def funcao_custo(imagens, reconstrucoes, media, log_variancia):
    """Termos de reconstrução ||x - d(z)||² e KL(N(mu, sigma), N(0, I)), médios por imagem."""
    reconstrucao = ((imagens - reconstrucoes) ** 2).sum(dim=1).mean()
    kl = (-0.5 * (1 + log_variancia - media ** 2 - log_variancia.exp()).sum(dim=1)).mean()
    return reconstrucao, kl


def avaliar(modelo, imagens):
    """Custo total, termo de reconstrução e termo KL num conjunto inteiro."""
    modelo.eval()
    with torch.no_grad():
        reconstrucao, kl = funcao_custo(imagens, *modelo(imagens))
    return {"custo": (reconstrucao + kl).item(), "reconstrucao": reconstrucao.item(),
            "kl": kl.item()}


def treinar(modelo, imagens_treino, imagens_validacao, num_epocas=NUM_EPOCAS):
    """Treina com Adam e devolve o histórico por época; o modelo fica com os pesos
    da época de menor custo na validação."""
    otimizador = torch.optim.Adam(modelo.parameters(), lr=TAXA_APRENDIZADO)
    historico = []
    melhor_custo, melhores_pesos = float("inf"), None
    for epoca in range(1, num_epocas + 1):
        modelo.train()
        for indices in torch.randperm(len(imagens_treino)).split(TAMANHO_LOTE):
            lote = imagens_treino[indices]
            reconstrucao, kl = funcao_custo(lote, *modelo(lote))
            otimizador.zero_grad()
            (reconstrucao + kl).backward()
            otimizador.step()

        treino = avaliar(modelo, imagens_treino)
        validacao = avaliar(modelo, imagens_validacao)
        historico.append({"epoca": epoca, "custo_treino": treino["custo"],
                          "custo_validacao": validacao["custo"],
                          "reconstrucao_validacao": validacao["reconstrucao"],
                          "kl_validacao": validacao["kl"]})
        if validacao["custo"] < melhor_custo:
            melhor_custo = validacao["custo"]
            melhores_pesos = copy.deepcopy(modelo.state_dict())

    modelo.load_state_dict(melhores_pesos)
    return historico

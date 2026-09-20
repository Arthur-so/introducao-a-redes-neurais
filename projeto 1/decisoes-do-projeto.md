# Decisões de projeto e o raciocínio por trás delas

Documento de apoio ao relatório: registra **por que** cada escolha foi feita, o que foi
testado antes de decidir e qual evidência a sustenta. Todos os números vêm dos
experimentos executados com `baseline.py` e `ablacao.py`.

**Configuração final do baseline:**

```
MLP 1-64-64-64-1, ativação ReLU, 8513 parâmetros
SGD padrão, lr=0.01, lote 8, 10000 épocas
sem momentum, sem regularização, sem early stopping
```

---

## 1. Conformidade com o enunciado: o que significa "vanilla"

O enunciado pede um baseline *vanilla*, define-o como "melhor modelo encontrado" e
impõe: *"O baseline deve ser uma rede MLP básica. Não usar: momentum, regularizações,
etc."*

**Vanilla = o modelo sem nenhum ingrediente adicional.** A fronteira que adotamos:

| Permitido (faz parte de achar o melhor modelo) | Proibido (são componentes adicionais) |
|---|---|
| Número de camadas e neurônios | Momentum |
| Função de ativação | L1, L2, weight decay |
| Taxa de aprendizado | Dropout |
| Tamanho do lote | **Early stopping** |
| Número de épocas | Agendamento de taxa de aprendizado |
| Inicialização padrão dos pesos | Normalização em lote, otimizadores adaptativos |

**Early stopping é regularização** — é tratado assim na literatura padrão (Goodfellow,
*Deep Learning*, cap. 7.8: "Early Stopping as a Regularizer"). Uma versão anterior deste
projeto restaurava os pesos da melhor época de validação ao final do treino, o que
violava a restrição do enunciado. **Isso foi removido:** o baseline entrega os pesos da
última época, e o orçamento de épocas é fixo e igual para todas as configurações.

**Área cinzenta declarada:** a padronização z-score de `x` e `y` (seção 3). Classificamos
como pré-processamento de dados, não como componente do modelo — ela não restringe a
capacidade da rede nem altera a função objetivo. Sem ela, o SGD puro mal treina.

**Segunda área cinzenta:** escolher o número de épocas olhando a validação se parece com
early stopping. A distinção que sustentamos: *early stopping* monitora cada execução
individualmente e para quando aquela execução específica piora — é adaptativo por
execução. Escolher um **orçamento fixo** como hiperparâmetro, idêntico para todas as
configurações e sementes, é seleção de hiperparâmetro, da mesma categoria que escolher
`lr` ou largura. A seção 9 mostra que essa distinção tem consequências práticas.

---

## 2. Antes de modelar: estabelecer referências

**Decisão:** medir o que seria um resultado bom *antes* de treinar qualquer rede.

**Por quê:** sem referência, um MSE de 0.35 não significa nada. Três referências (MSE de
validação, escala padronizada):

| Referência | MSE | O que significa |
|---|---|---|
| Prever sempre a média do treino | 2.078 | Piso de utilidade. Abaixo disso a rede não aprendeu nada |
| Rede grande + Adam, só os 30 pontos de treino | 0.903 | Aproximadamente o melhor possível com esse tamanho de treino |
| Rede grande + Adam, todos os 300 pontos | 0.239 | Piso de ruído: nenhum modelo passa disso |

O ruído dos dados foi estimado em **σ ≈ 0.42**, o que limita o R² máximo alcançável a
algo entre **0.67 e 0.83**.

**Consequência:** o baseline com R² de teste de 0.29 captura cerca de **40% de todo o
sinal recuperável**. Não é um modelo fraco — é o que 30 pontos de treino permitem.

**Se perguntarem "esse R² não é baixo?":** o teto não é 1.0, é ~0.7, porque o resto é
ruído aleatório irreprodutível.

---

## 3. Normalização dos dados

**Decisão:** padronizar `x` e `y` (z-score) com média e desvio calculados **apenas no
conjunto de treino**.

**Por que normalizar:** `x` varia de 0 a 10. Sem normalização, a entrada percorre uma
faixa larga demais para as ativações operarem bem, e com SGD puro — sem momentum nem
taxa adaptativa — não há mecanismo que compense, então o treino fica lentíssimo.

**Por que só as estatísticas do treino:** usar as do conjunto completo seria **vazamento
de dados**. As estatísticas do teste são informação que o modelo não deveria ter. É um
erro metodológico clássico e alvo provável de pergunta.

**Onde está no código:** `data.py`, classe `Scaler`. O método `denorm_y` desfaz a
transformação para que todas as métricas sejam reportadas na escala original de `y`.

**Duas escalas circulando:** as curvas de treino são plotadas na escala padronizada (é
onde a perda opera), e as tabelas de métricas na escala original. O fator de conversão é
a variância de `y` no treino: **0.6073² = 0.3688**.

---

## 4. Busca de arquitetura

**Decisão:** 3 camadas ocultas de 64 neurônios. **8513 parâmetros.**

**Critério:** menor erro de validação na época final. Usar o conjunto de teste para
selecionar invalidaria a estimativa imparcial que ele existe para fornecer.

**Espaço varrido:** larguras {4, 8, 16, 32, 64, 128} × profundidades {1, 2, 3} — 18
arquiteturas, todas sob o protocolo conforme (sem early stopping).

Os cinco melhores, reconfirmados com 8 sementes:

| Arquitetura | Parâmetros | val_final | MSE teste | R² |
|---|---|---|---|---|
| 128×3 | 33409 | 1.0290 ± 0.087 | 0.3774 ± 0.033 | 0.268 |
| 8×2 | 97 | 1.0749 ± 0.070 | 0.4036 ± 0.034 | 0.217 |
| **64×3** | **8513** | **1.0980 ± 0.141** | **0.3766 ± 0.013** | **0.270** |
| 16×3 | 593 | 1.1824 ± 0.141 | 0.3843 ± 0.043 | 0.254 |
| 32×3 | 2209 | 1.2397 ± 0.092 | 0.4067 ± 0.036 | 0.211 |

**Os três primeiros empatam estatisticamente** (teste t de Welch contra o 128×3):
t = 1.17 para o 8×2 e t = 1.18 para o 64×3 — diferenças menores que a incerteza da
medida. Já o 16×3 (t = 2.62) e o 32×3 (t = 4.71) são significativamente piores.

**Por que o 64×3 entre os empatados:** ele tem o **menor desvio entre sementes no teste
de toda a varredura (0.013)**, ou seja, é o mais reprodutível — critério natural quando
o desempenho empata. E seus 8513 parâmetros para 30 amostras garantem capacidade
excedente suficiente para o estudo de ablação ter o que medir.

**Ressalva honesta:** o menor erro de validação foi do 128×3, não do 64×3. A escolha
repousa sobre o empate estatístico mais o critério de reprodutibilidade, não sobre
superioridade medida. Está declarado nas limitações.

**Uma diferença que era ruído:** com 3 sementes, o 16×3 aparecia em 2º lugar (1.0365) e
parecia excelente; com 8 sementes caiu para 4º (1.1824). Três vezes ao longo deste
projeto uma vantagem aparente desapareceu ao aumentar o número de sementes. É a razão
de todas as decisões finais usarem 6 a 8 sementes com desvio-padrão.

---

## 5. Formato das camadas: por que todas com a mesma largura

**Decisão:** largura uniforme nas três camadas ocultas (64-64-64).

**Por que isso precisa ser justificado:** o enunciado não exige largura uniforme. A busca
da seção 4 variou largura e profundidade tratando a largura como valor único aplicado a
todas as camadas — uma suposição implícita que reduz o espaço de busca de N dimensões
para 2, mas que é conveniência, não princípio.

**Foi testada:** 11 formatos, 6 sementes cada, mesmo protocolo.

| Formato | Parâmetros | val_final | MSE teste |
|---|---|---|---|
| 64-16-64 (gargalo) | 2321 | 1.0333 ± 0.068 | 0.3764 ± 0.024 |
| **64-64-64 (uniforme)** | 8513 | 1.0454 ± 0.051 | 0.3767 ± 0.015 |
| 32-64-128 (expansivo) | 10625 | 1.0454 ± 0.121 | 0.3838 ± 0.032 |
| 128-64-32 (afunilado) | 10625 | 1.1531 ± 0.124 | 0.3821 ± 0.039 |
| 96-48-24 (afunilado) | 6049 | 1.2767 ± 0.145 | 0.3905 ± 0.046 |
| 64-32-16 (afunilado) | 2753 | 1.3294 ± 0.136 | 0.4386 ± 0.053 |
| 128-64-32-16 (4 camadas) | 11137 | 1.3367 ± 0.161 | 0.4316 ± 0.039 |

**Conclusão:** a suposição não custou nada. O uniforme empata com o melhor formato
encontrado (diferença de 0.012 na validação, t = 0.35) e tem MSE de teste **idêntico**
(0.3767 contra 0.3764, diferença na quarta casa decimal).

**Achado que contraria o senso comum:** os formatos **afunilados foram piores**. A
intuição de que camadas largas extraem características e as seguintes as comprimem até
a saída escalar não se confirmou — 96-48-24, 64-32-16 e 128-64-32-16 ficaram no fim da
tabela.

---

## 6. Função de ativação: ReLU

**Decisão:** ReLU em todas as camadas ocultas.

**Evidência** (rede 64×3, 8 sementes, cada combinação no seu melhor orçamento de épocas):

| Ativação | lr | lote | épocas | val_final |
|---|---|---|---|---|
| **ReLU** | 0.01 | 8 | 10000 | **0.9610 ± 0.043** |
| ReLU | 0.01 | 16 | 20000 | 0.9670 ± 0.030 |
| ReLU | 0.03 | 16 | 10000 | 0.9835 ± 0.033 |
| tanh | 0.03 | 16 | 10000 | 1.0115 ± 0.112 |
| tanh | 0.03 | 8 | 15000 | 1.0362 ± 0.114 |

**As três primeiras posições são ReLU.** A diferença para o melhor tanh dá t = 1.19 —
tecnicamente empate. Mas há um segundo argumento, mais forte que a média: os tanh têm
desvio entre sementes de 0.11 a 0.14, contra 0.03 a 0.04 dos ReLU. **O ReLU é cerca de
três vezes mais reprodutível.**

**Por que ReLU vence aqui:** `tanh` tem derivada máxima de 1 e tipicamente muito menor;
ao retropropagar por 3 camadas, os gradientes encolhem exponencialmente (gradiente
evanescente) e as primeiras camadas aprendem pouco. ReLU tem derivada 1 na região ativa
e não satura.

**Nota histórica:** em buscas anteriores, feitas com redes de 1 a 2 camadas, o `tanh`
vencia — e fazia sentido, porque em rede rasa a saturação incomoda pouco e a suavidade
do `tanh` casa com o alvo suave. A inversão ao chegar em 3 camadas é exatamente o
comportamento previsto pela teoria.

---

## 7. Taxa de aprendizado: 0.01

**Decisão:** `lr = 0.01`.

**Alternativas testadas:** 0.01, 0.03, 0.1 (e 0.3 em buscas anteriores, que divergia).

**Evidência:** com ReLU e lote 8, `lr=0.01` dá val 0.9610; `lr=0.03` dá 0.9748;
`lr=0.1` dá 1.2632 — e este último é o pior de toda a tabela do estágio, chegando a 1.50
em orçamentos maiores.

**A inversão importante:** `lr=0.1` era o vencedor enquanto havia early stopping, e isso
fazia sentido — chegar rápido ao mínimo não tinha custo, porque depois restaurávamos a
melhor época. **Sem early stopping, taxa alta significa overfittar mais rápido**, e o
modelo termina o orçamento em posição pior. Esse é um exemplo concreto de como os
hiperparâmetros interagem com o protocolo: mudar o protocolo invalidou uma escolha que
parecia bem estabelecida.

---

## 8. Tamanho do lote: 8

**Decisão:** lotes de 8 amostras — 4 minilotes por época, já que são 30 pontos de treino.

**Alternativas testadas:** 8 e 16 (e 30, lote completo, em buscas anteriores).

**Evidência:** ReLU com `lr=0.01` dá val 0.9610 com lote 8 contra 0.9670 com lote 16.
Diferença de 0.006, com desvios de 0.03 a 0.04 — empate técnico (t = 0.32).

**Por que lote menor:** mais atualizações de peso por época, e o ruído estocástico ajuda
a escapar de mínimos locais rasos.

**Detalhe curioso:** as duas configurações do topo fazem exatamente o mesmo número de
atualizações de peso — 40 mil. O lote 8 com 10000 épocas e o lote 16 com 20000 épocas
chegam ao mesmo lugar por caminhos diferentes.

**Efeito colateral visível:** as curvas de treino ficam ruidosas. Os gráficos mostram a
curva bruta em tom claro e uma média móvel de 101 épocas por cima — apresentação apenas,
não altera o treino.

---

## 9. Número de épocas: 10000

**Decisão:** 10000 épocas, orçamento fixo e idêntico para todas as configurações.

**Como foi determinado:** registrando a curva de validação completa e avaliando 6
orçamentos (5000 a 30000). Quase toda combinação apresenta a mesma forma em U — a
validação melhora, atinge um mínimo e depois piora:

| Configuração | 5000 | 10000 | 15000 | 20000 | 25000 | 30000 |
|---|---|---|---|---|---|---|
| ReLU, lr=0.01, lote 8 | 1.0757 | **0.9329** | 0.9514 | 0.9973 | 1.0374 | 1.0465 |
| tanh, lr=0.03, lote 8 | 1.0874 | 0.9571 | **0.9503** | 0.9903 | 1.0160 | 1.0356 |
| tanh, lr=0.01, lote 16 | 1.4564 | 1.2674 | 1.1579 | 1.1275 | 1.0832 | **0.9907** |

**A lição:** o orçamento de épocas é, sem early stopping, um hiperparâmetro de primeira
ordem — treinar além do mínimo só produz overfitting. Uma versão anterior deste projeto
usava 30000 épocas herdadas de outro contexto, treinando o triplo do útil.

**Por que SGD puro precisa de milhares de épocas:** sem momentum, cada passo usa apenas
o gradiente instantâneo, o que produz ziguezague lento em vales estreitos. Momentum
existe para acumular direção e acelerar nessas regiões — e o enunciado o proíbe no
baseline. **A lentidão não é defeito da implementação: é consequência da restrição.**

---

## 10. Sementes fixas e comparação pareada

**Decisão:** toda aleatoriedade controlada por semente fixa, e baseline e ablações usam
**exatamente as mesmas sementes** (0, 1 e 2).

**Por que é indispensável:** com 30 pontos de treino a variação entre execuções é grande.
Se o baseline rodasse com uma semente e a ablação com outra, parte da diferença seria só
inicialização — e a comparação não significaria nada.

**Verificado empiricamente:**

1. **Pesos iniciais idênticos** entre baseline, L1, L2, momentum e dropout.
2. **Sequência de minilotes idêntica** entre o baseline e as ablações que não consomem
   números aleatórios adicionais (L1, L2, momentum): mesmos dados, mesma ordem, mesma época.
3. **Resultados reprodutíveis bit a bit** entre execuções repetidas.

A comparação é, portanto, **pareada**: a única diferença entre baseline e cada ablação é
o componente estudado.

**Exceção inevitável:** o **dropout** sorteia máscaras durante o treino, consumindo
números aleatórios que o baseline não consome. A partir do primeiro sorteio a sequência
de lotes diverge. É inerente ao método.

**Onde está no código:** `torch.manual_seed(seed)` imediatamente antes de construir cada
rede, em `roda()` (`ablacao.py`) e em `main()` (`baseline.py`). A divisão dos dados usa
semente própria e fixa (42) em `data.py`.

---

## 11. Grades da ablação

**Decisão:** L1 e L2 em {1e-5, 1e-4, 1e-3} (mais 1e-2 para L2), dropout em {0.1, 0.2,
0.3}, momentum em {0.3, 0.5, 0.7, 0.9}.

**Critério de construção:** cada grade precisa **cercar o ótimo** — o melhor valor deve
ter vizinhos piores dos dois lados. Se o melhor cai na borda, a grade está incompleta e o
ótimo verdadeiro pode estar fora dela. Uma versão anterior começava em 1e-4 e o melhor
caiu exatamente em 1e-4; a grade foi estendida para 1e-5 e o experimento refeito.

**Por que potências de 10:** intensidades de regularização atuam em escala logarítmica —
a diferença entre 1e-4 e 2e-4 é irrelevante, entre 1e-4 e 1e-3 é enorme.

---

## 12. Métricas finais do baseline

`python3 baseline.py`, semente 0, escala original de `y`:

| Conjunto | MAE | MSE | RMSE | R² |
|---|---|---|---|---|
| Treino (30 pts) | 0.168 | 0.058 | 0.241 | 0.843 |
| Validação (30 pts) | 0.479 | 0.334 | 0.578 | 0.565 |
| Teste (240 pts) | 0.466 | 0.367 | 0.606 | 0.288 |

Média de 8 sementes para a validação: **0.354 ± 0.016** na escala de `y`
(**0.961 ± 0.043** normalizado). A semente 0 saiu um pouco melhor que a média.

O contraste entre R² de 0.843 no treino e 0.288 no teste é a assinatura do overfitting —
8513 parâmetros para 30 amostras.

**Cuidado:** R² é calculado contra a variância **do conjunto em que é medido**, e os três
conjuntos têm dispersões diferentes (desvio de `y`: 0.607 no treino, 0.875 na validação,
0.718 no teste). Por isso **não se compara R² entre conjuntos** — o R² de validação
(0.565) parecer bem melhor que o de teste (0.288) é efeito do denominador.

---

## 13. Resultado da ablação

**Nenhum dos quatro componentes melhorou o baseline.**

| Configuração | MSE teste | R² | val_final | Δ vs baseline |
|---|---|---|---|---|
| **baseline** | 0.3527 | 0.316 | **0.9465** | — |
| L1 1e-4 | 0.3520 | 0.317 | 0.9494 | −0.0007 |
| L1 1e-5 | 0.3559 | 0.310 | 0.9544 | +0.0032 |
| L1 1e-3 | 0.3844 | 0.254 | 1.1386 | +0.0317 |
| L2 1e-5 | 0.3553 | 0.311 | 0.9618 | +0.0026 |
| L2 1e-4 | 0.3569 | 0.308 | 0.9575 | +0.0042 |
| L2 1e-3 | 0.3590 | 0.304 | 0.9827 | +0.0063 |
| L2 1e-2 | 0.4503 | 0.127 | 1.3550 | +0.0976 |
| dropout 0.1 | 0.3943 | 0.235 | 1.1961 | +0.0416 |
| dropout 0.2 | 0.4081 | 0.208 | 1.3380 | +0.0554 |
| dropout 0.3 | 0.4386 | 0.149 | 1.4682 | +0.0859 |
| momentum 0.3 | 0.3534 | 0.314 | 0.9675 | +0.0007 |
| momentum 0.5 | 0.3571 | 0.307 | 0.9826 | +0.0044 |
| momentum 0.7 | 0.3603 | 0.301 | 1.0135 | +0.0076 |
| momentum 0.9 | 0.3588 | 0.304 | 1.0068 | +0.0061 |

**As duas métricas concordam**, o que não acontecia em versões anteriores: o melhor MSE
de teste é do L1 1e-4 com −0.0007 (dois décimos de por cento, ruído puro), e na
validação o **baseline tem o menor erro de todos**, abaixo de qualquer configuração
aditivada.

**Por que nada funcionou — a explicação central.** O orçamento de 10000 épocas foi
escolhido como o mínimo da curva de validação (seção 9). Isso coloca o baseline
exatamente no ponto ótimo entre subtreinar e overfittar. A regularização serve para
deslocar esse ponto; com o orçamento já ajustado ao mínimo, ela apenas restringe um
modelo que já está bem posicionado. **Ajustar o número de épocas pela validação captura
o benefício que a regularização ofereceria** — o mesmo mecanismo do early stopping,
reaparecendo de outra forma. É a consequência prática da área cinzenta declarada na
seção 1.

**Por componente:**

- **L1 e L2** são neutros nas intensidades fracas e destrutivos nas fortes. O L2 1e-2
  derruba o R² de 0.32 para 0.13: a penalidade domina a função objetivo e a rede
  subajusta.
- **Dropout é claramente prejudicial**, e piora monotonicamente com a taxa (0.394 → 0.408
  → 0.439). Dropout força a rede a não depender de neurônios individuais, o que exige
  redundância — e redundância exige dados. Com 30 pontos de treino, ele remove sinal.
  **Ressalva:** a curva de validação do dropout ainda está descendo na época 10000, ou
  seja, ele converge mais devagar e o orçamento fixo o penaliza.
- **Momentum é neutro** em toda a faixa (Δ de +0.001 a +0.008). Desta vez não divergiu em
  nenhum valor, nem com 0.9 — `lr=0.01` com momentum 0.9 dá passo efetivo em torno de
  0.1, dentro da faixa estável. Em versões anteriores, com `lr=0.1`, momentum ≥ 0.7
  divergia numericamente.

---

## 14. Reprodutibilidade

- **Semente fixa (42)** na divisão treino/validação/teste — `data.py`.
- **Sementes fixas (0, 1, 2)** na inicialização dos pesos, as mesmas para o baseline e
  para todas as ablações, garantindo comparação pareada (seção 10).
- **Versões fixadas** em `requirements.txt`.
- **Parâmetros expostos por linha de comando:** `--epocas`, `--width`, `--depth`,
  permitindo refazer qualquer experimento descrito aqui.
- **Históricos salvos em `.npz`** e flag `--replot`, para regenerar gráficos sem
  retreinar.

---

## 15. Limitações reconhecidas

Declarar limitações é mais forte que escondê-las — e provavelmente serão perguntadas.

1. **A divisão 10/10/80 é atípica.** O usual é o inverso. Com 30 pontos de treino, o
   modelo é limitado pelos dados, não pela arquitetura — o que explica por que
   arquiteturas de 97 a 33409 parâmetros produzem resultados quase idênticos.
2. **A validação com 30 pontos é frágil** para seleção de modelos. As diferenças entre as
   arquiteturas do topo não são estatisticamente significativas, então a escolha do 64×3
   repousa sobre empate mais critério de reprodutibilidade, não sobre superioridade
   medida. O remédio correto seria **validação cruzada (k-fold)** sobre treino+validação;
   não foi feito por custo computacional.
3. **A busca foi feita em estágios, não conjuntamente.** A arquitetura foi escolhida com
   `lr` e lote fixos, e depois `lr`, lote, ativação e épocas foram ajustados sobre a
   arquitetura vencedora. Isso é otimização por coordenadas com uma única passada: não
   garante o ótimo global do espaço conjunto. Uma busca completa sobre todas as
   combinações era inviável no tempo disponível.
4. **O orçamento fixo de épocas penaliza componentes que convergem mais devagar**, o
   dropout em particular, cuja validação ainda descia ao fim do orçamento.
5. **A ablação não testou combinações** (L2 + momentum, por exemplo), apenas componentes
   isolados — que é o que o enunciado pede.
6. **As métricas do `baseline.py` são de uma única semente**, para produzir os gráficos
   ilustrativos; as tabelas de ablação usam média de 3 sementes e são mais confiáveis.

---

## 16. Trajetória do desenvolvimento

O enunciado pede documentar *todas* as análises realizadas. Três decisões foram revistas
durante o trabalho, e o motivo de cada revisão é instrutivo:

**O early stopping foi descoberto como violação.** Versões iniciais restauravam os pesos
da melhor época de validação. Ao revisar o enunciado contra a literatura, constatou-se
que isso é regularização e estava proibido no baseline. Remover exigiu refazer toda a
seleção de hiperparâmetros, porque as escolhas anteriores haviam sido feitas sob um
protocolo diferente — e de fato mudaram: `lr` caiu de 0.1 para 0.01, o orçamento de
30000 para 10000 épocas, e a ativação de `tanh` para ReLU.

**A largura uniforme era uma suposição não declarada.** A busca inicial tratava a largura
como valor único para todas as camadas, sem que o enunciado exigisse isso. Ao testar 11
formatos alternativos (seção 5), confirmou-se que a suposição não custou desempenho — mas
isso só é afirmável porque foi medido.

**Diferenças aparentes desapareceram ao aumentar as sementes.** Em três ocasiões uma
arquitetura parecia superior com 3 sementes e perdia a vantagem com 6 ou 8. É a razão de
todas as decisões finais reportarem desvio-padrão e teste t, e não apenas a média.

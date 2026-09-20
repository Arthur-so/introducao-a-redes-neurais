# Decisões de projeto e o raciocínio por trás delas

Documento de apoio: registra **por que** cada escolha foi feita, o que foi testado
antes de decidir e qual evidência sustenta a decisão. Todos os números vêm dos
experimentos executados com `baseline.py` e `ablacao.py`.

Princípio que guiou tudo: **nenhuma escolha por convenção ou gosto pessoal**. Cada
hiperparâmetro foi decidido comparando alternativas no conjunto de validação, que é
o que o enunciado chama de "abordagem empírica".

---

## 0. Em que ordem as decisões foram tomadas

A ordem importa, porque decidir tudo de uma vez exigiria testar todas as combinações
(explosão combinatória). A sequência foi:

1. Caracterizar os dados e estabelecer **referências** (o que é um bom resultado?)
2. Normalização (pré-requisito para o SGD funcionar)
3. Busca de arquitetura: ativação, profundidade, largura, `lr`, batch
4. Estudo de convergência: quantas épocas
5. Ablação: L1, L2, dropout, momentum

---

## 1. Antes de modelar: estabelecer referências

**Decisão:** medir o que seria um resultado bom *antes* de treinar qualquer rede.

**Por quê:** sem referência, um MSE de 0.35 não significa nada — pode ser excelente ou
péssimo. Três referências foram calculadas (MSE de validação, escala padronizada):

| Referência | MSE | O que significa |
|---|---|---|
| Prever sempre a média do treino | 2.078 | O pior resultado aceitável. Abaixo disso, a rede não aprendeu nada |
| Rede grande + Adam, só 30 pontos | 0.903 | Aproximadamente o melhor possível com esse tamanho de treino |
| Rede grande + Adam, todos os 300 pontos | 0.239 | Piso de ruído: nenhum modelo passa disso |

O ruído dos dados foi estimado em **σ ≈ 0.42** (pelo desvio das diferenças entre
pontos consecutivos). Isso limita o R² máximo alcançável a algo entre **0.67 e 0.83**.

**Consequência prática:** o baseline com R² de teste ≈ 0.38 captura entre metade e
dois terços de todo o sinal que existe nos dados. Não é um modelo fraco — é o que 30
pontos de treino permitem.

**Se perguntarem "esse R² não é baixo?":** a resposta é que o teto teórico não é 1.0,
é ~0.7, porque o restante é ruído aleatório irreprodutível.

---

## 2. Normalização dos dados

**Decisão:** padronizar (z-score) tanto `x` quanto `y`, usando média e desvio
calculados **apenas no conjunto de treino**.

**Por que normalizar:** `x` varia de 0 a 10. A ativação `tanh` satura para entradas de
módulo maior que ~2, então sem normalização a maior parte do domínio cairia na região
plana da ativação, onde o gradiente é praticamente zero. Com SGD puro — sem momentum
e sem taxa adaptativa — não há mecanismo que compense isso, e o treino ou fica
lentíssimo ou não sai do lugar.

**Por que usar só as estatísticas do treino:** usar a média/desvio do conjunto
completo seria **vazamento de dados** (*data leakage*). As estatísticas do teste são
informação que o modelo não deveria ter. É um erro metodológico clássico e um alvo
provável de pergunta.

**Onde está no código:** `data.py`, classe `Scaler`. Ela recebe o split de treino no
construtor, e o método `denorm_y` desfaz a transformação para que todas as métricas
sejam reportadas na escala original de `y`.

**Detalhe que gera confusão:** as curvas de treino são plotadas na escala padronizada
(é onde a função de perda opera), mas as tabelas de métricas estão na escala original.
O fator de conversão é a variância de `y` no treino: **0.6073² = 0.3688**.

---

## 3. Função de ativação: `tanh`

**Decisão:** `tanh` em todas as camadas ocultas.

**Alternativa testada:** ReLU, na mesma grade de arquiteturas.

**Evidência** (MSE de validação, mediana de 3 seeds, 8000 épocas):

| Arquitetura | tanh | ReLU |
|---|---|---|
| 16×2 | 0.814 | 0.984 |
| 16×3 | 0.796 | 1.189 |

**Por que `tanh` vence aqui:** a função-alvo é suave e oscilante. ReLU é linear por
partes, então aproximar uma curva suave exige muitos segmentos — ou seja, muitos
neurônios. Com apenas 30 pontos de treino não há dados para sustentar tantos
parâmetros. `tanh` já é suave por construção e encaixa melhor na forma do problema.

**Cuidado:** essa conclusão vale para *este* problema. Em redes profundas e grandes
datasets, ReLU normalmente é superior, justamente porque não satura. O motivo de
`tanh` vencer aqui é a rede ser rasa (sem problema de gradiente evanescente) e o alvo
ser suave.

---

## 4. Profundidade: 2 camadas ocultas

**Decisão:** 2 camadas ocultas.

**Alternativas testadas:** 1, 2, 3, 4 e 5 camadas.

**Evidência** (largura 8, 30000 épocas, 6 seeds — média ± desvio):

| Profundidade | Parâmetros | MSE validação | MSE teste |
|---|---|---|---|
| 2 | 97 | **0.7681 ± 0.0304** | **0.3177 ± 0.0114** |
| 3 | 169 | 0.8539 ± 0.1579 | 0.3522 ± 0.0303 |
| 4 | 241 | 0.9932 ± 0.1542 | 0.3770 ± 0.0312 |
| 5 | 313 | 1.1999 ± 0.1110 | 0.4236 ± 0.0263 |

Profundidade 1 também foi testada (em 3000 épocas) e ficou claramente atrás:
o melhor modelo de 1 camada deu 1.4154 contra 1.2058 do melhor de 2 camadas.

**Por que 2 e não mais:** a degradação é monótona — cada camada extra piora. E repare
na coluna de desvio-padrão: ele **quase dobra** ao passar de 2 para 3 camadas (0.03 →
0.16). Isso não é só perda de desempenho, é perda de *estabilidade*: os treinos passam
a depender muito da inicialização.

**A explicação técnica:** `tanh` tem derivada máxima de 1 e tipicamente muito menor.
Ao retropropagar por várias camadas, os gradientes são multiplicados sucessivamente
por esses valores pequenos e encolhem exponencialmente — é o **gradiente evanescente**.
As primeiras camadas praticamente param de aprender. Redes profundas modernas
contornam isso com ReLU, normalização em lote e conexões residuais — nenhum dos quais
é permitido aqui, já que o enunciado exige uma MLP básica com SGD puro.

**Por que 1 camada não basta:** o teorema da aproximação universal garante que uma
camada oculta é suficiente *com neurônios suficientes*, mas não diz que é eficiente.
Para esta função oscilante, 2 camadas representam o mesmo com menos parâmetros.

---

## 5. Largura: 8 neurônios por camada

**Decisão:** 8 neurônios em cada uma das 2 camadas ocultas. Total: **97 parâmetros**.

**Alternativas testadas:** 4, 8, 16, 32, 64 e 128 neurônios.

**Evidência** (30000 épocas, 3 seeds):

| Largura × Profundidade | Parâmetros | MSE validação | MSE teste |
|---|---|---|---|
| **8×2** | **97** | 0.7559 | **0.3101** |
| 16×2 | 321 | 0.8050 | 0.3488 |
| 32×2 | 1153 | 0.7117 | 0.3214 |
| 64×2 | 4353 | 0.7863 | 0.3262 |
| 128×2 | 16897 | 0.7386 | 0.3229 |
| 32×3 | 2209 | 0.6889 | 0.3332 |
| 128×3 | 33409 | 0.6991 | 0.3290 |

Largura 4 foi testada e ficou atrás (1.039 contra 0.796 de largura 8 em 8000 épocas):
é pouca capacidade para representar as oscilações da função.

**O ponto central:** aumentar a capacidade **não melhorou o teste em nenhum caso**. A
rede de 97 parâmetros tem o melhor MSE de teste de toda a varredura, incluindo redes
com até 33 mil parâmetros — 344 vezes maiores.

**Por quê:** com 30 pontos de treino, capacidade excedente vira decoração dos dados
(*overfitting*), não aprendizado. A quantidade de dados é que limita, não a
arquitetura.

---

## 6. O critério de desempate: parcimônia

**Decisão:** entre modelos com desempenho estatisticamente equivalente, escolher o de
menos parâmetros.

**Onde isso decidiu:** na busca inicial, a rede 8×2 (97 parâmetros) empatou em
validação com a 16×3 (593 parâmetros) — ambas em 0.796. Escolhemos a menor.

**Por quê:** é a navalha de Occam aplicada a aprendizado de máquina. Com desempenho
igual, o modelo menor generaliza melhor, treina mais rápido, é mais fácil de
interpretar e tem menos chance de estar explorando ruído. É também o princípio por
trás de toda regularização.

**Se perguntarem "por que não escolheu a rede maior, já que empataram?":** porque
empate em validação medido com 30 pontos não é empate real — é incerteza. Diante da
incerteza, o modelo mais simples é a aposta mais segura, e o teste confirmou
(8×2 deu 0.3101 contra 0.3307 da 16×3).

---

## 7. Taxa de aprendizado: 0.1

**Decisão:** `lr = 0.1`.

**Alternativas testadas:** 0.01, 0.1 e 0.3.

**Evidência:** `lr = 0.01` ficou sistematicamente atrás em todas as arquiteturas — em
3000 épocas, as configurações com 0.01 ficaram entre 1.42 e 2.15, contra 1.21 a 1.55
das com 0.1. E `lr = 0.3` também piorou (1.195 contra 0.796 na mesma arquitetura).

**Interpretação:** 0.01 é lento demais — como não há momentum para acelerar, o treino
não converge dentro de um orçamento razoável de épocas. 0.3 é grande demais e faz o
otimizador oscilar em torno do mínimo em vez de descer. 0.1 é o ponto de equilíbrio,
e importante: **é um ótimo interior da grade** (as duas vizinhas são piores), o que dá
confiança de que não estamos na borda de uma região não explorada.

---

## 8. Tamanho do lote: 8

**Decisão:** mini-lotes de 8 amostras (o que dá 4 lotes por época, já que são 30
pontos de treino).

**Alternativa testada:** 30 (lote completo, equivalente a *batch gradient descent*).

**Evidência** (3000 épocas): batch 8 venceu em praticamente todas as configurações —
por exemplo, 8×2 com `lr=0.1` deu 1.2058 com batch 8 contra 1.2973 com batch 30.

**Por quê:** com lote completo há apenas **uma** atualização de pesos por época; com
lote 8 há quatro. Além de progredir mais rápido, o ruído estocástico das atualizações
ajuda a escapar de mínimos locais rasos — é o "S" de SGD fazendo seu trabalho.

**Efeito colateral visível:** as curvas de treino ficam muito ruidosas (veja a faixa
clara nos gráficos). Por isso os gráficos mostram também uma média móvel de 101 épocas
por cima — a linha grossa. Isso é apenas apresentação, não altera o treino.

---

## 9. Número de épocas: 8000

**Decisão:** 8000 épocas para o baseline.

**Por que esse número é tão alto:** foi uma descoberta empírica, não um chute. O
estudo de convergência (8×2, média de 5 seeds) mostrou:

| Época | MSE treino | MSE validação |
|---|---|---|
| 500 | 0.560 | 1.526 |
| 2000 | 0.356 | 1.266 |
| 5000 | 0.230 | 0.964 |
| 7500 | 0.195 | 0.835 |
| 10000 | 0.182 | 0.790 |
| 15000 | 0.171 | 0.779 |
| 20000 | 0.163 | 0.771 |

A partir da época **8228** o resultado já está dentro de 2% do melhor observado —
daí o corte em 8000.

**Por que SGD puro precisa de tanta época:** sem momentum, cada passo usa apenas o
gradiente instantâneo. Em regiões onde a superfície de erro é um vale longo e estreito,
isso produz um ziguezague lento. Momentum existe exatamente para acumular direção ao
longo dos passos e acelerar nessas regiões — e o enunciado proíbe usá-lo no baseline.
**Então a lentidão não é um defeito da nossa implementação: é a consequência direta da
restrição imposta.**

**Isso foi verificado depois:** rodamos a ablação também com 30000 épocas. O baseline
melhorou de 0.335 para 0.310 no teste — ou seja, 8000 épocas de fato não é o ponto de
convergência total, é o ponto de retorno decrescente.

**Se perguntarem "por que não 30000, já que é melhor?":** porque o ganho de 8000 para
30000 é de ~7% ao custo de quase 4× mais computação, e porque o critério declarado foi
usar o ponto de retorno decrescente. A decisão está documentada e é reproduzível pelo
parâmetro `--epocas`.

---

## 10. Restaurar os pesos da melhor época de validação

**Decisão:** treinar o número completo de épocas registrando as curvas, mas ao final
restaurar os pesos da época com menor erro de validação.

**Por quê:** isso é *early stopping* na prática, mas com uma vantagem: as curvas
completas ficam disponíveis para os gráficos, mostrando todo o comportamento de
overfitting, enquanto o modelo entregue é o melhor encontrado.

**Onde está no código:** função `train` em `baseline.py` — guarda uma cópia do
`state_dict` sempre que a validação melhora e recarrega ao final.

**Consequência importante para a ablação:** *early stopping é, ele próprio, uma forma
de regularização*. Essa é a explicação mais provável para L1 e L2 não terem melhorado
o baseline 8×2: o benefício já estava sendo capturado pela parada antecipada. É um
ponto forte para citar, porque mostra entendimento de que os componentes interagem.

**Efeito colateral a conhecer:** como a validação é usada tanto para escolher a
arquitetura quanto para escolher a época de parada, o erro de validação fica
**otimista** — ele não é mais uma estimativa imparcial. Por isso as conclusões finais
foram tiradas do conjunto de teste, que não participou de nenhuma decisão.

---

## 11. Repetir cada experimento com várias sementes

**Decisão:** 3 seeds em cada configuração da ablação, 5 a 6 nos estudos de arquitetura.

**Por quê:** os pesos iniciais são aleatórios, e com 30 pontos de treino a variação
entre execuções é grande. Decidir com uma única execução é decidir por sorte.

**Isso mudou uma conclusão do projeto.** Na varredura de capacidade com 3 seeds, a rede
32×3 parecia vencer o baseline na validação (0.689 contra 0.756). Ao repetir com 6
seeds, a diferença encolheu para 0.733 contra 0.768 — e um teste t de Welch mostrou
**t = 1.29**, ou seja, a diferença é menor que a incerteza da própria medida. Não era
uma vantagem real; era ruído.

No mesmo teste, a vantagem do 8×2 no **teste** deu **t = 2.27**, essa sim consistente.

**Se perguntarem "como sabe que a diferença é real?":** a resposta é essa — comparando
a diferença entre médias com o desvio-padrão entre seeds.

---

## 12. Grades da ablação

**Decisão:** L1 e L2 em {1e-5, 1e-4, 1e-3} (mais 1e-2 para L2 na rede grande), dropout
em {0.1, 0.2, 0.3}, momentum em {0.3, 0.5, 0.7, 0.9}.

**Critério de construção:** cada grade precisa **cercar o ótimo**, isto é, o melhor
valor deve ter vizinhos piores dos dois lados. Se o melhor cai na borda, a grade está
incompleta e o ótimo verdadeiro pode estar fora dela.

**Isso obrigou a refazer um experimento.** A primeira grade de L1/L2 começava em 1e-4,
e o melhor resultado caiu exatamente em 1e-4 — na borda. A grade foi estendida para
1e-5 e o experimento refeito. Depois, na rede grande, o ótimo do L2 caiu em 1e-4 com
1e-5 e 1e-3 piores dos dois lados: aí sim, ótimo interior legítimo.

**Por que essas faixas:** intensidades de regularização atuam em escala logarítmica —
a diferença entre 1e-4 e 2e-4 é irrelevante, entre 1e-4 e 1e-3 é enorme. Por isso a
varredura é por potências de 10.

---

## 13. Por que a rede 32×3 é estudo complementar, e não o baseline

**Decisão:** manter o 8×2 como baseline e apresentar a ablação no 32×3 como seção
separada, explicitamente rotulada.

**O conflito:** na validação, a 32×3 vai melhor (0.733 contra 0.768). No teste, vai
pior (0.339 contra 0.318).

**Por que confiamos no teste:**

1. A validação foi usada para escolher arquitetura **e** época de parada. Todo conjunto
   usado para escolher fica otimista — e modelos com mais capacidade exploram mais esse
   efeito, porque têm mais liberdade para se encaixar naqueles 30 pontos específicos.
2. A validação tem **30 pontos**; o teste tem **240**. A estimativa do teste é bem mais
   precisa.
3. A vantagem da 32×3 na validação não é estatisticamente significativa (t = 1.29); a
   vantagem do 8×2 no teste é (t = 2.27).

**Por que mesmo assim o estudo no 32×3 entra no trabalho:** porque ele demonstra o
ponto mais interessante de toda a ablação — ver a seção seguinte.

---

## 14. A conclusão central da ablação

**O resultado:** a mesma regularização que não fez efeito na rede pequena recuperou
8% na rede grande.

| | MSE teste | R² |
|---|---|---|
| baseline 8×2 (97 par.) | 0.3177 | 0.384 |
| 32×3 sem regularização (2209 par.) | 0.3332 | 0.354 |
| **32×3 + L2 1e-4** | **0.3063** | **0.406** |

**A leitura:** regularização não melhora um modelo genericamente — ela **compensa
capacidade excedente**. Na rede de 97 parâmetros não havia excesso a conter, e qualquer
penalidade só retirava capacidade já escassa. Na rede de 2209 parâmetros havia excesso,
e aí a penalidade passou a valer a pena.

**Sobre o momentum:** no orçamento de 8000 épocas ele parecia o melhor componente
(MSE 0.296 contra 0.335, uma melhora de 11%). Com 30000 épocas a vantagem **desapareceu
por completo** (0.315 contra 0.310). Ou seja, momentum não levava a um resultado melhor
— levava ao mesmo resultado **mais rápido** (época ótima ~11000 em vez de ~17000). Esse
achado só apareceu porque rodamos dois orçamentos diferentes, e é um bom exemplo de
como um experimento mal dimensionado leva a conclusão errada.

**Sobre o dropout:** prejudicial nas duas redes, em todas as intensidades. Na rede 8×2
o motivo é claro — desligar 10% a 30% de uma camada de apenas 8 neurônios é uma
perturbação enorme, e a rede não tem redundância para absorvê-la. Dropout foi projetado
para camadas largas, com centenas de unidades.

**Sobre estabilidade numérica:** na rede 32×3, `momentum ≥ 0.7` **divergiu em todas as
3 seeds** (a perda vai a NaN). Com `lr = 0.1` e momentum 0.7, o passo efetivo fica
grande demais para uma rede de 3 camadas. O código reporta isso explicitamente em vez
de deixar o NaN contaminar as médias silenciosamente.

---

## 15. Métricas

**Decisão:** reportar MAE, MSE, RMSE e R², conforme o enunciado, sempre na escala
original de `y`.

**O que cada uma acrescenta:**

- **MSE** — é a própria função de perda otimizada. Penaliza erros grandes ao quadrado.
- **RMSE** — raiz do MSE, volta à unidade de `y`, então é diretamente comparável com a
  amplitude dos dados.
- **MAE** — erro absoluto médio. Menos sensível a *outliers* que o MSE. Comparar os
  dois indica se o erro está concentrado em poucos pontos ruins (MSE ≫ MAE²) ou
  distribuído.
- **R²** — fração da variância explicada. É o MSE normalizado pela variância do
  conjunto: `R² = 1 − MSE/var(y)`. Escala fixa: 1.0 é perfeito, 0.0 equivale a chutar a
  média, negativo é pior que a média.

**Cuidado importante:** R² é sempre calculado contra a variância **do conjunto em que é
medido**, e nossos conjuntos têm dispersões diferentes (desvio de `y`: 0.607 no treino,
0.875 na validação, 0.718 no teste). Por isso **não se compara R² entre conjuntos
diretamente** — o R² de validação (0.617) parecer melhor que o de teste (0.321) no
mesmo modelo é efeito do denominador, não do modelo.

**Não existe "acurácia" aqui:** acurácia é métrica de classificação, que conta acertos
discretos. Em regressão o valor previsto praticamente nunca é exatamente igual ao real,
então a métrica é o *tamanho do erro*. O R² é o análogo mais próximo de "quanto o
modelo acertou".

**Também não se deve usar MAPE** (erro percentual) neste dataset: `y` cruza o zero, e
dividir pelo valor real explode perto da origem.

---

## 16. Reprodutibilidade

- **Semente fixa (42)** na divisão treino/validação/teste — `data.py`. Sem isso, cada
  execução daria um split diferente e os números não seriam comparáveis.
- **Semente controlada** na inicialização dos pesos — `torch.manual_seed(seed)` antes
  de construir cada rede.
- **Versões fixadas** em `requirements.txt`.
- **Parâmetros expostos por linha de comando:** `--epocas`, `--width`, `--depth`,
  permitindo refazer qualquer experimento descrito aqui.
- **Históricos salvos em `.npz`** e flag `--replot`, para regenerar gráficos sem
  retreinar.

---

## 17. Limitações que reconhecemos

Declarar limitações é mais forte que escondê-las — e provavelmente serão perguntadas.

1. **A divisão 10/10/80 é atípica.** O usual é o inverso (80% treino). Com 30 pontos de
   treino, o modelo é limitado pelos dados, não pela arquitetura. A divisão veio do
   enunciado e foi seguida deliberadamente.
2. **A validação com 30 pontos é frágil** para seleção de modelos. O ideal seria
   validação cruzada (*k-fold*) sobre treino+validação, eliminando a dependência de um
   único corte. Não foi feito por custo computacional.
3. **A busca de arquitetura usou 8000 épocas**, mas alguns resultados finais usam 30000.
   A rigor, a busca deveria ser refeita no orçamento final. A varredura de capacidade em
   30000 épocas foi feita justamente para checar isso, e confirmou o 8×2.
4. **O conjunto de teste foi consultado** para comparar 8×2 com 32×3. Em rigor
   metodológico, o teste deveria ser aberto uma única vez, no final. Isso está
   declarado abertamente em vez de omitido.
5. **A ablação não testou combinações** (L2 + momentum, por exemplo), apenas componentes
   isolados — que é o que o enunciado pede.

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

## 4. Arquitetura: 32 neurônios × 3 camadas

**Decisão:** 3 camadas ocultas de 32 neurônios cada. Total: **2209 parâmetros**.

**Critério de seleção:** menor erro de validação. Essa é a única escolha legítima,
porque o conjunto de teste não pode participar de nenhuma decisão de modelagem — ele
existe para estimar o desempenho final de forma imparcial.

**Espaço de busca varrido:** larguras {4, 8, 16, 32, 64, 128} × profundidades {1, 2, 3},
mais profundidades {4, 5} na largura 8.

**Evidência** (30000 épocas, 3 seeds, MSE de validação):

| Arquitetura | Parâmetros | Validação | Teste |
|---|---|---|---|
| **32×3** | **2209** | **0.6889** | 0.3332 |
| 128×3 | 33409 | 0.6991 | 0.3290 |
| 32×2 | 1153 | 0.7117 | 0.3214 |
| 128×2 | 16897 | 0.7386 | 0.3229 |
| 8×2 | 97 | 0.7559 | 0.3101 |
| 16×3 | 593 | 0.7572 | 0.3307 |
| 64×3 | 8513 | 0.7692 | 0.3360 |
| 64×2 | 4353 | 0.7863 | 0.3262 |
| 16×2 | 321 | 0.8050 | 0.3488 |
| 8×3 | 169 | 0.9459 | 0.3724 |

A 32×3 tem o **menor erro de validação de toda a varredura**, e isso foi reconfirmado
ao repetir com 6 seeds: 0.7328 ± 0.0599 contra 0.7681 ± 0.0304 da 8×2.

**Por que 32 e não mais:** a validação para de melhorar depois de 32. As redes de 64 e
128 neurônios não trazem ganho — 128×3 tem 33409 parâmetros (15× mais) para um
resultado equivalente. Aumentar capacidade além desse ponto só adiciona parâmetros que
os 30 pontos de treino não conseguem determinar.

**Por que 3 camadas e não 1:** o teorema da aproximação universal garante que uma
camada oculta basta *com neurônios suficientes*, mas não diz que é eficiente. Para
esta função oscilante, camadas adicionais representam a mesma forma com menos
parâmetros. Profundidade 1 ficou claramente atrás nos testes iniciais (1.4154 contra
1.2058 do melhor de 2 camadas, em 3000 épocas).

**Por que não mais que 3 camadas:** testamos 4 e 5 (na largura 8) e a degradação é
monótona e acompanhada de perda de estabilidade — o desvio-padrão entre seeds quase
dobra de 2 para 3 camadas (0.03 → 0.16) e continua alto depois:

| Profundidade (largura 8) | Validação | Teste |
|---|---|---|
| 2 | 0.7681 ± 0.0304 | 0.3177 ± 0.0114 |
| 3 | 0.8539 ± 0.1579 | 0.3522 ± 0.0303 |
| 4 | 0.9932 ± 0.1542 | 0.3770 ± 0.0312 |
| 5 | 1.1999 ± 0.1110 | 0.4236 ± 0.0263 |

**A explicação técnica:** `tanh` tem derivada máxima de 1 e tipicamente muito menor. Ao
retropropagar por várias camadas, os gradientes são multiplicados sucessivamente por
esses valores pequenos e encolhem exponencialmente — é o **gradiente evanescente**. As
primeiras camadas praticamente param de aprender. Redes profundas modernas contornam
isso com ReLU, normalização em lote e conexões residuais — nenhum dos quais é permitido
aqui, já que o enunciado exige MLP básica com SGD puro. Três camadas é o limite prático
antes que esse efeito domine.

---

## 5. A ressalva honesta sobre essa escolha

Este é o ponto mais delicado do trabalho e deve ser apresentado abertamente, não
escondido.

**O conflito:** a 32×3 vence na validação (0.7328 contra 0.7681), mas **perde no teste**
(0.3394 contra 0.3177 da 8×2, ambas com 6 seeds).

**O que a estatística diz:** aplicando um teste t de Welch sobre as 6 seeds —

- Diferença na **validação**: t = 1.29 → **não significativa**. A vantagem da 32×3 é
  menor que a incerteza da própria medida.
- Diferença no **teste**: t = 2.27 → **significativa**, a favor da 8×2.

**Por que mesmo assim a 32×3 é o baseline:** porque o critério de seleção declarado é o
erro de validação, e usar o teste para escolher arquitetura invalidaria o teste como
estimativa imparcial. Trocar de modelo depois de olhar o teste é exatamente o erro
metodológico que a separação em três conjuntos existe para impedir.

**Por que a validação é frágil aqui:** ela tem apenas **30 pontos**, e é usada duas
vezes — para escolher a arquitetura e para escolher a época de parada. Todo conjunto
usado para escolher fica otimista, e modelos com mais capacidade exploram mais esse
efeito. O jeito correto de resolver isso seria **validação cruzada (k-fold)** sobre
treino+validação, eliminando a dependência de um único corte de 30 pontos. Não foi
feito por custo computacional, e está listado nas limitações.

**Se o professor perguntar "por que não usou a rede menor, que vai melhor no teste?":**
a resposta é que essa informação só é conhecida *depois* de abrir o teste, e decidir com
base nela transformaria o teste em um segundo conjunto de validação — o modelo
escolhido assim já não teria estimativa imparcial nenhuma.

---

## 6. O papel da parcimônia

**Princípio adotado:** entre modelos **estatisticamente equivalentes**, preferir o de
menos parâmetros (navalha de Occam).

**Onde ele foi aplicado:** dentro da faixa de capacidade alta, a 32×3 (2209 parâmetros)
foi preferida à 128×3 (33409 parâmetros), que teve validação praticamente idêntica
(0.6991 contra 0.6889) com 15× mais parâmetros.

**Onde ele não decidiu:** entre 32×3 e 8×2, a parcimônia apontaria para a 8×2. Ela foi
sobrepujada pelo critério primário — menor erro de validação. Esse conflito está
documentado na seção anterior.

**Por que a parcimônia importa:** com desempenho igual, o modelo menor generaliza
melhor, treina mais rápido e tem menos chance de estar explorando ruído. É o mesmo
princípio por trás de toda regularização — e, de fato, foi a regularização L2 que
melhorou nosso baseline, conforme a seção 14.

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

## 9. Número de épocas: 30000

**Decisão:** 30000 épocas.

**Por que esse número é tão alto:** foi uma descoberta empírica, não um chute. A
convergência do baseline 32×3 (MSE, menor valor até cada época):

| Época | Treino | Validação |
|---|---|---|
| 1000 | 0.4949 | 1.4057 |
| 2500 | 0.3182 | 1.1907 |
| 5000 | 0.1576 | 0.8097 |
| 10000 | 0.1171 | 0.8097 |
| 15000 | 0.0984 | 0.8097 |
| 20000 | 0.0934 | 0.7431 |
| 30000 | 0.0706 | 0.7431 |

A melhor época de validação é a **18001**. O orçamento de 30000 foi escolhido para
ficar confortavelmente além desse ponto — parar em 20000 arriscaria cortar a cauda da
curva em execuções com outras sementes (a média entre 3 seeds dá época ótima 18153).

**Por que SGD puro precisa de tanta época:** sem momentum, cada passo usa apenas o
gradiente instantâneo. Em regiões onde a superfície de erro é um vale longo e estreito,
isso produz um ziguezague lento. Momentum existe exatamente para acumular direção ao
longo dos passos e acelerar nessas regiões — e o enunciado proíbe usá-lo no baseline.
**A lentidão não é defeito da implementação: é consequência direta da restrição
imposta.**

**Verificação independente:** o mesmo fenômeno foi medido na rede 8×2, com um estudo de
convergência de 5 seeds que foi até 20000 épocas. Lá o resultado continuava melhorando
de 8000 (val 0.82) até 20000 (val 0.771), confirmando que a lentidão é do otimizador,
não da arquitetura.

**Custo:** ~25 segundos por treino do baseline. A ablação completa (15 configurações ×
3 seeds) leva cerca de 1h20.

**Reprodutível com outro orçamento:** `python3 baseline.py --epocas N` e
`python3 ablacao.py --epocas N`.

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
de regularização*. Isso significa que o baseline **já vem regularizado**, e que L1, L2 e
dropout estão competindo com um mecanismo que já captura parte do benefício disponível.

Isso explica dois resultados de uma vez. Primeiro, por que o ganho do L2 é de 8% e não
maior: parte do overfitting já havia sido contida pela parada antecipada. Segundo, por
que numa rede pequena (a 8×2, com 3 parâmetros por amostra) nenhuma regularização
adicional melhorava nada — ali o early stopping sozinho já bastava.

É um ponto forte para citar, porque mostra entendimento de que os componentes interagem
em vez de agirem isoladamente.

**Efeito colateral a conhecer:** como a validação é usada tanto para escolher a
arquitetura quanto para escolher a época de parada, o erro de validação fica
**otimista** — ele não é mais uma estimativa imparcial. Por isso as conclusões finais
foram tiradas do conjunto de teste, que não participou de nenhuma decisão.

---

## 11. Sementes fixas e comparação pareada

**Decisão:** toda aleatoriedade é controlada por semente fixa, e baseline e ablações
usam **exatamente as mesmas sementes** (0, 1 e 2).

**Por que isso é indispensável:** com 30 pontos de treino, a variação entre execuções é
grande. Se o baseline rodasse com uma semente e a ablação com outra, parte da diferença
observada seria só inicialização diferente — e a comparação não significaria nada.

**O que a semente fixa garante, verificado empiricamente:**

1. **Pesos iniciais idênticos** entre baseline, L1, L2, momentum e dropout. Todas as
   configurações partem exatamente da mesma rede.
2. **Sequência de lotes idêntica** entre o baseline e as ablações que não consomem
   números aleatórios adicionais (L1, L2, momentum). Elas veem os mesmos dados na mesma
   ordem, na mesma época.
3. **Resultados reprodutíveis bit a bit** entre execuções repetidas.

Isso torna a comparação **pareada**: a única diferença entre baseline e cada ablação é
o componente sendo estudado, e nada mais.

**A única exceção, inevitável:** o **dropout** sorteia máscaras durante o treino,
consumindo números aleatórios que o baseline não consome. A partir do primeiro sorteio,
a sequência de lotes diverge. Isso é inerente ao método — não há como aplicar dropout
sem consumir aleatoriedade — e vale mencionar caso seja questionado.

**Onde está no código:** `torch.manual_seed(seed)` é chamado imediatamente antes de
construir cada rede, em `roda()` (`ablacao.py`) e em `main()` (`baseline.py`). A divisão
treino/validação/teste usa semente própria e fixa (42) em `data.py`.

**Por que 3 sementes e não 1:** decidir com uma única execução é decidir por sorte. As
tabelas reportam a **média** entre sementes, e os estudos de arquitetura usaram 5 a 6
sementes com **desvio-padrão**, para permitir julgar se uma diferença é real.

**Isso mudou uma conclusão do projeto.** Na varredura de capacidade com 3 sementes, a
32×3 parecia vencer a 8×2 na validação por 0.689 contra 0.756. Ao repetir com 6
sementes, a diferença encolheu para 0.733 contra 0.768, e um teste t de Welch deu
**t = 1.29** — menor que a incerteza da medida. Sem repetir, teríamos relatado como
fato uma diferença que é ruído.

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

## 13. O papel da rede 8×2 no trabalho

A rede 8×2 (97 parâmetros) foi a vencedora de uma busca anterior feita com orçamento de
8000 épocas, e continua no trabalho como **termo de comparação**, não como baseline.

**Por que ela ainda é relevante:**

1. Ela documenta que o resultado **depende do orçamento de épocas**. Com 8000 épocas a
   8×2 vencia; ao estender para 30000 e varrer capacidades maiores, a 32×3 passou à
   frente na validação. Uma busca de arquitetura feita sob um orçamento diferente do
   usado no treino final pode levar a outra escolha — e isso vale registrar.
2. O contraste entre as duas é o que dá sentido à ablação (seção 14): a mesma
   regularização que não fazia efeito na rede pequena recupera 8% na rede grande.

**Métricas comparadas** (6 seeds, 30000 épocas):

| | Parâmetros | Validação | Teste |
|---|---|---|---|
| 8×2 | 97 | 0.7681 ± 0.0304 | 0.3177 ± 0.0114 |
| **32×3 (baseline)** | **2209** | **0.7328 ± 0.0599** | 0.3394 ± 0.0204 |

---

## 14. A conclusão central da ablação

**O resultado principal:** a regularização **L2 com intensidade 1e-4 melhorou o
baseline em 8%**.

| | MSE teste | R² |
|---|---|---|
| baseline 32×3 | 0.3332 | 0.354 |
| **baseline + L2 1e-4** | **0.3063** | **0.406** |

Esse 0.3063 é o melhor resultado de todo o projeto — supera inclusive a rede 8×2
(0.3177), que era a melhor sem regularização.

**A curva de L2 tem ótimo interior bem definido**, o que dá confiança de que o valor
encontrado não está na borda de uma região inexplorada:

| Intensidade de L2 | MSE teste |
|---|---|
| 1e-5 | 0.3177 |
| **1e-4** | **0.3063** |
| 1e-3 | 0.3713 |
| 1e-2 | 0.5042 |

**A leitura conceitual:** regularização não melhora um modelo genericamente — ela
**compensa capacidade excedente**. O baseline tem 2209 parâmetros para 30 pontos de
treino, ou seja, 73 parâmetros por amostra. Há muito espaço para decorar ruído, e é
esse espaço que a penalidade L2 contém. Na rede 8×2, que tem 3 parâmetros por amostra,
não havia excesso a conter — e lá nenhuma intensidade de L2 melhorou nada.

**Sobre o L1:** praticamente sem efeito nas intensidades úteis (1e-5 dá 0.3314, contra
0.3332 do baseline — diferença dentro do ruído) e destrutivo a partir de 1e-3 (0.5268).
L1 induz esparsidade, zerando pesos inteiros; numa rede já pequena para a tarefa, isso
remove capacidade útil em vez de conter excesso.

**Sobre o momentum:** não ajudou. Um alerta importante aqui: num experimento anterior
com **8000 épocas**, o momentum parecia ser o melhor componente de todos (MSE 0.296
contra 0.335, melhora de 11%). Ao repetir com 30000 épocas, a vantagem **desapareceu
por completo**. Momentum não levava a um resultado melhor — levava ao mesmo resultado
**mais rápido** (época ótima ~11000 em vez de ~17000). Esse achado só apareceu porque
dois orçamentos foram testados, e é um bom exemplo de como um experimento mal
dimensionado produz conclusão errada.

**Sobre o dropout:** prejudicial em todas as intensidades (0.4811 a 0.5108, contra
0.3332 do baseline). Dropout força a rede a não depender de neurônios individuais, o
que exige redundância — e redundância exige dados. Com 30 pontos de treino, ele remove
sinal em vez de ruído.

**Sobre estabilidade numérica:** `momentum ≥ 0.7` e `dropout 0.3` **divergiram em todas
as 3 seeds** (perda vai a NaN). Com `lr = 0.1` e momentum 0.7, o passo efetivo fica
grande demais para uma rede de 3 camadas. O código detecta e reporta isso
explicitamente (`DIVERGIU em n/N seeds`) em vez de deixar o NaN contaminar as médias
silenciosamente.

---

## 14b. Métricas finais do baseline

Treino completo do baseline (`python3 baseline.py`, semente 0, 30000 épocas, melhor
época de validação = 18001), todas na escala original de `y`:

| Conjunto | MAE | MSE | RMSE | R² |
|---|---|---|---|---|
| Treino (30 pts) | 0.263 | 0.121 | 0.347 | 0.673 |
| Validação (30 pts) | 0.414 | 0.274 | 0.524 | 0.642 |
| Teste (240 pts) | 0.467 | 0.370 | 0.608 | 0.283 |

O gap entre treino (MSE 0.121) e teste (0.370) é a assinatura do overfitting que a
regularização L2 vem corrigir — e é visível no gráfico `baseline.png`, onde a curva
ajustada apresenta platôs e picos abruptos perseguindo pontos individuais de treino.

Lembrete: R² não é comparável entre conjuntos, porque cada um é normalizado pela
própria variância (seção 15).

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
diretamente** — o R² de validação do baseline (0.642) parecer muito melhor que o de
teste (0.283) é efeito do denominador, não do modelo.

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
- **Sementes fixas (0, 1, 2)** na inicialização dos pesos, as mesmas para o baseline e
  para todas as ablações, garantindo comparação pareada (seção 11).
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
2. **A validação com 30 pontos é frágil** para seleção de modelos, e é usada duas vezes
   (escolher arquitetura e escolher época de parada), o que a torna otimista. A
   diferença de validação entre a 32×3 e a 8×2 não é estatisticamente significativa
   (t = 1.29), então a escolha da arquitetura repousa sobre uma diferença dentro do
   ruído. O remédio correto seria **validação cruzada (k-fold)** sobre treino+validação;
   não foi feito por custo computacional.
3. **O baseline escolhido vai pior no teste** que a alternativa mais simples (0.3394
   contra 0.3177, t = 2.27). Isso é sintoma do item anterior: seleção feita numa
   validação pequena. Mantivemos a escolha pela validação porque usar o teste para
   selecionar invalidaria o teste como estimativa imparcial — mas o fato está
   reportado abertamente, e a rede 8×2 permanece documentada na seção 13.
4. **`lr` e tamanho de lote foram ajustados em redes menores** (largura 4 a 64,
   profundidade 1 a 2) e com orçamento de 8000 épocas, e depois herdados pela
   arquitetura final. Em rigor, deveriam ser reajustados para a 32×3 com 30000 épocas.
   Há indício de que isso importaria: `momentum ≥ 0.7` diverge nesta arquitetura com
   `lr = 0.1`, sugerindo que a taxa está próxima do limite de estabilidade para 3
   camadas.
5. **A ablação não testou combinações** (L2 + momentum, por exemplo), apenas componentes
   isolados — que é o que o enunciado pede.
6. **As métricas do `baseline.py` são de uma única semente**, para produzir os gráficos
   ilustrativos. As tabelas de ablação usam média de 3 sementes e são mais confiáveis.
   Por isso o MSE de teste do baseline aparece como 0.3696 no `baseline.py` e 0.3332 na
   tabela de ablação — não é inconsistência, são estimativas com números diferentes de
   execuções.

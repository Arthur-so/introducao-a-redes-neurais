• Tarefa de Regressão
    Utililizar o dataset compartilhado

• Efetuar a separação do conjunto em treino (10%), validação(10%) e teste (80%)

• Usando uma abordagem empírica, gerar um modelo, denominado baseline (vanilla - melhor modelo encontrado)

• Sobre o baseline:
    Usar o método padrão SGD (NÃO USAR ALGORITMOS OTIMIZADOS, e.g. Adam)
    O baseline deve ser uma rede MLP básica. Não usar: momentum, regularizações, etc.

• Estudos de Ablação:
    Avaliar o baseline com os seguintes componentes: L1, L2, dropout, momentum


• Considerar:
    Ilustrar graficamente a evolução do treinamento (treino/validação)
    Documentar todas as análises realizadas durante o desenvolvimento do baseline
    Os modelos “aditivados” não devem sofrer alterações em suas arquiteturas
    Considerar várias métricas de avaliação, MAE, MSE, RMSE e R2.

• Entregável:
    Relatório curto descrevendo todo o estudo
    Código em python (pytorch)
    Obs. O relatório não é um caderno colab (Jupyter)
    Código e relatório são arquivos separados
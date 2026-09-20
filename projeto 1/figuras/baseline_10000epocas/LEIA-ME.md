# Figuras do relatório — baseline com 10.000 épocas

Estas são as figuras usadas no `relatorio.tex`, correspondentes à configuração
final do baseline vanilla:

```
MLP 1-64-64-64-1, ReLU, 8513 parâmetros
SGD puro, lr=0.01, lote 8, 10000 épocas, sem early stopping
```

| Arquivo | Conteúdo | Figura no relatório |
|---|---|---|
| `baseline.png` | Curvas de treino/validação e ajuste sobre o teste | Figura 1 |
| `ablacao_comparacao_64x3_10000.png` | Validação sobreposta: baseline vs. componentes | Figura 2 |
| `ablacao_64x3_10000.png` | Curvas por componente, 6 painéis | Figura 3 |
| `baseline_residuos.png` | Paridade e resíduos no teste | Figura 4 |

Cópia de segurança: `baseline.png` e `baseline_residuos.png` têm nome fixo e são
sobrescritos a cada execução do `baseline.py` com outro orçamento de épocas.

Reproduzir: `python3 baseline.py` e `python3 ablacao.py` com os padrões do código.

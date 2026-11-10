---
name: documentacao-didatica-codigo
description: Use quando o usuário pedir para explicar, documentar ou estudar um projeto de código linha por linha, para aprender como ele funciona.
---

# Guia: documentação didática de código (linha por linha)

## Objetivo

Você vai escrever uma documentação para **uma pessoa aprender** como um projeto funciona, não para servir de referência a quem já o conhece. O leitor quer entender cada linha, e ao terminar deve ser capaz de explicar o projeto com as próprias palavras e de modificá-lo.

Documentações de código costumam falhar em quatro pontos. Este guia trata de cada um:

1. **Ordem ruim**: segue a ordem dos arquivos ou a ordem alfabética, e não a ordem em que o leitor consegue entender.
2. **Falta de exemplos**: explica em abstrato, sem mostrar valores reais passando pelo código.
3. **Falta de conexão**: cada trecho é explicado isolado, e o leitor não vê como as partes se encaixam.
4. **Pressupõe conhecimento prévio**: usa termos e conceitos sem explicá-los.

---

## Fase 0: calibrar antes de escrever

Antes de qualquer coisa, pergunte ao usuário (de uma vez só, em poucas perguntas):

- Qual é a experiência dele com programação em geral e com **esta linguagem/framework** em particular (nenhuma, básica, intermediária)?
- Ele já consegue rodar o projeto?
- Há alguma parte que ele considera prioritária ou que o confunde mais?

Use as respostas para definir o **nível-base**. Na dúvida, assuma alguém que sabe o básico de lógica (variável, if, loop, função), mas que **não conhece** a linguagem, o framework nem as bibliotecas usadas.

## Fase 1: entender o projeto e montar o roteiro (antes de escrever)

1. **Mapeie o projeto**: a árvore de pastas, os arquivos de configuração (package.json, requirements.txt, etc.), as dependências externas e o ponto de entrada (onde a execução começa).
2. **Rode o projeto** se for possível. Observe o que ele faz do ponto de vista de quem o usa.
3. **Escolha um "caso guia"**: uma ação concreta e típica do usuário (ex.: "o usuário preenche o formulário e clica em Salvar"). Esse caso vai atravessar a documentação inteira como fio condutor.
4. **Monte o grafo de dependências de conceitos**: liste os conceitos que o leitor precisa conhecer (da linguagem, do framework, do domínio) e em que ordem cada um aparece.
5. **Escreva o roteiro** (lista de capítulos na ordem de leitura) e **mostre ao usuário antes de escrever os capítulos**. Ajuste se ele pedir.

## Regras de ordem

- **Do geral para o específico**: primeiro o que o projeto faz, depois como ele é organizado, só então o código linha por linha.
- **Siga o caminho da execução**, não a lista de arquivos. Comece pelo ponto de entrada e acompanhe o caso guia: o que acontece primeiro, o que é chamado depois, para onde os dados vão.
- **Nunca use um conceito antes de explicá-lo.** Se o arquivo A depende de algo definido em B, ou você explica B antes, ou dá uma explicação curta e provisória ("por enquanto, basta saber que `db` é o objeto que conversa com o banco; vamos abri-lo no capítulo 5").
- Arquivos de apoio (utilitários, configurações, estilos, testes) vêm **depois** do fluxo principal, exceto quando o fluxo depender deles para fazer sentido.

## Regras de exemplos

- **Todo trecho de código não trivial ganha valores concretos.** Mostre um exemplo de entrada, o que acontece com ela linha a linha e a saída.
- Use uma **tabela de rastreio** em loops, transformações de dados e condicionais:

  | Passo | Linha | `total` | `item` | Observação |
  |---|---|---|---|---|
  | 1 | 12 | 0 | — | inicializa |
  | 2 | 14 | 0 | `{preco: 10}` | primeira volta do loop |
  | 3 | 15 | 10 | `{preco: 10}` | soma |

- Use os **dados do caso guia** sempre que possível, para que os exemplos se conectem entre capítulos.
- Quando ajudar, mostre o **contrafactual**: "se esta linha não existisse, aconteceria X" ou "se `usuario` viesse vazio, cairíamos no `else` da linha 23".
- Prefira exemplos que o leitor possa **executar** (um comando, um `console.log`/`print` sugerido, uma chamada no terminal), e diga o resultado esperado.
- Nunca invente comportamento: se não tiver certeza do que a linha faz, rode, teste ou marque explicitamente como incerto.

## Regras de conexão

Todo capítulo segue esta moldura:

1. **Onde estamos**: 2 a 4 linhas retomando o capítulo anterior e situando este no fluxo ("No capítulo 3 o formulário enviou os dados para `/api/salvar`. Agora vamos ver quem recebe esses dados.").
2. **O que este arquivo/trecho faz, em uma frase**, e por que ele existe no projeto.
3. **Explicação linha por linha** (formato abaixo).
4. **Resumo**: o que entrou, o que saiu, o que mudou.
5. **Próximo passo**: para onde o fluxo vai e qual capítulo trata disso.

Além disso:

- Use **referências cruzadas explícitas**: "a função `formatarData` (capítulo 6, linha 8)", nunca "aquela função de antes".
- Mantenha um **diagrama de fluxo** simples (Mermaid ou ASCII) no capítulo de visão geral, e repita a parte relevante dele, destacada, no início de cada capítulo ("você está aqui").
- Quando o mesmo padrão aparecer de novo, **aponte a repetição** ("é o mesmo padrão de `try/catch` do capítulo 4") em vez de explicar do zero ou ignorar.

## Regras de conhecimento prévio

- **Na primeira vez que um conceito aparece**, pare e explique-o num bloco próprio, antes da linha que o usa:

  > **Conceito: Promise / `async` / `await`**
  > O que é, em uma frase. Uma analogia do dia a dia. Um mini-exemplo isolado, de 3 a 5 linhas, fora do projeto. Como ele aparece aqui no projeto.

- Nas vezes seguintes, só referencie: "(ver Conceito: Promise, cap. 2)".
- **Explique a sintaxe estranha.** Operadores e construções que não são óbvios para quem não conhece a linguagem (`?.`, `??`, `...spread`, desestruturação, decorators, list comprehension, arrow functions, generics, etc.) ganham explicação na primeira ocorrência.
- **Explique o que vem de fora.** Para cada biblioteca ou framework: o que ela é, por que o projeto a usa e o que a função específica chamada faz. Deixe claro o que é código do projeto e o que é código de terceiros.
- **Evite palavras que escondem lacunas**: "simplesmente", "obviamente", "apenas", "como sabemos", "trivial".
- Termos técnicos: defina na primeira ocorrência e adicione ao **glossário**.

## Formato da explicação linha por linha

Mostre o código em **blocos pequenos** (cerca de 3 a 10 linhas que formam uma unidade de sentido), **com os números de linha reais do arquivo**, e explique cada linha logo abaixo do bloco:

````markdown
```js
12  const itens = await buscarItens(usuario.id);
13  const total = itens.reduce((soma, item) => soma + item.preco, 0);
14  return { total, quantidade: itens.length };
```

- **L12**: chama `buscarItens` (definida no cap. 4), passando o id do usuário. O `await` faz o código esperar a resposta do banco antes de seguir (ver Conceito: async/await). No caso guia, `itens` vira `[{preco: 10}, {preco: 25}]`.
- **L13**: soma os preços. `reduce` percorre a lista acumulando um valor; começa em `0` (o último argumento). Resultado no caso guia: `35`. (ver tabela de rastreio abaixo)
- **L14**: devolve um objeto com dois campos. `{ total }` é atalho para `{ total: total }`. Resultado: `{ total: 35, quantidade: 2 }`.
````

Cada explicação de linha deve responder, quando fizer sentido:

- **O quê**: o que a linha faz, com a sintaxe decifrada.
- **Por quê**: por que ela está aqui e qual problema resolve. Não basta parafrasear: "`x = 5` atribui 5 a x" não ensina nada; diga para que serve o `x`.
- **Com o exemplo**: qual valor sai dela no caso guia.

Linhas realmente triviais (imports repetidos, chaves de fechamento, boilerplate idêntico ao já explicado) podem ser agrupadas numa frase, mas nunca puladas em silêncio.

## Estrutura dos arquivos da documentação

Crie uma pasta `docs-estudo/` (ou o nome que o usuário preferir) com:

```
00-como-ler-este-guia.md   ordem de leitura, pré-requisitos, como rodar o projeto
01-visao-geral.md          o que o projeto faz, caso guia, diagrama do fluxo, mapa de pastas
02-conceitos-base.md       conceitos que aparecem em quase tudo (só os realmente transversais)
03-...md                   um capítulo por etapa do fluxo do caso guia, na ordem da execução
NN-arquivos-de-apoio.md    configs, utilitários e estilos não cobertos antes
glossario.md               termos em ordem alfabética, com link para onde aparecem
exercicios.md              exercícios para fixar o aprendizado (ver abaixo)
```

## Pontos de atenção no código

O projeto foi gerado por IA, então pode ter código morto, duplicado, inseguro ou desnecessariamente complicado. Quando encontrar algo assim, **não esconda nem "conserte em silêncio" na explicação**: aponte num bloco separado, para que o leitor não ache que aquilo é boa prática:

> **Ponto de atenção:** a função `validar` (L30–45) nunca é chamada em lugar nenhum do projeto. Provavelmente sobrou de uma versão anterior.

## Verificação do aprendizado

Ao final de cada capítulo, inclua 2 ou 3 **perguntas de checagem** (com respostas escondidas em `<details>`), por exemplo:

- "O que aconteceria se `usuario.id` viesse `undefined` na linha 12?"
- "Em que arquivo você mexeria para mudar o formato da data exibida?"

No `exercicios.md`, proponha pequenas modificações práticas, em dificuldade crescente (mudar um texto, adicionar um campo, criar uma rota nova).

## Processo de escrita

1. Faça a Fase 0 e a Fase 1, e **mostre o roteiro ao usuário antes de escrever**.
2. Escreva **um capítulo por vez**. Ao terminar cada um, pergunte se o nível de detalhe e o ritmo estão bons e ajuste os próximos capítulos de acordo.
3. Antes de entregar cada capítulo, confira o checklist abaixo.

## Checklist por capítulo

- [ ] Começa com "Onde estamos" e termina com "Próximo passo"?
- [ ] Todo conceito usado já foi explicado antes ou é explicado aqui?
- [ ] Toda sintaxe não óbvia foi decifrada na primeira ocorrência?
- [ ] Todo trecho não trivial tem exemplo com valores concretos (de preferência do caso guia)?
- [ ] As explicações dizem o **porquê**, não só parafraseiam o código?
- [ ] Os números de linha batem com o arquivo real?
- [ ] As referências cruzadas apontam capítulo e linha específicos?
- [ ] Novos termos foram adicionados ao glossário?
- [ ] Afirmações sobre o comportamento foram verificadas (lendo ou rodando o código), sem suposições?
- [ ] Há perguntas de checagem no final?

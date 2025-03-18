# Projeto de Algoritmos de Ordenação

## 🔍 Visão Geral

Este projeto implementa diversos algoritmos de ordenação e compara seu desempenho em termos de tempo de execução, número de comparações e número de trocas. A implementação utiliza o padrão de projeto Strategy para permitir uma arquitetura modular e facilmente extensível, além de integrar OpenTelemetry para monitoramento detalhado das operações internas dos algoritmos.

## 🧮 Algoritmos Implementados

- Bubble Sort
- Bubble Sort Melhorado
- Insertion Sort
- Selection Sort
- Quick Sort
- Merge Sort
- Tim Sort
- Heap Sort
- Counting Sort (para números inteiros)
- Radix Sort (para inteiros)
- Shell Sort

## 🛠️ Instalação

1. Clone o repositório:
```
git clone https://github.com/Fhuller/sortComparisons.git
```

2. Instale as dependências:
```
pip install -r requirements.txt
```

### Dependências principais:

- **matplotlib**: Para geração de gráficos de desempenho
- **opentelemetry-api**: API OpenTelemetry
- **opentelemetry-sdk**: SDK OpenTelemetry
- **opentelemetry-exporter-otlp**: Exportador OTLP para OpenTelemetry

## 📝 Como Usar

O projeto oferece uma interface de linha de comando (CLI) com dois comandos principais:

### Gerando Dados

Para gerar um conjunto de números aleatórios e salvá-los em um arquivo:

```bash
python main.py generate --size <quantidade> [--min <valor-mínimo>] [--max <valor-máximo>] [--output <arquivo-saída>]
```

| Parâmetro | Descrição | Padrão |
|-----------|-----------|--------|
| `--size`  | Quantidade de números a serem gerados (obrigatório) | - |
| `--min`   | Valor mínimo para os números gerados | 0 |
| `--max`   | Valor máximo para os números gerados | 100000 |
| `--output`| Nome do arquivo de saída | data.txt |

### Executando Algoritmos

Para executar os algoritmos de ordenação em um conjunto de dados:

```bash
python main.py run [--input <arquivo-entrada>] [--algorithms <algoritmos>] [--repetitions <repetições>] [--output <arquivo-resultados>] [--graph] [--graph-output <arquivo-grafico>]
```

| Parâmetro | Descrição | Padrão |
|-----------|-----------|--------|
| `--input` | Arquivo contendo os números a serem ordenados | data.txt |
| `--algorithms` | Lista de algoritmos a serem executados (separados por espaço) ou "all" para todos | all |
| `--repetitions` | Número de repetições para cada algoritmo | 5 |
| `--output` | Nome do arquivo para salvar os resultados | results.txt |
| `--graph` | Flag para gerar um gráfico de desempenho | False |
| `--graph-output` | Nome do arquivo para salvar o gráfico | performance_graph.png |

### Exemplos de Uso

Gerar 10.000 números aleatórios:
```bash
python main.py generate --size 10000 --min 1 --max 50000 --output meus_dados.txt
```

Executar todos os algoritmos:
```bash
python main.py run --input meus_dados.txt --algorithms all --repetitions 5 --output resultados.txt
```

Executar apenas alguns algoritmos específicos:
```bash
python main.py run --input meus_dados.txt --algorithms quick merge heap --repetitions 3 --output resultados_comparativos.txt
```

Executar algoritmos e gerar um gráfico de desempenho:
```bash
python main.py run --input meus_dados.txt --algorithms all --graph --graph-output grafico_desempenho.png
```

## 📊 Visualização de Desempenho

O projeto agora inclui uma ferramenta de visualização que gera gráficos de dispersão mostrando a relação entre:
- **Eixo X**: Número médio de trocas (swaps)
- **Eixo Y**: Tempo médio de execução (ms)
- **Tamanho dos pontos**: Representa o número de comparações

Cada algoritmo é representado como um ponto no gráfico, com uma linha de tendência que ajuda a visualizar a eficiência relativa de cada um.

## 📡 Monitoramento com OpenTelemetry

O projeto integra OpenTelemetry para monitoramento detalhado e observabilidade:

- **Traces**: Cada algoritmo gera spans detalhados que permitem analisar seu comportamento interno
- **Métricas**: São coletadas métricas como tempo de execução, número de comparações e trocas
- **Exportação**: Os dados podem ser exportados para um coletor OpenTelemetry (configurado por padrão para `http://localhost:4317`)

Para visualizar os dados de telemetria, você pode usar:
- Jaeger UI (para traces)
- Prometheus + Grafana (para métricas)
- Qualquer outra ferramenta compatível com OpenTelemetry

## 📂 Estrutura do Projeto

```
sorting-algorithms/
├── main.py                 # Script principal com CLI
├── README.md               # Este arquivo
├── requirements.txt        # Dependências do projeto
├── data.txt                # Exemplo de arquivo de dados (gerado)
├── results.txt             # Exemplo de arquivo de resultados (gerado)
└── performance_graph.png   # Exemplo de gráfico de desempenho (gerado)
```

## 🏗️ Padrão Strategy

O projeto utiliza o padrão de projeto Strategy para implementar os algoritmos de ordenação de forma modular:

1. `SortingStrategy` (classe abstrata): Define a interface comum para todos os algoritmos de ordenação
   - Implementa a instrumentação OpenTelemetry
   - Centraliza a lógica de métricas e spans
   
2. Classes concretas (BubbleSort, QuickSort, etc.): 
   - Implementam apenas o método `_sort_implementation()` com a lógica específica do algoritmo
   - Herdam toda a instrumentação automaticamente

3. `SortingContext`: 
   - Utiliza as estratégias (algoritmos) para executar a ordenação
   - Gerencia a coleta de métricas para comparação

Este padrão permite adicionar novos algoritmos de ordenação sem modificar o código existente, seguindo o princípio Open/Closed do SOLID.

## 📊 Métricas Coletadas

Para cada algoritmo, as seguintes métricas são coletadas:

- **Tempo de execução** (em milissegundos)
- **Número de comparações** entre elementos
- **Número de trocas** (movimentações de elementos)
- **Spans detalhados** para análise de comportamento interno

Para garantir resultados confiáveis, cada algoritmo é executado várias vezes, e as métricas médias são calculadas.

## 🔜 Roadmap

Próximos passos para o projeto:

- [x] Implementação dos algoritmos básicos
- [x] Coleta de métricas de desempenho
- [x] Interface de linha de comando
- [x] Implementação de logs com OpenTelemetry
- [x] Integração com ferramentas de visualização (gráficos matplotlib e logs Jaeger)

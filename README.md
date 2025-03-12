# Projeto de Algoritmos de Ordenação

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Um projeto para implementar, analisar e comparar a performance de diferentes algoritmos de ordenação utilizando o padrão de projeto Strategy. O projeto inclui geração de dados aleatórios, execução dos algoritmos e coleta de métricas de desempenho.

## 📋 Tabela de Conteúdos

- [Visão Geral](#visão-geral)
- [Algoritmos Implementados](#algoritmos-implementados)
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Como Usar](#como-usar)
  - [Gerando Dados](#gerando-dados)
  - [Executando Algoritmos](#executando-algoritmos)
  - [Exemplos de Uso](#exemplos-de-uso)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Padrão Strategy](#padrão-strategy)
- [Métricas Coletadas](#métricas-coletadas)
- [Roadmap](#roadmap)
- [Contribuição](#contribuição)
- [Licença](#licença)

## 🔍 Visão Geral

Este projeto implementa diversos algoritmos de ordenação e compara seu desempenho em termos de tempo de execução, número de comparações e número de trocas. A implementação utiliza o padrão de projeto Strategy para permitir uma arquitetura modular e facilmente extensível.

## 🧮 Algoritmos Implementados

### Básicos
- Bubble Sort
- Bubble Sort Melhorado
- Insertion Sort
- Selection Sort

### Avançados (Dividir para Conquistar)
- Quick Sort
- Merge Sort
- Tim Sort

### Outros Algoritmos
- Heap Sort
- Counting Sort (para números inteiros)
- Radix Sort (para inteiros)
- Shell Sort

## 📋 Pré-requisitos

- Python 3.8 ou superior
- Bibliotecas padrão: os, time, random, argparse, statistics, abc, typing

## 🛠️ Instalação

1. Clone o repositório:
```
git clone https://github.com/Fhuller/sortComparisons.git
```

2. Não é necessário instalar dependências externas para a funcionalidade básica.

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
python main.py run [--input <arquivo-entrada>] [--algorithms <algoritmos>] [--repetitions <repetições>] [--output <arquivo-resultados>]
```

| Parâmetro | Descrição | Padrão |
|-----------|-----------|--------|
| `--input` | Arquivo contendo os números a serem ordenados | data.txt |
| `--algorithms` | Lista de algoritmos a serem executados (separados por espaço) ou "all" para todos | all |
| `--repetitions` | Número de repetições para cada algoritmo | 5 |
| `--output` | Nome do arquivo para salvar os resultados | results.txt |

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

## 📂 Estrutura do Projeto

```
sorting-algorithms/
├── main.py              # Script principal com CLI
├── README.md            # Este arquivo
├── data.txt             # Exemplo de arquivo de dados (gerado)
└── results.txt          # Exemplo de arquivo de resultados (gerado)
```

## 🏗️ Padrão Strategy

O projeto utiliza o padrão de projeto Strategy para implementar os algoritmos de ordenação de forma modular:

1. `SortingStrategy` (classe abstrata): Define a interface comum para todos os algoritmos de ordenação
2. Classes concretas (BubbleSort, QuickSort, etc.): Implementam a interface SortingStrategy
3. `SortingContext`: Utiliza as estratégias (algoritmos) para executar a ordenação

Este padrão permite adicionar novos algoritmos de ordenação sem modificar o código existente, seguindo o princípio Open/Closed do SOLID.

## 📊 Métricas Coletadas

Para cada algoritmo, as seguintes métricas são coletadas:

- **Tempo de execução** (em milissegundos)
- **Número de comparações** entre elementos
- **Número de trocas** (movimentações de elementos)

Para garantir resultados confiáveis, cada algoritmo é executado várias vezes, e as métricas médias são calculadas.

## 🔜 Roadmap

Próximos passos para o projeto:

- [x] Implementação dos algoritmos básicos
- [x] Coleta de métricas de desempenho
- [x] Interface de linha de comando
- [ ] Implementação de logs com OpenTelemetry
- [ ] Integração com ferramentas de visualização (Jaeger, Prometheus + Grafana, etc.)
- [ ] Visualização de gráficos comparativos
- [ ] Interface gráfica para interação com o sistema

## 👥 Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.

1. Fork o projeto
2. Crie sua branch de feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

## 📜 Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para mais detalhes.

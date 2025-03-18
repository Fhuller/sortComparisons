"""
Projeto de Algoritmos de Ordenação
===================================
Este projeto implementa diversos algoritmos de ordenação e compara seu desempenho,
utilizando o padrão de projeto Strategy para uma implementação modular.
"""

import os
import time
import random
import argparse
import statistics
import logging
import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry import metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configure OpenTelemetry
def setup_opentelemetry(service_name="sorting-algorithms"):
    """
    Configura o OpenTelemetry para coletar traces e métricas e enviá-los para o OpenTelemetry Collector.
    """

    # Definir os recursos do serviço
    resource = Resource.create({SERVICE_NAME: service_name})

    # Criar o provedor de traces
    tracer_provider = TracerProvider(resource=resource)

    # Configurar o exportador OTLP para o Collector rodando no Docker (gRPC)
    otlp_endpoint = "http://localhost:4317"

    # Exportador de spans (traces)
    otlp_trace_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    tracer_provider.add_span_processor(BatchSpanProcessor(otlp_trace_exporter))

    # Definir o provedor de traces globalmente
    trace.set_tracer_provider(tracer_provider)

    # Criar o provedor de métricas
    meter_provider = MeterProvider(
        resource=resource,
        metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=otlp_endpoint, insecure=True))]
    )

    # Definir o provedor de métricas globalmente
    metrics.set_meter_provider(meter_provider)

    logger.info(f"✅ OpenTelemetry configurado e enviando dados para {otlp_endpoint}")

    return trace.get_tracer(service_name), metrics.get_meter(service_name)


# Get tracer and meter
tracer, meter = setup_opentelemetry()

# Create metrics
sort_execution_time = meter.create_histogram(
    name="sort_execution_time",
    description="Time taken to execute sorting algorithm",
    unit="ms"
)

sort_comparisons = meter.create_counter(
    name="sort_comparisons",
    description="Number of comparisons made during sorting"
)

sort_swaps = meter.create_counter(
    name="sort_swaps",
    description="Number of swaps made during sorting"
)


# Implementação do padrão Strategy para algoritmos de ordenação
class SortingStrategy(ABC):
    """Classe abstrata que define a interface para todos os algoritmos de ordenação."""
    
    def __init__(self):
        self.comparisons = 0
        self.swaps = 0
        self.algorithm_name = self.__class__.__name__
    
    @abstractmethod
    def _sort_implementation(self, data: List[int]) -> List[int]:
        """
        Método abstrato que deve ser implementado por todas as estratégias de ordenação.
        Esta é a implementação específica do algoritmo.
        """
        pass
    
    def sort(self, data: List[int]) -> List[int]:
        """
        Método público que executa o algoritmo de ordenação com toda a instrumentação necessária.
        Este método faz a inicialização, chama a implementação específica e registra as métricas.
        """
        self.reset_metrics()
        arr = data.copy()
        n = len(arr)
        
        # Execute o algoritmo com o span
        with tracer.start_as_current_span(f"{self.algorithm_name.lower()}_algorithm") as span:
            start_time = time.time()  # ⏳ Inicia a contagem do tempo
            
            # Chama a implementação específica
            self._sort_implementation(arr)
            
            end_time = time.time()  # ⏳ Finaliza a contagem do tempo
            
            # 📌 Adicionando atributos importantes ao span
            span.set_attribute("algorithm", self.algorithm_name)
            span.set_attribute("dataset_size", n)
            span.set_attribute("execution_time_ms", (end_time - start_time) * 1000)
            span.set_attribute("comparisons", self.comparisons)
            span.set_attribute("swaps", self.swaps)
            
            # Registra métricas no OpenTelemetry
            execution_time_ms = (end_time - start_time) * 1000
            sort_execution_time.record(execution_time_ms, {"algorithm": self.algorithm_name})
        
        return arr
    
    def get_metrics(self) -> Dict[str, int]:
        """Retorna as métricas de comparações e trocas."""
        return {
            "comparisons": self.comparisons,
            "swaps": self.swaps
        }
    
    def reset_metrics(self) -> None:
        """Reinicia as métricas para nova execução."""
        self.comparisons = 0
        self.swaps = 0
    
    def compare(self, a: int, b: int) -> bool:
        """Compara dois elementos e incrementa o contador de comparações."""
        self.comparisons += 1
        # Record comparison in OpenTelemetry
        sort_comparisons.add(1, {"algorithm": self.algorithm_name})
        return a > b
    
    def swap(self, data: List[int], i: int, j: int) -> None:
        """Troca dois elementos e incrementa o contador de trocas."""
        if i != j:
            self.swaps += 1
            # Record swap in OpenTelemetry
            sort_swaps.add(1, {"algorithm": self.algorithm_name})
            data[i], data[j] = data[j], data[i]

    def run_with_telemetry(self, data: List[int]) -> Tuple[List[int], Dict[str, Any]]:
        """Execute o algoritmo de ordenação com telemetria OpenTelemetry."""
        # Start a span for this sorting operation
        with tracer.start_as_current_span(f"{self.algorithm_name}_sort", 
                                         attributes={"array_size": len(data)}):
            # Record the start time
            start_time = time.time()
            
            # Execute the sort algorithm
            result = self.sort(data)
            
            # Calculate execution time
            execution_time = (time.time() - start_time) * 1000  # in milliseconds
            
            # Record metrics
            sort_execution_time.record(execution_time, {"algorithm": self.algorithm_name})
            
            # Log metrics
            logger.info(f"{self.algorithm_name} sorted {len(data)} elements in {execution_time:.2f}ms")
            logger.info(f"Comparisons: {self.comparisons}, Swaps: {self.swaps}")
            
            # Return the sorted array and metrics
            return result, {
                "algorithm": self.algorithm_name,
                "execution_time_ms": execution_time,
                "comparisons": self.comparisons,
                "swaps": self.swaps,
                "array_size": len(data)
            }


# Implementações dos algoritmos básicos
class BubbleSort(SortingStrategy):
    """Implementação do algoritmo Bubble Sort."""
    
    def _sort_implementation(self, arr: List[int]) -> List[int]:
        n = len(arr)
        
        for i in range(n):
            for j in range(0, n - i - 1):
                if self.compare(arr[j], arr[j + 1]):
                    self.swap(arr, j, j + 1)
        
        return arr


class ImprovedBubbleSort(SortingStrategy):
    """Implementação do algoritmo Bubble Sort Melhorado."""
    
    def _sort_implementation(self, arr: List[int]) -> List[int]:
        n = len(arr)
        
        for i in range(n):
            swapped = False
            for j in range(0, n - i - 1):
                if self.compare(arr[j], arr[j + 1]):
                    self.swap(arr, j, j + 1)
                    swapped = True
            
            # Se nenhuma troca foi feita na passagem, o array já está ordenado
            if not swapped:
                break
        
        return arr


class InsertionSort(SortingStrategy):
    """Implementação do algoritmo Insertion Sort."""
    
    def _sort_implementation(self, arr: List[int]) -> List[int]:
        n = len(arr)
        
        for i in range(1, n):
            key = arr[i]
            j = i - 1
            
            # Move os elementos maiores que key uma posição à frente
            while j >= 0 and self.compare(arr[j], key):
                arr[j + 1] = arr[j]
                self.swaps += 1
                sort_swaps.add(1, {"algorithm": self.algorithm_name})
                j -= 1
            
            arr[j + 1] = key
        
        return arr


class SelectionSort(SortingStrategy):
    """Implementação do algoritmo Selection Sort."""
    
    def _sort_implementation(self, arr: List[int]) -> List[int]:
        n = len(arr)
        
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if self.compare(arr[min_idx], arr[j]):
                    min_idx = j
            
            self.swap(arr, min_idx, i)
        
        return arr


# Implementações dos algoritmos avançados
class QuickSort(SortingStrategy):
    """Implementação do algoritmo Quick Sort."""
    
    def _sort_implementation(self, arr: List[int]) -> List[int]:
        self._quick_sort(arr, 0, len(arr) - 1)
        return arr
    
    def _quick_sort(self, arr: List[int], low: int, high: int) -> None:
        if low < high:
            # Particiona o array e obtém o índice do pivô
            pi = self._partition(arr, low, high)
            
            # Ordena os elementos antes e depois do pivô
            self._quick_sort(arr, low, pi - 1)
            self._quick_sort(arr, pi + 1, high)
    
    def _partition(self, arr: List[int], low: int, high: int) -> int:
        # Seleciona o pivô (o último elemento)
        pivot = arr[high]
        i = low - 1
        
        for j in range(low, high):
            # Se o elemento atual for menor que o pivô
            if not self.compare(arr[j], pivot):
                i += 1
                self.swap(arr, i, j)
        
        # Coloca o pivô na posição correta
        self.swap(arr, i + 1, high)
        return i + 1


class MergeSort(SortingStrategy):
    """Implementação do algoritmo Merge Sort."""
    
    def _sort_implementation(self, arr: List[int]) -> List[int]:
        self._merge_sort(arr, 0, len(arr) - 1)
        return arr
    
    def _merge_sort(self, arr: List[int], left: int, right: int) -> None:
        if left < right:
            mid = (left + right) // 2
            
            # Ordena as duas metades
            self._merge_sort(arr, left, mid)
            self._merge_sort(arr, mid + 1, right)
            
            # Mescla as metades ordenadas
            self._merge(arr, left, mid, right)
    
    def _merge(self, arr: List[int], left: int, mid: int, right: int) -> None:
        # Cria arrays temporários
        L = arr[left:mid + 1]
        R = arr[mid + 1:right + 1]
        
        # Índices iniciais dos subarrays
        i = j = 0
        k = left
        
        # Mescla os subarrays
        while i < len(L) and j < len(R):
            if not self.compare(L[i], R[j]):
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
            self.swaps += 1
            sort_swaps.add(1, {"algorithm": self.algorithm_name})
        
        # Copia os elementos restantes de L[], se houver
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
            self.swaps += 1
            sort_swaps.add(1, {"algorithm": self.algorithm_name})
        
        # Copia os elementos restantes de R[], se houver
        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1
            self.swaps += 1
            sort_swaps.add(1, {"algorithm": self.algorithm_name})


class TimSort(SortingStrategy):
    """Implementação simplificada do algoritmo Tim Sort."""
    
    def __init__(self):
        super().__init__()
        self.MIN_MERGE = 32  # Tamanho mínimo para fusão
    
    def _sort_implementation(self, arr: List[int]) -> List[int]:
        n = len(arr)
        
        # Ordena pequenos subarrays usando insertion sort
        for i in range(0, n, self.MIN_MERGE):
            self._insertion_sort(arr, i, min((i + self.MIN_MERGE - 1), (n - 1)))
        
        # Começa a mesclar os subarrays
        size = self.MIN_MERGE
        while size < n:
            for left in range(0, n, 2 * size):
                mid = min(n - 1, left + size - 1)
                right = min(left + 2 * size - 1, n - 1)
                
                # Mescla subarrays arr[left...mid] e arr[mid+1...right]
                if mid < right:
                    self._merge(arr, left, mid, right)
            
            size = 2 * size
        
        return arr
    
    def _insertion_sort(self, arr: List[int], left: int, right: int) -> None:
        for i in range(left + 1, right + 1):
            temp = arr[i]
            j = i - 1
            while j >= left and self.compare(arr[j], temp):
                arr[j + 1] = arr[j]
                self.swaps += 1
                sort_swaps.add(1, {"algorithm": self.algorithm_name})
                j -= 1
            arr[j + 1] = temp
    
    def _merge(self, arr: List[int], left: int, mid: int, right: int) -> None:
        # Mesma implementação do merge do MergeSort
        L = arr[left:mid + 1]
        R = arr[mid + 1:right + 1]
        
        i = j = 0
        k = left
        
        while i < len(L) and j < len(R):
            if not self.compare(L[i], R[j]):
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
            self.swaps += 1
            sort_swaps.add(1, {"algorithm": self.algorithm_name})
        
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
            self.swaps += 1
            sort_swaps.add(1, {"algorithm": self.algorithm_name})
        
        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1
            self.swaps += 1
            sort_swaps.add(1, {"algorithm": self.algorithm_name})


class HeapSort(SortingStrategy):
    """Implementação do algoritmo Heap Sort."""
    
    def _sort_implementation(self, arr: List[int]) -> List[int]:
        n = len(arr)
        
        # Constrói um heap máximo
        for i in range(n // 2 - 1, -1, -1):
            self._heapify(arr, n, i)
        
        # Extrai elementos do heap um por um
        for i in range(n - 1, 0, -1):
            self.swap(arr, i, 0)
            self._heapify(arr, i, 0)
        
        return arr
    
    def _heapify(self, arr: List[int], n: int, i: int) -> None:
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        
        # Verifica se o filho esquerdo existe e é maior que a raiz
        if left < n and self.compare(arr[left], arr[largest]):
            largest = left
        
        # Verifica se o filho direito existe e é maior que a raiz
        if right < n and self.compare(arr[right], arr[largest]):
            largest = right
        
        # Troca e continua heapificando se necessário
        if largest != i:
            self.swap(arr, i, largest)
            self._heapify(arr, n, largest)


class CountingSort(SortingStrategy):
    """Implementação do algoritmo Counting Sort (para inteiros com range limitado)."""
    
    def _sort_implementation(self, arr: List[int]) -> List[int]:
        # Encontra o maior elemento no array
        max_val = max(arr)
        min_val = min(arr)
        range_val = max_val - min_val + 1
        
        # Log range info for monitoring
        logger.info(f"Counting Sort range: min={min_val}, max={max_val}, range={range_val}")
        
        # Cria um array de contagem e inicializa com zeros
        count = [0] * range_val
        output = [0] * len(arr)
        
        # Armazena a contagem de cada elemento
        for i in range(len(arr)):
            count[arr[i] - min_val] += 1
            self.comparisons += 1  # Comparação implícita ao indexar
            sort_comparisons.add(1, {"algorithm": self.algorithm_name})
        
        # Modifica o array de contagem para conter posições reais
        for i in range(1, len(count)):
            count[i] += count[i - 1]
            self.comparisons += 1  # Comparação implícita
            sort_comparisons.add(1, {"algorithm": self.algorithm_name})
        
        # Constrói o array de saída
        for i in range(len(arr) - 1, -1, -1):
            output[count[arr[i] - min_val] - 1] = arr[i]
            count[arr[i] - min_val] -= 1
            self.swaps += 1
            sort_swaps.add(1, {"algorithm": self.algorithm_name})
        
        # Copia o array de saída para o array original
        for i in range(len(arr)):
            arr[i] = output[i]
            self.swaps += 1
            sort_swaps.add(1, {"algorithm": self.algorithm_name})
        
        return arr


class RadixSort(SortingStrategy):
    """Implementação do algoritmo Radix Sort (para inteiros)."""
    
    def _sort_implementation(self, arr: List[int]) -> List[int]:
        # Encontra o número máximo para saber o número de dígitos
        max_val = max(arr)
        
        # Log max value for monitoring
        logger.info(f"Radix Sort max value: {max_val}")
        
        # Faz o counting sort para cada posição de dígito
        exp = 1
        digits_processed = 0
        while max_val // exp > 0:
            self._counting_sort(arr, exp)
            exp *= 10
            digits_processed += 1
        
        return arr
    
    def _counting_sort(self, arr: List[int], exp: int) -> None:
        n = len(arr)
        output = [0] * n
        count = [0] * 10
        
        # Armazena a contagem de ocorrências no array count
        for i in range(n):
            index = (arr[i] // exp) % 10
            count[index] += 1
            self.comparisons += 1  # Comparação implícita
            sort_comparisons.add(1, {"algorithm": self.algorithm_name})
        
        # Modifica count para que contenha a posição real
        for i in range(1, 10):
            count[i] += count[i - 1]
            self.comparisons += 1  # Comparação implícita
            sort_comparisons.add(1, {"algorithm": self.algorithm_name})
        
        # Constrói o array de saída
        for i in range(n - 1, -1, -1):
            index = (arr[i] // exp) % 10
            output[count[index] - 1] = arr[i]
            count[index] -= 1
            self.swaps += 1
            sort_swaps.add(1, {"algorithm": self.algorithm_name})
        
        # Copia o array de saída para arr[]
        for i in range(n):
            arr[i] = output[i]
            self.swaps += 1
            sort_swaps.add(1, {"algorithm": self.algorithm_name})


class ShellSort(SortingStrategy):
    """Implementação do algoritmo Shell Sort."""
    
    def _sort_implementation(self, arr: List[int]) -> List[int]:
        n = len(arr)
        
        # Começa com um gap grande e vai reduzindo
        gap = n // 2
        
        while gap > 0:
            for i in range(gap, n):
                temp = arr[i]
                j = i
                
                # Faz um insertion sort para elementos com distância gap
                while j >= gap and self.compare(arr[j - gap], temp):
                    arr[j] = arr[j - gap]
                    self.swaps += 1
                    sort_swaps.add(1, {"algorithm": self.algorithm_name})
                    j -= gap
                
                arr[j] = temp
            
            gap //= 2
        
        return arr


# Contexto que utiliza as estratégias
class SortingContext:
    """Contexto que aplica a estratégia de ordenação escolhida, agora com OpenTelemetry."""

    def __init__(self, strategy: SortingStrategy = None):
        self._strategy = strategy

    def set_strategy(self, strategy: SortingStrategy) -> None:
        """Define a estratégia de ordenação."""
        self._strategy = strategy

    def sort(self, data: List[int]) -> Tuple[List[int], float, Dict[str, int]]:
        """
        Executa a ordenação usando a estratégia atual e mede o tempo de execução,
        incluindo OpenTelemetry para rastreamento.

        Returns:
            tuple: (dados ordenados, tempo de execução, métricas)
        """
        if not self._strategy:
            raise ValueError("Estratégia de ordenação não definida")

        # Criar um span para rastrear a execução do algoritmo
        with tracer.start_as_current_span("sorting_execution", attributes={
            "algorithm": self._strategy.algorithm_name,
            "array_size": len(data)
        }):
            start_time = time.time()
            sorted_data = self._strategy.sort(data)
            execution_time = (time.time() - start_time) * 1000  # Tempo em milissegundos

            # Registrar o tempo de execução na métrica do OpenTelemetry
            sort_execution_time.record(execution_time, {"algorithm": self._strategy.algorithm_name})

            # Obter métricas do algoritmo
            metrics = self._strategy.get_metrics()
            metrics["execution_time_ms"] = execution_time

            # Log para depuração
            logger.info(f"Algoritmo {self._strategy.algorithm_name} executado em {execution_time:.2f} ms")

            return sorted_data, execution_time, metrics


# Funções para geração e manipulação de dados
def generate_random_numbers(size: int, min_val: int = 0, max_val: int = 100000) -> List[int]:
    """Gera uma lista de números aleatórios."""
    return [random.randint(min_val, max_val) for _ in range(size)]


def save_numbers_to_file(numbers: List[int], filename: str) -> None:
    """Salva uma lista de números em um arquivo de texto."""
    with open(filename, 'w', encoding='utf-8') as file:
        for number in numbers:
            file.write(f"{number}\n")


def load_numbers_from_file(filename: str) -> List[int]:
    """Carrega uma lista de números de um arquivo de texto."""
    numbers = []
    with open(filename, 'r') as file:
        for line in file:
            numbers.append(int(line.strip()))
    return numbers


def save_results_to_file(results: Dict[str, Any], filename: str) -> None:
    """Salva os resultados da execução em um arquivo de texto."""
    with open(filename, 'w', encoding='utf-8') as file:
        file.write("Resultados de Desempenho dos Algoritmos de Ordenação\n")
        file.write("=================================================\n\n")

        file.write(f"Tamanho do Conjunto de Dados: {results['data_size']}\n")
        file.write(f"Repetições por Algoritmo: {results['repetitions']}\n\n")

        file.write("Métricas de Desempenho:\n")
        file.write("---------------------\n")

        for algo_name, metrics in results['algorithms'].items():
            file.write(f"\n{algo_name}:\n")
            file.write(f"  Tempo médio de execução: {metrics['avg_time']:.2f} ms\n")
            file.write(f"  Tempo mínimo: {metrics['min_time']:.2f} ms\n")
            file.write(f"  Tempo máximo: {metrics['max_time']:.2f} ms\n")
            file.write(f"  Comparações médias: {metrics['avg_comparisons']:.0f}\n")
            file.write(f"  Trocas médias: {metrics['avg_swaps']:.0f}\n")


def generate_performance_graph(results: Dict[str, Any], output_filename: str = "performance_graph.png") -> None:
    """
    Gera um gráfico de dispersão onde:
    - Eixo X representa o número médio de trocas (swaps)
    - Eixo Y representa o tempo médio de execução
    - Cada ponto representa um algoritmo de ordenação
    """
    # Extrair dados para o gráfico
    algorithms = []
    swaps = []
    times = []
    comparisons = []  # Para o tamanho dos pontos

    for algo_name, metrics in results['algorithms'].items():
        algorithms.append(algo_name)
        swaps.append(metrics['avg_swaps'])
        times.append(metrics['avg_time'])
        comparisons.append(metrics['avg_comparisons'])

    # Normalizar o tamanho dos pontos para melhor visualização
    # (usamos comparações para definir o tamanho)
    min_comp = min(comparisons) if comparisons else 1
    max_comp = max(comparisons) if comparisons else 1
    if min_comp == max_comp:
        point_sizes = [100] * len(comparisons)
    else:
        point_sizes = [50 + 150 * (c - min_comp) / (max_comp - min_comp) for c in comparisons]

    # Cores diferentes para cada algoritmo
    colors = plt.cm.viridis(np.linspace(0, 1, len(algorithms)))

    # Criar o gráfico
    plt.figure(figsize=(12, 8))
    
    # Plotar pontos
    scatter = plt.scatter(swaps, times, s=point_sizes, c=colors, alpha=0.7)
    
    # Adicionar rótulos aos pontos
    for i, algo in enumerate(algorithms):
        plt.annotate(algo, (swaps[i], times[i]), 
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center')
    
    # Configurar o gráfico
    plt.title(f'Desempenho dos Algoritmos de Ordenação (Tamanho do Array: {results["data_size"]})', 
              fontsize=16)
    plt.xlabel('Número Médio de Trocas (swaps)', fontsize=14)
    plt.ylabel('Tempo Médio de Execução (ms)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adicionar linha de tendência
    if len(swaps) > 1:  # Precisa de pelo menos 2 pontos para uma linha de tendência
        z = np.polyfit(swaps, times, 1)
        p = np.poly1d(z)
        plt.plot(sorted(swaps), p(sorted(swaps)), "r--", alpha=0.5, 
                 label=f"Tendência: y={z[0]:.2e}x+{z[1]:.2f}")
        plt.legend()
    
    # Adicionar anotações explicativas
    plt.figtext(0.02, 0.02, 
                "Tamanho dos pontos representa o número de comparações", 
                fontsize=10)
    
    # Salvar o gráfico
    plt.tight_layout()
    plt.savefig(output_filename)
    
    # Fechando a figura para liberar memória
    plt.close()
    
    logger.info(f"Gráfico de desempenho salvo em {output_filename}")


# Função principal
def main():
    parser = argparse.ArgumentParser(description='Projeto de Algoritmos de Ordenação')
    subparsers = parser.add_subparsers(dest='command', help='Comando a ser executado')

    # Subcomando para gerar dados
    generate_parser = subparsers.add_parser('generate', help='Gera números aleatórios')
    generate_parser.add_argument('--size', type=int, required=True, help='Quantidade de números a serem gerados')
    generate_parser.add_argument('--min', type=int, default=0, help='Valor mínimo (padrão: 0)')
    generate_parser.add_argument('--max', type=int, default=100000, help='Valor máximo (padrão: 100000)')
    generate_parser.add_argument('--output', type=str, default='data.txt', help='Arquivo de saída (padrão: data.txt)')

    # Subcomando para executar algoritmos
    run_parser = subparsers.add_parser('run', help='Executa algoritmos de ordenação')
    run_parser.add_argument('--input', type=str, default='data.txt', help='Arquivo de entrada (padrão: data.txt)')
    run_parser.add_argument('--algorithms', type=str, nargs='+', default=['all'],
                            help='Algoritmos a serem executados (padrão: all)')
    run_parser.add_argument('--repetitions', type=int, default=5,
                            help='Número de repetições para cada algoritmo (padrão: 5)')
    run_parser.add_argument('--output', type=str, default='results.txt',
                            help='Arquivo de saída para os resultados (padrão: results.txt)')
    run_parser.add_argument('--graph', action='store_true',
                            help='Gerar gráfico de desempenho')
    run_parser.add_argument('--graph-output', type=str, default='performance_graph.png',
                            help='Arquivo de saída para o gráfico (padrão: performance_graph.png)')

    args = parser.parse_args()

    # Dicionário de estratégias disponíveis
    available_strategies = {
        'bubble': BubbleSort(),
        'improved_bubble': ImprovedBubbleSort(),
        'insertion': InsertionSort(),
        'selection': SelectionSort(),
        'quick': QuickSort(),
        'merge': MergeSort(),
        'tim': TimSort(),
        'heap': HeapSort(),
        'counting': CountingSort(),
        'radix': RadixSort(),
        'shell': ShellSort()
    }

    if args.command == 'generate':
        print(f"Gerando {args.size} números aleatórios entre {args.min} e {args.max}...")
        numbers = generate_random_numbers(args.size, args.min, args.max)
        save_numbers_to_file(numbers, args.output)
        print(f"Dados salvos em {args.output}")

    elif args.command == 'run':
        if not os.path.exists(args.input):
            print(f"Erro: Arquivo de entrada '{args.input}' não encontrado")
            return

        print(f"Carregando dados de {args.input}...")
        data = load_numbers_from_file(args.input)
        data_size = len(data)
        print(f"Dados carregados: {data_size} números")

        # Determina quais algoritmos executar
        algorithms_to_run = []
        if 'all' in args.algorithms:
            algorithms_to_run = list(available_strategies.keys())
        else:
            algorithms_to_run = args.algorithms

        # Resultados
        results = {
            'data_size': data_size,
            'repetitions': args.repetitions,
            'algorithms': {}
        }

        # Inicializa o contexto
        context = SortingContext()

        # Executa cada algoritmo
        for algo_name in algorithms_to_run:
            if algo_name not in available_strategies:
                print(f"Aviso: Algoritmo '{algo_name}' não encontrado, pulando...")
                continue

            print(f"Executando {algo_name}...")
            context.set_strategy(available_strategies[algo_name])

            # Métricas para múltiplas execuções
            execution_times = []
            comparison_counts = []
            swap_counts = []

            # Executa várias vezes para obter uma média confiável
            for i in range(args.repetitions):
                sorted_data, execution_time, metrics = context.sort(data)

                print(f"  Repetição {i+1}: {execution_time:.2f} ms, "
                      f"{metrics['comparisons']} comparações, {metrics['swaps']} trocas")

                execution_times.append(execution_time)
                comparison_counts.append(metrics['comparisons'])
                swap_counts.append(metrics['swaps'])

            # Calcula médias e adiciona aos resultados
            results['algorithms'][algo_name] = {
                'avg_time': statistics.mean(execution_times),
                'min_time': min(execution_times),
                'max_time': max(execution_times),
                'avg_comparisons': statistics.mean(comparison_counts),
                'avg_swaps': statistics.mean(swap_counts)
            }

            print(f"  Tempo médio: {results['algorithms'][algo_name]['avg_time']:.2f} ms")

        # Salva os resultados em um arquivo
        save_results_to_file(results, args.output)
        print(f"Resultados salvos em {args.output}")
        
        # Gera o gráfico de desempenho se solicitado
        if args.graph:
            try:
                print("Gerando gráfico de desempenho...")
                generate_performance_graph(results, args.graph_output)
                print(f"Gráfico salvo em {args.graph_output}")
            except Exception as e:
                print(f"Erro ao gerar o gráfico: {e}")
                logger.error(f"Erro ao gerar o gráfico: {e}", exc_info=True)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
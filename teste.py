from knapsack_ga import GeneticKnapsack
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict


def test_example_from_prompt():
    """
    Testa o exemplo citado no enunciado
    """
    print("\n==== Teste do Exemplo do Enunciado ====")
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    capacity = 5
    
    # Sabemos que a solução ótima deve incluir os itens 0 e 1 (pesos 2 e 3)
    # com valor total 7 e peso total 5
    
    ga = GeneticKnapsack(
        weights=weights,
        values=values,
        capacity=capacity,
        pop_size=100,
        generations=50,
        crossover_rate=0.8,
        mutation_rate=0.1
    )
    
    result = ga.run()
    
    print(f"Pesos: {weights}")
    print(f"Valores: {values}")
    print(f"Capacidade: {capacity}")
    print("\nResultados:")
    print(f"Itens selecionados: {result['selected_items']} (índices começando em 0)")
    print(f"Valor total: {result['total_value']}")
    print(f"Peso total: {result['total_weight']}")
    print(f"Tempo de execução: {result['execution_time']:.4f} segundos")
    
    return result


def test_larger_example():
    """
    Testa um exemplo maior
    """
    print("\n==== Teste com Exemplo Maior ====")
    weights = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    values = [15, 25, 35, 45, 55, 65, 75, 85, 95]
    capacity = 150
    
    ga = GeneticKnapsack(
        weights=weights,
        values=values,
        capacity=capacity,
        pop_size=200,
        generations=100,
        crossover_rate=0.8,
        mutation_rate=0.1
    )
    
    result = ga.run()
    
    print(f"Pesos: {weights}")
    print(f"Valores: {values}")
    print(f"Capacidade: {capacity}")
    print("\nResultados:")
    print(f"Itens selecionados: {result['selected_items']} (índices começando em 0)")
    print(f"Valor total: {result['total_value']}")
    print(f"Peso total: {result['total_weight']}")
    print(f"Tempo de execução: {result['execution_time']:.4f} segundos")
    
    # Exibe a relação valor/peso dos itens selecionados
    if result['selected_items']:
        print("\nRelação valor/peso dos itens selecionados:")
        for idx in result['selected_items']:
            print(f"Item {idx}: {values[idx]/weights[idx]:.2f} (valor={values[idx]}, peso={weights[idx]})")
    
    return result


def test_random_instances(num_tests=5, min_items=5, max_items=20):
    """
    Testa o algoritmo com instâncias aleatórias
    
    Args:
        num_tests: Número de testes a realizar
        min_items: Mínimo número de itens por instância
        max_items: Máximo número de itens por instância
    """
    print(f"\n==== Teste com {num_tests} Instâncias Aleatórias ====")
    
    results = []
    
    for test in range(num_tests):
        # Gera um número aleatório de itens
        num_items = np.random.randint(min_items, max_items + 1)
        
        # Gera pesos e valores aleatórios
        weights = np.random.randint(1, 100, size=num_items).tolist()
        values = np.random.randint(1, 100, size=num_items).tolist()
        
        # Define capacidade como ~40% da soma dos pesos
        capacity = int(0.4 * sum(weights))
        
        print(f"\nInstância {test+1}:")
        print(f"Número de itens: {num_items}")
        print(f"Capacidade da mochila: {capacity}")
        
        # Configura o algoritmo genético
        ga = GeneticKnapsack(
            weights=weights,
            values=values,
            capacity=capacity,
            pop_size=100,
            generations=100,
            crossover_rate=0.8,
            mutation_rate=0.1
        )
        
        # Executa o algoritmo
        start = time.time()
        result = ga.run()
        execution_time = time.time() - start
        
        # Resultados
        print(f"Valor total: {result['total_value']}")
        print(f"Peso total: {result['total_weight']} / {capacity}")
        print(f"Número de itens selecionados: {len(result['selected_items'])}")
        print(f"Tempo de execução: {execution_time:.4f} segundos")
        
        # Armazena resultado
        results.append({
            'instance': test+1,
            'num_items': num_items,
            'capacity': capacity,
            'total_value': result['total_value'],
            'total_weight': result['total_weight'],
            'weight_ratio': result['total_weight'] / capacity,
            'items_selected': len(result['selected_items']),
            'execution_time': execution_time
        })
    
    # Cria um DataFrame para análise
    df = pd.DataFrame(results)
    print("\nResumo dos testes:")
    print(df)
    
    # Análise do tempo de execução vs. número de itens
    plt.figure(figsize=(10, 6))
    plt.scatter(df['num_items'], df['execution_time'])
    plt.xlabel('Número de Itens')
    plt.ylabel('Tempo de Execução (s)')
    plt.title('Tempo de Execução vs. Número de Itens')
    plt.grid(True)
    plt.show()
    
    return results


def compare_parameter_settings():
    """
    Compara diferentes configurações de parâmetros do algoritmo genético
    """
    print("\n==== Comparação de Configurações de Parâmetros ====")
    
    # Problema a ser resolvido
    weights = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    values = [15, 25, 35, 45, 55, 65, 75, 85, 95]
    capacity = 150
    
    # Diferentes configurações a testar
    configurations = [
        {"pop_size": 50, "generations": 50, "crossover_rate": 0.7, "mutation_rate": 0.05},
        {"pop_size": 50, "generations": 50, "crossover_rate": 0.8, "mutation_rate": 0.1},
        {"pop_size": 100, "generations": 100, "crossover_rate": 0.7, "mutation_rate": 0.05},
        {"pop_size": 100, "generations": 100, "crossover_rate": 0.8, "mutation_rate": 0.1},
        {"pop_size": 200, "generations": 50, "crossover_rate": 0.8, "mutation_rate": 0.1},
        {"pop_size": 50, "generations": 200, "crossover_rate": 0.8, "mutation_rate": 0.1},
    ]
    
    results = []
    
    for i, config in enumerate(configurations):
        print(f"\nConfiguracao {i+1}:")
        print(f"População: {config['pop_size']}, Gerações: {config['generations']}")
        print(f"Taxa de Crossover: {config['crossover_rate']}, Taxa de Mutação: {config['mutation_rate']}")
        
        # Configura o algoritmo
        ga = GeneticKnapsack(
            weights=weights,
            values=values,
            capacity=capacity,
            **config
        )
        
        # Executa o algoritmo
        start = time.time()
        result = ga.run()
        execution_time = time.time() - start
        
        # Armazena resultados
        results.append({
            'config': i+1,
            'pop_size': config['pop_size'],
            'generations': config['generations'],
            'crossover_rate': config['crossover_rate'],
            'mutation_rate': config['mutation_rate'],
            'total_value': result['total_value'],
            'total_weight': result['total_weight'],
            'items_selected': len(result['selected_items']),
            'execution_time': execution_time
        })
        
        # Exibe resultados
        print(f"Valor total: {result['total_value']}")
        print(f"Peso total: {result['total_weight']} / {capacity}")
        print(f"Itens selecionados: {result['selected_items']}")
        print(f"Tempo de execução: {execution_time:.4f} segundos")
    
    # Cria um DataFrame para análise
    df = pd.DataFrame(results)
    print("\nResumo das configurações:")
    print(df[['config', 'pop_size', 'generations', 'total_value', 'execution_time']])
    
    # Plota comparação de valor vs. tempo
    plt.figure(figsize=(10, 6))
    plt.scatter(df['execution_time'], df['total_value'])
    
    # Anota pontos com número da configuração
    for i, row in df.iterrows():
        plt.annotate(f"Config {row['config']}", 
                    (row['execution_time'], row['total_value']),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Tempo de Execução (s)')
    plt.ylabel('Valor Total Obtido')
    plt.title('Valor Total vs. Tempo de Execução para Diferentes Configurações')
    plt.grid(True)
    plt.show()
    
    return results


def compare_with_brute_force(max_items=15):
    """
    Compara o algoritmo genético com a solução por força bruta
    
    Args:
        max_items: Número máximo de itens para o teste de força bruta
    """
    print("\n==== Comparação com Solução por Força Bruta ====")
    
    # Gera um problema aleatório
    num_items = np.random.randint(10, max_items + 1)
    weights = np.random.randint(1, 50, size=num_items).tolist()
    values = np.random.randint(1, 50, size=num_items).tolist()
    capacity = int(0.4 * sum(weights))
    
    print(f"Número de itens: {num_items}")
    print(f"Capacidade da mochila: {capacity}")
    
    # Solução por força bruta
    print("\nCalculando solução por força bruta...")
    start = time.time()
    best_value, best_combination = brute_force_knapsack(weights, values, capacity)
    bf_time = time.time() - start
    
    # Converte para formato de índices
    bf_selected = [i for i, bit in enumerate(best_combination) if bit == 1]
    bf_weight = sum(weights[i] for i in bf_selected)
    
    print(f"Solução ótima (força bruta):")
    print(f"Valor total: {best_value}")
    print(f"Peso total: {bf_weight} / {capacity}")
    print(f"Itens selecionados: {bf_selected}")
    print(f"Tempo de execução: {bf_time:.6f} segundos")
    
    # Solução por algoritmo genético
    print("\nCalculando solução por algoritmo genético...")
    ga = GeneticKnapsack(
        weights=weights,
        values=values,
        capacity=capacity,
        pop_size=100,
        generations=100,
        crossover_rate=0.8,
        mutation_rate=0.1
    )
    
    start = time.time()
    result = ga.run()
    ga_time = time.time() - start
    
    print(f"Solução por algoritmo genético:")
    print(f"Valor total: {result['total_value']}")
    print(f"Peso total: {result['total_weight']} / {capacity}")
    print(f"Itens selecionados: {result['selected_items']}")
    print(f"Tempo de execução: {ga_time:.6f} segundos")
    
    # Comparação
    print("\nComparação:")
    print(f"Diferença de valor: {best_value - result['total_value']} ({(best_value - result['total_value'])/best_value*100:.2f}%)")
    print(f"Aceleração: {bf_time/ga_time:.2f}x")
    
    return {
        'brute_force': {
            'value': best_value,
            'weight': bf_weight,
            'selected': bf_selected,
            'time': bf_time
        },
        'genetic': {
            'value': result['total_value'],
            'weight': result['total_weight'],
            'selected': result['selected_items'],
            'time': ga_time
        },
        'comparison': {
            'value_diff': best_value - result['total_value'],
            'value_diff_pct': (best_value - result['total_value'])/best_value*100 if best_value > 0 else 0,
            'speedup': bf_time/ga_time
        }
    }


def brute_force_knapsack(weights: List[int], values: List[int], capacity: int) -> tuple:
    """
    Resolve o problema da mochila por força bruta testando todas as combinações
    
    Args:
        weights: Lista de pesos dos itens
        values: Lista de valores dos itens
        capacity: Capacidade da mochila
        
    Returns:
        Tupla (melhor_valor, melhor_combinação)
    """
    n = len(weights)
    best_value = 0
    best_combination = [0] * n
    
    # Total de combinações: 2^n
    total_combinations = 2**n
    
    for i in range(total_combinations):
        # Converte o número i para representação binária
        binary = format(i, f'0{n}b')
        combination = [int(bit) for bit in binary]
        
        # Calcula peso e valor total
        total_weight = sum(weights[j] for j in range(n) if combination[j] == 1)
        total_value = sum(values[j] for j in range(n) if combination[j] == 1)
        
        # Verifica se é uma solução válida e melhor que a atual
        if total_weight <= capacity and total_value > best_value:
            best_value = total_value
            best_combination = combination
    
    return best_value, best_combination


if __name__ == "__main__":
    # Executa os testes
    test_example_from_prompt()
    test_larger_example()
    test_random_instances(num_tests=3)
    compare_parameter_settings()
    
    # Comparação com força bruta (limite de itens para evitar explosão combinatória)
    compare_with_brute_force(max_items=12)
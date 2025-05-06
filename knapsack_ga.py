import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Dict

class GeneticKnapsack:
    """
    Solução para o problema da Mochila 0/1 usando Algoritmo Genético
    """
    
    def __init__(self, weights: List[int], values: List[int], capacity: int,
                 pop_size: int = 100, generations: int = 100, 
                 crossover_rate: float = 0.8, mutation_rate: float = 0.1):
        """
        Inicializa o algoritmo genético
        
        Args:
            weights: Lista com os pesos dos itens
            values: Lista com os valores dos itens
            capacity: Capacidade máxima da mochila
            pop_size: Tamanho da população
            generations: Número de gerações
            crossover_rate: Taxa de crossover
            mutation_rate: Taxa de mutação
        """
        self.weights = weights
        self.values = values
        self.capacity = capacity
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.n_items = len(weights)
        
        # Validação dos dados de entrada
        if len(weights) != len(values):
            raise ValueError("As listas de pesos e valores devem ter o mesmo tamanho")
        if any(w <= 0 for w in weights) or any(v <= 0 for v in values):
            raise ValueError("Pesos e valores devem ser positivos")
        if capacity <= 0:
            raise ValueError("A capacidade da mochila deve ser positiva")
    
    def initialize_population(self) -> np.ndarray:
        """
        Inicializa a população com soluções aleatórias
        
        Returns:
            População inicial (matriz binária)
        """
        return np.random.randint(0, 2, size=(self.pop_size, self.n_items))
    
    def fitness(self, solution: np.ndarray) -> Tuple[float, int, int]:
        """
        Calcula o fitness de uma solução
        
        Args:
            solution: Vetor binário representando uma solução
            
        Returns:
            Tripla (fitness, valor total, peso total)
        """
        total_weight = np.sum(solution * self.weights)
        total_value = np.sum(solution * self.values)
        
        # Penalização para soluções que excedem a capacidade
        if total_weight > self.capacity:
            fitness = 0  # Forte penalização
        else:
            fitness = total_value
            
        return fitness, total_value, total_weight
    
    def evaluate_population(self, population: np.ndarray) -> Tuple[np.ndarray, List[Tuple[float, int, int]]]:
        """
        Avalia toda a população
        
        Args:
            population: Matriz de soluções
            
        Returns:
            Array de fitness e lista com detalhes (fitness, valor, peso)
        """
        fitness_values = np.zeros(self.pop_size)
        details = []
        
        for i in range(self.pop_size):
            fitness, value, weight = self.fitness(population[i])
            fitness_values[i] = fitness
            details.append((fitness, value, weight))
            
        return fitness_values, details
    
    def selection(self, population: np.ndarray, fitness_values: np.ndarray) -> np.ndarray:
        """
        Seleção de indivíduos por torneio
        
        Args:
            population: População atual
            fitness_values: Valores de fitness da população
            
        Returns:
            Nova população após seleção
        """
        tournament_size = 3
        new_population = np.zeros((self.pop_size, self.n_items), dtype=int)
        
        for i in range(self.pop_size):
            # Seleciona indivíduos aleatórios para o torneio
            candidates = np.random.choice(self.pop_size, tournament_size, replace=False)
            # Escolhe o melhor entre os candidatos
            best_candidate = candidates[np.argmax(fitness_values[candidates])]
            new_population[i] = population[best_candidate]
            
        return new_population
    
    def crossover(self, population: np.ndarray) -> np.ndarray:
        """
        Aplica crossover (recombinação) nos indivíduos
        
        Args:
            population: População após seleção
            
        Returns:
            População após crossover
        """
        new_population = population.copy()
        
        for i in range(0, self.pop_size, 2):
            if i + 1 < self.pop_size and np.random.random() < self.crossover_rate:
                # Ponto de corte para o crossover
                crossover_point = np.random.randint(1, self.n_items)
                
                # Troca os genes após o ponto de corte
                temp = new_population[i, crossover_point:].copy()
                new_population[i, crossover_point:] = new_population[i+1, crossover_point:]
                new_population[i+1, crossover_point:] = temp
                
        return new_population
    
    def mutation(self, population: np.ndarray) -> np.ndarray:
        """
        Aplica mutação nos indivíduos
        
        Args:
            population: População após crossover
            
        Returns:
            População após mutação
        """
        for i in range(self.pop_size):
            for j in range(self.n_items):
                if np.random.random() < self.mutation_rate:
                    # Inverte o bit (0->1 ou 1->0)
                    population[i, j] = 1 - population[i, j]
                    
        return population
    
    def elitism(self, population: np.ndarray, new_population: np.ndarray, 
                fitness_values: np.ndarray, new_fitness_values: np.ndarray, 
                elite_size: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preserva os melhores indivíduos da geração anterior
        
        Args:
            population: População anterior
            new_population: Nova população
            fitness_values: Fitness da população anterior
            new_fitness_values: Fitness da nova população
            elite_size: Quantidade de indivíduos elite a preservar
            
        Returns:
            Nova população com elitismo e seus valores de fitness
        """
        # Índices dos melhores indivíduos da população anterior
        elite_indices = np.argsort(fitness_values)[-elite_size:]
        
        # Índices dos piores indivíduos da nova população
        worst_indices = np.argsort(new_fitness_values)[:elite_size]
        
        # Substitui os piores da nova população pelos melhores da antiga
        for i, elite_idx in enumerate(elite_indices):
            new_population[worst_indices[i]] = population[elite_idx]
            new_fitness_values[worst_indices[i]] = fitness_values[elite_idx]
            
        return new_population, new_fitness_values
    
    def run(self) -> Dict:
        """
        Executa o algoritmo genético
        
        Returns:
            Dicionário com resultados e estatísticas
        """
        start_time = time.time()
        
        # Inicialização
        population = self.initialize_population()
        fitness_values, details = self.evaluate_population(population)
        
        # Rastreia o melhor de cada geração
        best_fitness_history = []
        avg_fitness_history = []
        
        best_solution = None
        best_fitness = 0
        best_value = 0
        best_weight = 0
        
        # Loop principal - evolução por gerações
        for generation in range(self.generations):
            # Aplica operadores genéticos
            selected_population = self.selection(population, fitness_values)
            crossover_population = self.crossover(selected_population)
            mutated_population = self.mutation(crossover_population)
            
            # Avalia nova população
            new_fitness_values, new_details = self.evaluate_population(mutated_population)
            
            # Aplica elitismo
            mutated_population, new_fitness_values = self.elitism(
                population, mutated_population, fitness_values, new_fitness_values
            )
            
            # Atualiza população e fitness
            population = mutated_population
            fitness_values = new_fitness_values
            details = new_details
            
            # Registra estatísticas
            gen_best_idx = np.argmax(fitness_values)
            gen_best_fitness = fitness_values[gen_best_idx]
            gen_best_value = details[gen_best_idx][1]
            gen_best_weight = details[gen_best_idx][2]
            
            best_fitness_history.append(gen_best_fitness)
            avg_fitness_history.append(np.mean(fitness_values))
            
            # Atualiza melhor solução global
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_solution = population[gen_best_idx].copy()
                best_value = gen_best_value
                best_weight = gen_best_weight
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Prepara resultado final
        selected_items = [i for i, bit in enumerate(best_solution) if bit == 1]
        
        result = {
            "solution": best_solution,
            "selected_items": selected_items,
            "total_value": best_value,
            "total_weight": best_weight,
            "capacity": self.capacity,
            "execution_time": execution_time,
            "generations": self.generations,
            "best_fitness_history": best_fitness_history,
            "avg_fitness_history": avg_fitness_history
        }
        
        return result
    
    def plot_progress(self, best_fitness_history: List[float], avg_fitness_history: List[float]) -> None:
        """
        Plota o progresso do algoritmo ao longo das gerações
        
        Args:
            best_fitness_history: Histórico do melhor fitness por geração
            avg_fitness_history: Histórico do fitness médio por geração
        """
        plt.figure(figsize=(10, 6))
        plt.plot(best_fitness_history, label='Melhor Fitness')
        plt.plot(avg_fitness_history, label='Fitness Médio')
        plt.xlabel('Geração')
        plt.ylabel('Fitness')
        plt.title('Evolução do Fitness ao Longo das Gerações')
        plt.legend()
        plt.grid(True)
        plt.show()


# Função para demonstrar o uso do algoritmo
def solve_knapsack_example():
    """
    Exemplo de uso do algoritmo genético para o problema da mochila
    """
    # Exemplo do enunciado
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    capacity = 5
    
    # Exemplo maior
    # weights = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    # values = [15, 25, 35, 45, 55, 65, 75, 85, 95]
    # capacity = 150
    
    # Configuração do algoritmo
    pop_size = 50
    generations = 100
    crossover_rate = 0.8
    mutation_rate = 0.1
    
    # Instancia e executa o algoritmo
    ga = GeneticKnapsack(
        weights=weights,
        values=values,
        capacity=capacity,
        pop_size=pop_size,
        generations=generations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate
    )
    
    # Executa o algoritmo
    result = ga.run()
    
    # Imprime resultados
    print("Problema da Mochila - Algoritmo Genético")
    print(f"Pesos: {weights}")
    print(f"Valores: {values}")
    print(f"Capacidade: {capacity}")
    print("\nResultados:")
    print(f"Itens selecionados: {result['selected_items']} (índices começando em 0)")
    print(f"Valor total: {result['total_value']}")
    print(f"Peso total: {result['total_weight']}")
    print(f"Tempo de execução: {result['execution_time']:.4f} segundos")
    
    # Plota o progresso
    ga.plot_progress(result['best_fitness_history'], result['avg_fitness_history'])
    
    return result


if __name__ == "__main__":
    solve_knapsack_example()
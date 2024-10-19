import random
import math

# 適應度函數： f(x) = x^2
def fitness(x):
    return x**2

# 生成初始種群
def generate_population(size, chromosome_length):
    return [random.randint(0, 2**chromosome_length - 1) for _ in range(size)]

# 選擇個體 (輪盤賭選擇法)
def select(population):
    total_fitness = sum(fitness(ind) for ind in population)
    pick = random.uniform(-961, total_fitness)
    current = 0
    for ind in population:
        current += fitness(ind)
        if current > pick:
            return ind

def select_elitism(population, elite_size=2):
    # 按適應度排序
    sorted_population = sorted(population, key=lambda ind: fitness(ind), reverse=True)
    # 選擇前 elite_size 個菁英個體
    return sorted_population[:elite_size]


# 交叉
def crossover(parent1, parent2, chromosome_length):
    crossover_point = random.randint(1, chromosome_length - 1)
    mask = (1 << crossover_point) - 1
    child1 = (parent1 & mask) | (parent2 & ~mask)
    child2 = (parent2 & mask) | (parent1 & ~mask)
    return child1, child2

# 變異
def mutate(individual, mutation_rate, chromosome_length):
    for i in range(chromosome_length):
        if random.random() < mutation_rate:
            individual ^= (1 << i)
    return individual

# 遺傳演算法主函數
def genetic_algorithm(population_size, chromosome_length, generations, crossover_rate, mutation_rate):
    population = generate_population(population_size, chromosome_length)
    
    for generation in range(generations):
        new_population = []
        
        # 產生新種群
        while len(new_population) < population_size:
            parent1 = select(population)
            parent2 = select(population)
            #parent1,parent2 = select_elitism(population)
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2, chromosome_length)
            else:
                child1, child2 = parent1, parent2
            child1 = mutate(child1, mutation_rate, chromosome_length)
            child2 = mutate(child2, mutation_rate, chromosome_length)
            new_population.extend([child1, child2])
        
        population = new_population[:population_size]
    
    # 找出最佳解
    best_individual = max(population, key=fitness)
    return best_individual, fitness(best_individual)

# 執行遺傳演算法 
best_ind, best_fit = genetic_algorithm(
    population_size=10,  # N
    chromosome_length=5,  # B
    #pop_dimension=1, D = 1 ; x vector in f(x) is 1 * D array
    generations=1000,  # Iteration
    crossover_rate=0.8,  # cr
    mutation_rate=0.01 # mr
)
print(f"最佳解: {best_ind}, 適應度: {best_fit}")

import numpy as np
import random
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# 初始化種群
def initialize_population(pop_size, num_features):
    return [np.random.randint(2, size=num_features).tolist() for _ in range(pop_size)]

# 適應度函數
def fitness(individual, X, y):
    selected_features = [i for i, bit in enumerate(individual) if bit == 1]
    
    if len(selected_features) == 0:
        return 0
    
    X_selected = X[:, selected_features]
    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    scores = cross_val_score(model, X_selected, y, cv=5)
    accuracy = scores.mean()
    
    return accuracy - 0.01 * len(selected_features)

# 選擇父代（競爭選擇法）
def select_tournament(population, X, y, tournament_size=3):
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda ind: fitness(ind, X, y))

# 交叉操作
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

# 變異操作
def mutate(individual, mutation_rate=0.01):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual

# 遺傳演算法主函數
def genetic_algorithm(X, y, num_generations=50, pop_size=20, elite_size=2):
    num_features = X.shape[1]
    population = initialize_population(pop_size, num_features)
    
    for generation in range(num_generations):
        # 計算每個個體的適應度
        population = sorted(population, key=lambda ind: fitness(ind, X, y), reverse=True)
        
        # 保留菁英
        next_generation = population[:elite_size]
        
        # 生成新一代
        while len(next_generation) < pop_size:
            parent1 = select_tournament(population, X, y)
            parent2 = select_tournament(population, X, y)
            child1, child2 = crossover(parent1, parent2)
            next_generation.append(mutate(child1))
            if len(next_generation) < pop_size:
                next_generation.append(mutate(child2))
        
        population = next_generation
        print("generation : ",generation)
        print("population : \n",population)
        
    # 返回最優解
    best_individual = max(population, key=lambda ind: fitness(ind, X, y))
    return best_individual

# 運行範例
X = np.random.rand(100, 10)  # 假設有 10 個特徵
y = np.random.randint(2, size=100)  # 二元標籤
best_features = genetic_algorithm(X, y)
print("選擇的最佳特徵組合:", best_features)

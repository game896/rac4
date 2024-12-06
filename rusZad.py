import numpy as np
import random
import matplotlib.pyplot as plt
from math import sin, cos, pi


# Beale's function
def beale_function(x, y):
    term1 = (1.5 - x + x * y) ** 2
    term2 = (2.25 - x + x * y**2) ** 2
    term3 = (2.625 - x + x * y**3) ** 2
    return term1 + term2 + term3


# Generate initial population (real-coded)
def generate_initial_population(chrom_size, min_val, max_val, population_size):
    return np.random.uniform(min_val, max_val, (population_size, chrom_size))


# Crossover - blend crossover
def crossover(parents):
    children = []
    for parent1, parent2 in parents:
        alpha = random.random()
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        children.append(child1)
        children.append(child2)
    return children


# Mutation (Gaussian mutation)
def mutation(population, mutation_rate, mutation_std):
    mutated_population = []
    for individual in population:
        if random.random() < mutation_rate:
            mutation = np.random.normal(0, mutation_std, size=individual.shape)
            mutated_population.append(individual + mutation)
        else:
            mutated_population.append(individual)
    return np.array(mutated_population)


# Elitism
def elitism(ranked_population, costs, elite_rate):
    population_size = len(ranked_population)
    elite_count = int(population_size * elite_rate)
    
    elite_indices = np.argsort(costs)[:elite_count]
    
    # Pretvaranje population u numpy array (ako već nije)
    ranked_population = np.array(ranked_population)
    
    # Indeksiranje sa elite_indices
    elite_population = ranked_population[elite_indices]
    
    return elite_population


# Fitness function
def fitness_function(chromosome):
    return beale_function(chromosome[0], chromosome[1])


# Rank chromosomes based on fitness
# Rank chromosomes based on fitness
def rank_chromosomes(population):
    costs = [fitness_function(ind) for ind in population]  # Računanje troškova
    ranked_population = [x for _, x in sorted(zip(costs, population), key=lambda pair: pair[0])]  # Sortiranje po troškovima
    return ranked_population, sorted(costs)  # Vraćanje sortiranih populacija i troškova



# Genetic Algorithm
def genetic_algorithm(population_size, chrom_size, min_val, max_val, max_iter, mutation_rate, mutation_std, elite_rate):
    population = generate_initial_population(chrom_size, min_val, max_val, population_size)

    avg_list = []
    best_list = []

    for _ in range(max_iter):
        # Rank population
        ranked_population, costs = rank_chromosomes(population)

        # Get best and average cost
        best_cost = costs[0]
        avg_cost = np.mean(costs)

        avg_list.append(avg_cost)
        best_list.append(best_cost)

        # Elitism
        elite_population = elitism(ranked_population, costs, elite_rate)

        # Select parents (top 50%)
        parents = ranked_population[:population_size // 2]

        # Crossover
        children = crossover(list(zip(parents[::2], parents[1::2])))

        # Mutation
        children = mutation(np.array(children), mutation_rate, mutation_std)

        # Next generation (elitism + children)
        population = np.vstack([elite_population, children])

        if best_cost < 1e-6:  # Stop if the solution is good enough
            break

    return population, best_list, avg_list


# Parameters
population_size = 45
chrom_size = 2
min_val = -30
max_val = 30
max_iter = 1000
mutation_rate = 0.1
mutation_std = 1.0
elite_rate = 0.08

# Run the genetic algorithm
population, best_list, avg_list = genetic_algorithm(population_size, chrom_size, min_val, max_val, max_iter, mutation_rate, mutation_std, elite_rate)

# Plot the results on a contour plot

# Create a grid for plotting Beale's function
x = np.linspace(-30, 30, 400)
y = np.linspace(-30, 30, 400)
X, Y = np.meshgrid(x, y)
Z = beale_function(X, Y)

# Contour plot (filled contours)
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis')  # koristimo contourf za boje
plt.colorbar(contour, label='Function Value')  # Dodajemo legendu za boje

# Plot best solution found
best_solution = population[np.argmin([fitness_function(ind) for ind in population])]
plt.scatter(best_solution[0], best_solution[1], color='red', label='Best Solution', zorder=5)
print(best_solution[0], best_solution[1])

plt.title("Contour Plot of Beale's Function")
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# Plot the best and average costs over generations
plt.figure(figsize=(8, 6))
plt.plot(best_list, label='Best Cost')
plt.plot(avg_list, label='Average Cost')
plt.xlabel('Generation')
plt.ylabel('Cost')
plt.legend()
plt.title("Fitness Over Generations")
plt.show()

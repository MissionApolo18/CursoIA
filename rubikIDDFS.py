import time
import psutil
import matplotlib.pyplot as plt

# Clase Puzzle (Ejemplo simplificado)
class Puzzle:
    def __init__(self):
        self.state = 'solved_state'

    def shuffle(self, moves):
        self.state = 'shuffled_state'  # Ejemplo simple

    def to_list(self):
        return list(self.state)

    def is_solved(self):
        return self.state == 'solved_state'

    def possible_moves(self):
        # Ejemplo de movimientos posibles
        return ['U', 'U\'', 'R', 'R\'', 'F', 'F\'', 'D', 'D\'', 'L', 'L\'', 'B', 'B\'']

    def apply_move(self, move):
        # Aplica un movimiento y devuelve el nuevo estado (ejemplo simplificado)
        return self.state + move

# Función IDDFS
def iddfs_solve(puzzle, max_depth):
    start_time = time.time()
    process = psutil.Process()
    nodes_expanded = 0
    times = []
    memories = []

    def dls(current_state, path, depth):
        nonlocal nodes_expanded
        nodes_expanded += 1

        current_time = time.time() - start_time
        current_memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB
        times.append(current_time)
        memories.append(current_memory)

        if depth == 0 and puzzle.is_solved():
            return path
        if depth > 0:
            for move in puzzle.possible_moves():
                new_state = puzzle.apply_move(move)
                if new_state not in visited:
                    visited.add(new_state)
                    result = dls(new_state, path + [move], depth - 1)
                    if result is not None:
                        return result
        return None

    for depth in range(max_depth + 1):
        visited = set([puzzle.state])
        result = dls(puzzle.state, [], depth)
        if result is not None:
            return result, times, memories

    return None, times, memories

# Función para graficar
def plot_metrics(times, memories):
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Iteraciones')
    ax1.set_ylabel('Tiempo (s)', color=color)
    ax1.plot(range(len(times)), times, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Memoria (MB)', color=color)
    ax2.plot(range(len(memories)), memories, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Tiempo y Memoria Consumida por IDDFS')
    plt.show()

# Ejemplo de uso
ordenado = Puzzle()
print("rompecabezas ordenado:", ordenado)
lista_ordenada = ordenado.to_list()
print("el rompecabezas ordenado como una lista:\n", lista_ordenada)

from random import seed
seed(2019)
desordenado = Puzzle()
desordenado.shuffle(5)
print("rompecabezas desordenado:", desordenado)
lista_desordenada = desordenado.to_list()
print("el rompecabezas desordenado como una lista:\n", lista_desordenada)

solution, times, memories = iddfs_solve(desordenado, max_depth=10)

if solution:
    print(f'Solución encontrada: {solution}')
else:
    print('No se encontró solución dentro de la profundidad máxima')

plot_metrics(times, memories)

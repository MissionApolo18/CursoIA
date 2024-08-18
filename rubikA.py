import heapq
import time
import psutil
import matplotlib.pyplot as plt
from random import seed

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

# Creamos un rompecabezas ordenado
ordenado = Puzzle()
print("rompecabezas ordenado:", ordenado)
lista_ordenada = ordenado.to_list()
print("el rompecabezas ordenado como una lista:\n", lista_ordenada)

# Si desordenamos el rompecabezas su lista ya no estará ordenada
seed(2019)
desordenado = Puzzle()
desordenado.shuffle(5)
print("rompecabezas desordenado:", desordenado)
lista_desordenada = desordenado.to_list()
print("el rompecabezas desordenado como una lista:\n", lista_desordenada)

def h1(p_1, p_2):
    # cuenta el número de fichas que no están en orden
    return sum(1 for a, b in zip(p_1.to_list(), p_2.to_list()) if a != b)

print("ordenado:", ordenado)
print("desordenado:", desordenado)
print("número de fichas fuera de lugar:", h1(ordenado, desordenado))

# Función A*
def astar_solve(puzzle, max_depth):
    start_time = time.time()
    process = psutil.Process()

    open_set = []
    heapq.heappush(open_set, (0, puzzle.state, []))
    visited = set([puzzle.state])

    nodes_expanded = 0
    times = []
    memories = []

    while open_set:
        current_f, current_state, path = heapq.heappop(open_set)
        nodes_expanded += 1

        current_time = time.time() - start_time
        current_memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB

        times.append(current_time)
        memories.append(current_memory)

        if len(path) > max_depth:
            break

        if puzzle.is_solved():
            return path, times, memories

        for move in puzzle.possible_moves():
            new_state = puzzle.apply_move(move)
            if new_state not in visited:
                visited.add(new_state)
                new_path = path + [move]
                heuristic_cost = h1(puzzle, Puzzle())  # Aquí puedes ajustar la función heurística
                heapq.heappush(open_set, (len(new_path) + heuristic_cost, new_state, new_path))

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
    plt.title('Tiempo y Memoria Consumida por A*')
    plt.show()

# Ejemplo de uso
solution, times, memories = astar_solve(desordenado, max_depth=10)

if solution:
    print(f'Solución encontrada: {solution}')
else:
    print('No se encontró solución dentro de la profundidad máxima')

plot_metrics(times, memories)

import time
import psutil
import matplotlib.pyplot as plt
from collections import deque
import heapq

# Definimos la clase Puzzle
class Puzzle:
    def __init__(self):
        # Inicializa el puzzle en su estado resuelto
        self.state = tuple(range(1, 10))

    def is_solved(self):
        # Devuelve True si el puzzle está en el estado resuelto
        return self.state == tuple(range(1, 10))

    def possible_moves(self):
        # Devuelve una lista de posibles movimientos (este es un ejemplo)
        return ['U', 'D', 'L', 'R']

    def apply_move(self, move):
        # Aplica un movimiento y devuelve el nuevo estado (este es un ejemplo)
        new_state = list(self.state)
        # Simple move simulation (rotate first two elements)
        new_state[0], new_state[1] = new_state[1], new_state[0]
        return tuple(new_state)

    def __lt__(self, other):
        return self.state < other.state

def bfs_solve(puzzle, max_depth):
    start_time = time.time()
    process = psutil.Process()

    queue = deque([(puzzle.state, [])])
    visited = set([puzzle.state])

    depth = 0
    nodes_expanded = 0
    max_memory = 0

    while queue:
        if depth > max_depth:
            break

        current_state, path = queue.popleft()
        nodes_expanded += 1

        current_memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB
        max_memory = max(max_memory, current_memory)

        if Puzzle().is_solved():
            return path, max_memory

        for move in Puzzle().possible_moves():
            new_state = Puzzle().apply_move(move)
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, path + [move]))

        depth += 1

    return None, max_memory

def astar_solve(puzzle, max_depth):
    def heuristic(state):
        # Heurística simple: contar el número de fichas que no están en su lugar
        return sum(1 for i, v in enumerate(state) if v != i + 1)

    start_time = time.time()
    process = psutil.Process()

    open_list = []
    heapq.heappush(open_list, (0 + heuristic(puzzle.state), 0, puzzle.state, []))
    visited = set()

    nodes_expanded = 0
    max_memory = 0

    while open_list:
        _, g, current_state, path = heapq.heappop(open_list)
        nodes_expanded += 1

        current_memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB
        max_memory = max(max_memory, current_memory)

        if Puzzle().is_solved():
            return path, max_memory

        if current_state in visited:
            continue

        visited.add(current_state)

        for move in Puzzle().possible_moves():
            new_state = Puzzle().apply_move(move)
            if new_state not in visited:
                heapq.heappush(open_list, (g + 1 + heuristic(new_state), g + 1, new_state, path + [move]))

    return None, max_memory

def iddfs_solve(puzzle, max_depth):
    def dls(state, path, depth):
        if depth == 0:
            return None
        if Puzzle().is_solved():
            return path

        for move in Puzzle().possible_moves():
            new_state = Puzzle().apply_move(move)
            if new_state not in visited:
                visited.add(new_state)
                result = dls(new_state, path + [move], depth - 1)
                if result is not None:
                    return result
        return None

    start_time = time.time()
    process = psutil.Process()

    nodes_expanded = 0
    max_memory = 0

    for depth in range(max_depth + 1):
        visited = set([puzzle.state])
        result = dls(puzzle.state, [], depth)

        current_memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB
        max_memory = max(max_memory, current_memory)

        if result is not None:
            return result, max_memory

    return None, max_memory

def plot_combined_metrics(memory_bfs, memory_astar, memory_iddfs):
    algorithms = ['BFS', 'A*', 'IDDFS']
    memory_usage = [memory_bfs, memory_astar, memory_iddfs]

    plt.bar(algorithms, memory_usage, color=['blue', 'orange', 'green'])
    plt.xlabel('Algoritmo')
    plt.ylabel('Memoria Máxima Consumida (MB)')
    plt.title('Comparación de Memoria Consumida por Algoritmos')
    plt.show()

# Ejemplo de uso
puzzle = Puzzle()

solution_bfs, memory_bfs = bfs_solve(puzzle, max_depth=10)
solution_astar, memory_astar = astar_solve(puzzle, max_depth=10)
solution_iddfs, memory_iddfs = iddfs_solve(puzzle, max_depth=10)

plot_combined_metrics(memory_bfs, memory_astar, memory_iddfs)

import time
import psutil
import matplotlib.pyplot as plt
from collections import deque
import heapq

class Puzzle:
    def __init__(self, state=None):
        if state is None:
            self.state = 'solved_state'
        else:
            self.state = state

    def shuffle(self, moves):
        self.state = 'shuffled_state'  # Ejemplo simplificado

    def to_list(self):
        return list(self.state)

    def is_solved(self):
        return self.state == 'solved_state'

    def possible_moves(self):
        return ['U', 'U\'', 'R', 'R\'', 'F', 'F\'', 'D', 'D\'', 'L', 'L\'', 'B', 'B\'']

    def apply_move(self, move):
        return Puzzle(self.state + move)

    def heuristic(self):
        return sum(1 for c in self.state if c != 'solved_state')

def bfs_solve(puzzle, max_depth):
    start_time = time.time()
    process = psutil.Process()

    queue = deque([(puzzle.state, [])])
    visited = set([puzzle.state])

    nodes_expanded = 0
    times = []
    memories = []

    while queue:
        current_state, path = queue.popleft()
        nodes_expanded += 1

        current_memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB
        times.append(time.time() - start_time)
        memories.append(current_memory)

        if Puzzle(current_state).is_solved():
            return path, times, memories

        for move in puzzle.possible_moves():
            new_state = puzzle.apply_move(move).state
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, path + [move]))

    return None, times, memories

def astar_solve(puzzle):
    start_time = time.time()
    process = psutil.Process()

    heap = [(puzzle.heuristic(), 0, puzzle.state, [])]
    visited = set([puzzle.state])

    nodes_expanded = 0
    times = []
    memories = []

    while heap:
        _, cost, current_state, path = heapq.heappop(heap)
        nodes_expanded += 1

        current_memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB
        times.append(time.time() - start_time)
        memories.append(current_memory)

        if Puzzle(current_state).is_solved():
            return path, times, memories

        for move in puzzle.possible_moves():
            new_state = puzzle.apply_move(move).state
            if new_state not in visited:
                visited.add(new_state)
                heapq.heappush(heap, (cost + 1 + Puzzle(new_state).heuristic(), cost + 1, new_state, path + [move]))

    return None, times, memories

def iddfs_solve(puzzle, max_depth):
    start_time = time.time()
    process = psutil.Process()
    nodes_expanded = 0
    times = []
    memories = []

    def dls(current_state, path, depth):
        nonlocal nodes_expanded
        nodes_expanded += 1

        current_memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB
        times.append(time.time() - start_time)
        memories.append(current_memory)

        if depth == 0 and Puzzle(current_state).is_solved():
            return path
        if depth > 0:
            for move in puzzle.possible_moves():
                new_state = puzzle.apply_move(move).state
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

def plot_metrics(times_bfs, memories_bfs, times_astar, memories_astar, times_iddfs, memories_iddfs):
    plt.plot(range(len(memories_bfs)), memories_bfs, label='BFS')
    plt.plot(range(len(memories_astar)), memories_astar, label='A*')
    plt.plot(range(len(memories_iddfs)), memories_iddfs, label='IDDFS')

    plt.xlabel('Iteraciones')
    plt.ylabel('Memoria (MB)')
    plt.title('Comparación de Memoria Consumida por Algoritmos')
    plt.legend()
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

solution_bfs, times_bfs, memories_bfs = bfs_solve(desordenado, max_depth=10)
solution_astar, times_astar, memories_astar = astar_solve(desordenado)
solution_iddfs, times_iddfs, memories_iddfs = iddfs_solve(desordenado, max_depth=10)

plot_metrics(times_bfs, memories_bfs, times_astar, memories_astar, times_iddfs, memories_iddfs)

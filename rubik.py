import time
import psutil
import heapq
from collections import deque
import matplotlib.pyplot as plt

class Puzzle:
    def __init__(self):
        # Initialize the Puzzle in solved state
        self.state = 'solved_state'

    def is_solved(self):
        return self.state == 'solved_state'

    def possible_moves(self):
        # Returns a list of possible moves (this is an example)
        return ['U', 'U\'', 'R', 'R\'', 'F', 'F\'', 'D', 'D\'', 'L', 'L\'', 'B', 'B\'']

    def apply_move(self, move):
        # Applies a move and returns the new state (this is an example)
        return self.state + move

    def heuristic(self, other):
        # Example heuristic: number of mismatched pieces
        return sum(1 for a, b in zip(self.state, other.state) if a != b)

class HybridSolver:
    def __init__(self, puzzle):
        self.puzzle = puzzle
        self.max_depth = 10
        self.start_time = time.time()
        self.process = psutil.Process()
        self.times = []
        self.memories = []
        self.nodes_expanded = 0

    def record_metrics(self):
        current_time = time.time() - self.start_time
        current_memory = self.process.memory_info().rss / (1024 * 1024)  # Convert to MB
        self.times.append(current_time)
        self.memories.append(current_memory)

    def bfs_solve(self):
        queue = deque([(self.puzzle.state, [])])
        visited = set([self.puzzle.state])

        while queue:
            current_state, path = queue.popleft()
            self.nodes_expanded += 1
            self.record_metrics()

            if self.puzzle.is_solved():
                return path

            for move in self.puzzle.possible_moves():
                new_state = self.puzzle.apply_move(move)
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, path + [move]))

        return None

    def a_star_solve(self):
        open_set = []
        heapq.heappush(open_set, (0, self.puzzle.state, []))
        visited = set([self.puzzle.state])

        while open_set:
            _, current_state, path = heapq.heappop(open_set)
            self.nodes_expanded += 1
            self.record_metrics()

            if self.puzzle.is_solved():
                return path

            for move in self.puzzle.possible_moves():
                new_state = self.puzzle.apply_move(move)
                if new_state not in visited:
                    visited.add(new_state)
                    priority = len(path) + 1 + self.puzzle.heuristic(new_state)
                    heapq.heappush(open_set, (priority, new_state, path + [move]))

        return None

    def iddfs_solve(self):
        def dls(state, path, depth):
            if depth == 0 and self.puzzle.is_solved():
                return path
            if depth > 0:
                for move in self.puzzle.possible_moves():
                    new_state = self.puzzle.apply_move(move)
                    if new_state not in visited:
                        visited.add(new_state)
                        result = dls(new_state, path + [move], depth - 1)
                        if result is not None:
                            return result
            return None

        for depth in range(self.max_depth + 1):
            visited = set([self.puzzle.state])
            result = dls(self.puzzle.state, [], depth)
            self.nodes_expanded += 1
            self.record_metrics()
            if result is not None:
                return result

        return None

    def solve(self):
        bfs_result = self.bfs_solve()
        a_star_result = self.a_star_solve()
        iddfs_result = self.iddfs_solve()

        if bfs_result:
            return bfs_result
        elif a_star_result:
            return a_star_result
        elif iddfs_result:
            return iddfs_result
        else:
            return None

    def plot_metrics(self):
        fig, ax1 = plt.subplots()

        color = 'tab:blue'
        ax1.set_xlabel('Iteraciones')
        ax1.set_ylabel('Tiempo (s)', color=color)
        ax1.plot(range(len(self.times)), self.times, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Memoria (MB)', color=color)
        ax2.plot(range(len(self.memories)), self.memories, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        plt.title('Tiempo y Memoria Consumida por HybridSolver')
        plt.show()

# Ejemplo de uso
puzzle = Puzzle()
solver = HybridSolver(puzzle)
solution = solver.solve()

if solution:
    print(f'Solución encontrada: {solution}')
else:
    print('No se encontró solución dentro de la profundidad máxima')

solver.plot_metrics()

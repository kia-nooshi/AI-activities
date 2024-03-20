

from typing import List, Tuple, Set, Dict
import heapq 
from queue import PriorityQueue
from collections import deque
from itertools import count

class SearchAlgorithm:
    # Implement Uniform search
    @staticmethod
    def uniform_search(grid: List[List[str]]) -> List[Tuple[int, int]]:
        def explore_grid(grid, priority_queue, visited_nodes, predecessors):
            while priority_queue:
                cost, row, col = heapq.heappop(priority_queue)
                if grid[row][col] == 't':
                    return True, (row, col)
                if (row, col) not in visited_nodes:
                    visited_nodes.add((row, col))
                    adjacent_nodes = [(row - 1, col), (row, col - 1), (row + 1, col), (row, col + 1)]
                    for next_row, next_col in adjacent_nodes:
                        if 0 <= next_row < len(grid) and 0 <= next_col < len(grid[0]) and grid[next_row][next_col] != '-1':
                            if (next_row, next_col) not in visited_nodes:
                                predecessors[(next_row, next_col)] = (row, col)
                                heapq.heappush(priority_queue, (cost + 1, next_row, next_col))
            return False, None

        def reconstruct_path(predecessors, start, end):
            path = [end]
            while end != start:  # Change the loop to stop when start is reached
                end = predecessors.get(end)
                if end is None:  # Safety check to prevent infinite loops
                    return []
                path.append(end)
            path.reverse()
            return path

        start_position = next(((i, j) for i, row in enumerate(grid) for j, val in enumerate(row) if val == 's'), None)
        if not start_position:
            return []
        start_row, start_col = start_position
        priority_queue = [(0, start_row, start_col)]
        visited_nodes = set()
        predecessors = {}

        success, end_position = explore_grid(grid, priority_queue, visited_nodes, predecessors)
        if success:
            return reconstruct_path(predecessors, start_position, end_position)  # Adjusted to include start_position
        return []
        pass
    # Implement Depth First Search
    @staticmethod
    def dfs(grid: List[List[str]]) -> List[Tuple[int, int]]:
        def dfs_helper(grid, row, col, visited, predecessor):
            if grid[row][col] == 't':
                return True
            visited.add((row, col))
            
            directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]
            for dr, dc in directions:
                next_row, next_col = row + dr, col + dc
                if 0 <= next_row < len(grid) and 0 <= next_col < len(grid[0]) and \
                grid[next_row][next_col] != '-1' and (next_row, next_col) not in visited:
                    predecessor[(next_row, next_col)] = (row, col)
                    if dfs_helper(grid, next_row, next_col, visited, predecessor):
                        return True
            return False
        
        def bfs_shortest_path(predecessor, start, target):
            queue = deque([start])
            visited = set([start])
            while queue:
                current = queue.popleft()
                if current == target:
                    break
                for dx, dy in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                    next_row, next_col = current[0] + dx, current[1] + dy
                    if 0 <= next_row < len(grid) and 0 <= next_col < len(grid[0]) and \
                    (next_row, next_col) not in visited and (next_row, next_col) in predecessor:
                        visited.add((next_row, next_col))
                        predecessor[(next_row, next_col)] = current
                        queue.append((next_row, next_col))
        
        start_row, start_col = None, None
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == 's':
                    start_row, start_col = i, j
                    break
        
        visited = set()
        predecessor = {}
        found = dfs_helper(grid, start_row, start_col, visited, predecessor)
        
        if found:
            # Find the target position
            target = None
            for key, value in predecessor.items():
                if grid[key[0]][key[1]] == 't':
                    target = key
                    break
            
            # Run BFS on the path found by DFS to find the shortest path
            bfs_shortest_path(predecessor, (start_row, start_col), target)
            
            # Reconstruct the shortest path from predecessor map
            path = []
            current = target
            while current != (start_row, start_col):
                path.insert(0, current)
                current = predecessor.get(current)
            path.insert(0, (start_row, start_col))  # Insert start position at the beginning
            return path

        return []
        pass
    # Implement Breadth First Search
    @staticmethod
    def bfs(grid: List[List[str]]) -> List[Tuple[int, int]]:
        rows, cols = len(grid), len(grid[0])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        visited = [[False for _ in range(cols)] for _ in range(rows)]
        path = []
        start = end = None

        # Find start (s) and target (t) positions
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 's':
                    start = (r, c)
                elif grid[r][c] == 't':
                    end = (r, c)

        if not start or not end:
            return []

        # Initialize queue with start position and mark as visited
        queue = deque([(start, [start])])
        visited[start[0]][start[1]] = True

        while queue:
            (r, c), current_path = queue.popleft()
            if (r, c) == end:
                return current_path  # Found the target

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] != '-1':
                    visited[nr][nc] = True
                    queue.append(((nr, nc), current_path + [(nr, nc)]))

        return path  # Return empty if no path found

    # Your code here
        pass
    # Implement Best First Search
    @staticmethod
    def best_first_search(grid: List[List[str]]) -> List[Tuple[int, int]]:
        def get_neighbors(position):
            x, y = position
            for nx, ny in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
                if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] != '-1':
                    yield nx, ny

        def manhattan_distance(start, target):
            return abs(start[0] - target[0]) + abs(start[1] - target[1])

        # Find start (s) and target (t) positions
        start = target = None
        for i, row in enumerate(grid):
            for j, value in enumerate(row):
                if value == 's':
                    start = (i, j)
                elif value == 't':
                    target = (i, j)

        # Use a priority queue to store (distance, position) tuples
        frontier = [(manhattan_distance(start, target), start)]
        came_from = {start: None}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == target:
                break

            for neighbor in get_neighbors(current):
                if neighbor not in came_from:
                    priority = manhattan_distance(neighbor, target)
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current

        # Reconstruct path
        path = []
        if target and target in came_from:
            while current:
                path.append(current)
                current = came_from[current]
            path.reverse()

        return path        
        pass
    # Implement A* Search
    @staticmethod
    def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    @staticmethod
    def a_star_search(grid: List[List[str]]) -> List[Tuple[int, int]]:
        start = None
        target = None

        # Find start and target positions
        for i, row in enumerate(grid):
            for j, val in enumerate(row):
                if val == 's':
                    start = (i, j)
                elif val == 't':
                    target = (i, j)

        # Initialize open and closed sets
        open_set = [(0 + SearchAlgorithm.heuristic(start, target), 0, start)]
        came_from = {}
        g_score = {start: 0}
        closed_set = set()

        while open_set:
            _, g, current = heapq.heappop(open_set)

            if current == target:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            closed_set.add(current)

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)

                # Check if neighbor is within grid bounds
                if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]):
                    # Check if neighbor is traversable and not in closed set
                    if grid[neighbor[0]][neighbor[1]] != '-1' and neighbor not in closed_set:
                        tentative_g_score = g_score[current] + 1

                        if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                            came_from[neighbor] = current
                            g_score[neighbor] = tentative_g_score
                            f_score = tentative_g_score + SearchAlgorithm.heuristic(neighbor, target)
                            heapq.heappush(open_set, (f_score, tentative_g_score, neighbor))

        return []  # Return an empty path if no path is found
        pass
    # Implement Greedy Search
    @staticmethod
    def greedy_search(grid: List[List[str]]) -> List[Tuple[int, int]]:
        # Initialize variables
        rows, cols = len(grid), len(grid[0])  # number of rows and columns on the grid
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # order is right, down, left, up
        start_i, start_j = None, None  # start location
        target_i, target_j = None, None  # target location
        visited = set()  # keep track of visited nodes
        found = -1  # flag to indicate whether target is found
        visit = 0  # counter to mark visited nodes
        min_heuristic = float('inf') # start with positive infinite value
        

        # Find the starting and target positions
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 's':
                    start_i, start_j = i, j
                elif grid[i][j] == 't':
                    target_i, target_j = i, j
        
        pq = [(0, start_i, start_j, [(start_i, start_j)])] # priority queue to store nodes based on total cost and path
        heapq.heapify(pq) # heapify the priority queue

        # Manhattan distance heuristic function
        def heuristic(i, j):
            return abs(i - target_i) + abs(j - target_j)

        # main search loop, continue while the priority queue is not empty
        while pq:
            c, i, j, path = heapq.heappop(pq) # get the node with the lowest cost from priority queue

            # check if goal is reached
            if grid[i][j] == 't':
                found = 1
                return path # target found
            
            # goal not reached, next check if the node is already visited
            if (i, j) not in visited:
                visited.add((i, j))
                if grid[i][j] != 's':  # don't update 's'
                    visit += 1
                    grid[i][j] = str(visit) # mark other visited nodes

                next_move = None  # Initialize next_move before using it

                # check neighboring elements in all directions
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols and grid[ni][nj] != '-1' and (ni, nj) not in visited:
                        current_heuristic = heuristic(ni, nj) # value calculated in every direction
                        if current_heuristic < min_heuristic:
                            min_heuristic = current_heuristic # lowest heuristic value is stored
                            next_move = (min_heuristic, ni, nj, path + [(ni, nj)])  # Store the next move with the minimum heuristic value and updated path

                # Push the next move with the minimum heuristic value
                if next_move is not None:
                    heapq.heappush(pq, next_move)
                else:
                    continue  # Skip to the next iteration if next_move is still None

        return [] # target not found

        # Your code here
        pass

if __name__ == "__main__":

    example = [
    ['0', '0', '0', '0', '0', '0', '-1', '0', '0', '-1'],
    ['0', '0', 's', '0', '0', '0', '0', '0', '0', '0'],
    ['0', '0', '0', '0', '0', '0', '0', '0', '0', '-1'],
    ['0', '0', '0', '0', '-1', '0', '0', '0', '-1', '0'],
    ['0', '0', '0', '0', '0', '0', '-1', '0', '0', '0'],
    ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0'],
    ['0', '0', '0', '0', '-1', '0', '0', '0', '0', '0'],
    ['0', '0', '0', '0', '-1', '0', '0', '0', '0', '0'],
    ['0', '-1', '0', '0', '0', '0', '0', '0', '0', '0'],
    ['0', '-1', '0', '0', '0', '0', '0', '0', '-1', '0']
    ]
    print(SearchAlgorithm.uniform_search(example))
    print(SearchAlgorithm.dfs(example))
    print(SearchAlgorithm.bfs(example))
    print(SearchAlgorithm.best_first_search(example))
    print(SearchAlgorithm.a_star_search(example))
    print(SearchAlgorithm.greedy_search(example))





    
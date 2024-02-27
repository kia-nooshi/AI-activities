
from typing import List, Tuple
import heapq 
from queue import PriorityQueue
from collections import deque
from itertools import count

class SearchAlgorithm:

    # Implement Uniform search
    @staticmethod        
    def uniform_search(grid: List[List[str]]) -> Tuple[int, List[List[str]]]:
        # Your code here
        def explore_grid(grid, priority_queue, visited_nodes, path_counter):

            while priority_queue:
                # Pop the node with the smallest cost
                cost, row, col = heapq.heappop(priority_queue)

                # Return success if the target is found
                if grid[row][col] == 't':
                    return 1, grid

                # If the node is unvisited, mark it as visited and process
                if (row, col) not in visited_nodes:
                    visited_nodes.add((row, col))  # Mark node as visited

                    # Update grid to visualize path, skip if start node
                    if grid[row][col] != 's':
                        grid[row][col] = str(path_counter)
                        path_counter += 1

                    # Define potential moves from current position
                    adjacent_nodes = [
                        (row - 1, col),  # Move up
                        (row, col - 1),  # Move left
                        (row + 1, col),  # Move down
                        (row, col + 1)   # Move right
                    ]

                    # Enqueue valid adjacent nodes into the priority queue
                    for next_row, next_col in adjacent_nodes:
                        if 0 <= next_row < len(grid) and 0 <= next_col < len(grid[0]) and grid[next_row][next_col] != '-1':
                            heapq.heappush(priority_queue, (cost + 1, next_row, next_col))

            # Return failure if target is not found
            return -1, grid

        # Find the starting position (s) in the grid
        start_position = next(((i, j) for i, row in enumerate(grid) for j, val in enumerate(row) if val == 's'), (None, None))
        start_row, start_col = start_position

        # Initialize priority queue with the start position
        priority_queue = [(0, start_row, start_col)]
        heapq.heapify(priority_queue)
        visited_nodes = set()  # Set to track visited nodes
        path_counter = 1  # Counter to mark nodes during exploration

        # Process the grid and get the result
        result, final_grid = explore_grid(grid, priority_queue, visited_nodes, path_counter)
        return result, final_grid

        #pass

    # Implement Depth First Search
    @staticmethod
    def dfs(grid: List[List[str]]) -> Tuple[int, List[List[str]]]:
        def explore_adjacent(grid, current_row, current_col, path, search_target):
            directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]  # Define possible movement directions
            # Check each adjacent cell; if it meets criteria, explore it
            if any(
                0 <= current_row + dr < len(grid) and
                0 <= current_col + dc < len(grid[0]) and
                grid[current_row + dr][current_col + dc] != '-1' and  # Ensure cell is not blocked
                {(current_row + dr, current_col + dc)} not in path and  # Ensure cell is not already visited
                search_target(grid, current_row + dr, current_col + dc, path)  # Recursively search target
                for dr, dc in directions
            ):
                return True                
            return False

        # Update the grid with a sequence number based on the path taken
        def update_grid(path, grid):
            start_num = 1
            iteration_count = 0
            for step in path[1:]:  # Skip the first element (starting point)
                for row, col in step:
                    num = start_num + iteration_count
                    grid[row][col] = str(num)
                    iteration_count += 1

        # Search for the target, marking the path taken
        def search_target(grid, row, col, path):
            if grid[row][col] == 't':  # Target found
                return True
            path.append({(row, col)})  # Mark the current cell as visited
            
            return explore_adjacent(grid, row, col, path, search_target)
        
        try:
            # Locate the starting point 's' in the grid
            start_row = next(i for i, row in enumerate(grid) if 's' in row)
            start_col = grid[start_row].index('s')
        except StopIteration:
            start_row, start_col = None, None

        path = []  # Initialize the path taken as an empty list
        target_found = search_target(grid, start_row, start_col, path)
        update_grid(path, grid)
        result_status = 1 if target_found else -1  # Set result status based on target found or not

        return result_status, grid
            
        pass


            
        #pass
    
    # Implement Breadth First Search !!
    @staticmethod
    def bfs(grid: List[List[str]]) -> Tuple[int, List[List[str]]]:
               # Your code here
        def process_neighbors(row_num, col_num, grid, visited, queue):
            # Define the directions to check neighbors.
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            
            # Iterate through the directions to check neighboring cells.
            for dr, dc in directions:
                # Calculate the new row and column positions.
                new_row, new_col = row_num + dr, col_num + dc
                
                # Check if the new position is valid and hasn't been visited yet.
                valid_position = 0 <= new_row < len(grid) and 0 <= new_col < len(grid[0])
                not_visited = (new_row, new_col) not in visited
                
                # If the position is valid and not visited, and not a wall, add it to the queue.
                if valid_position and not_visited and grid[new_row][new_col] != '-1':
                    queue.append((new_row, new_col))
                    visited.add((new_row, new_col))
            

        # Find the starting position 's' in the grid.
        start_row, start_col = next(((i, j) for i, row in enumerate(grid) for j, val in enumerate(row) if val == 's'), (None, None))

        # If the starting position is found, initialize the queue and visited set.
        if start_row is not None and start_col is not None:
            queue = [(start_row, start_col)]
            visited = {(start_row, start_col)}

        count = 1 

        # Iterate through the grid while the queue is not empty.
        while len(queue) > 0:
            # Pop the first element from the queue.
            row, column = queue.pop(0)
            if grid[row][column] == 't':
                return 1, grid

            #update the grid with the count if not s.
            if grid[row][column] != 's':
                grid[row][column] = str(count)
            count = count + 1 if grid[row][column] != 's' else count
            print(f"{row} {column} {grid[row][column]}")
            
            # Process neighbors of the current cell.
            process_neighbors(row, column, grid, visited, queue)

        return -1, grid
        pass
    
    # Implement Best First Search 
    @staticmethod
    def best_first_search(grid: List[List[str]]) -> Tuple[int, List[List[str]]]:
        # Your code here
            # Your code here
        def get_valid_positions(grid, row, col, checked_nodes):
            # Define possible directions: up, left, down, right
            directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]
            # Generate valid positions based on directions and grid boundaries
            valid_positions = [(row + dr, col + dc) for dr, dc in directions 
                                if 0 <= row + dr < len(grid) and 0 <= col + dc < len(grid[0]) 
                                and grid[row + dr][col + dc] != '-1' 
                                and (row + dr, col + dc) not in checked_nodes]
            return valid_positions

        # Define search function using best-first search algorithm
        def search(grid, pq, checked_nodes, path_counter, end_row, end_column):
            while pq:
                # Extract node with lowest heuristic value
                item = heapq.heappop(pq)
                row, column = item[1], item[2]

                # Check if target node is reached
                if grid[row][column] == 't':
                    return 1, grid
                # Check if node is not visited
                elif (row, column) not in checked_nodes:
                    checked_nodes.add((row, column))
                    
                    # Mark node with path counter if not the start node
                    grid[row][column] = str(path_counter) if grid[row][column] != 's' else grid[row][column]
                    # Increment path counter if not the start node
                    path_counter = path_counter + 1 if grid[row][column] != 's' else path_counter
                    # Get valid positions around current node
                    valid_positions = get_valid_positions(grid, row, column, checked_nodes)

                    # Calculate heuristic for each valid position and add to priority queue
                    for new_row, new_column in valid_positions:
                        heuristic = abs(new_row - end_row) + abs(new_column - end_column)
                        heapq.heappush(pq, (heuristic, new_row, new_column))
            # Return failure if target node is not reachable
            return -1, grid
                                
        # Find starting and ending positions
        start_row, start_column = next((i, j) for i in range(len(grid)) for j in range(len(grid[i])) if grid[i][j] == 's')
        end_row, end_column = next((i, j) for i in range(len(grid)) for j in range(len(grid[i])) if grid[i][j] == 't')        
        # Initialize priority queue with starting position
        pq = [(0, start_row, start_column)]
        heapq.heapify(pq)
        checked_nodes = set()  # Set to track visited nodes
        path_counter = 1
        
        # Call search function to find path
        return search(grid, pq, checked_nodes, path_counter, end_row, end_column)
    
        pass
    
    # Implement A* Search
    @staticmethod
    def a_star_search(grid: List[List[str]]) -> Tuple[int, List[List[str]]]:
       
        def explore_neighbors(priority_queue, maze, cur_row, cur_col, path_cost, destination, visited):

            # Define possible movement directions (right, down, left, up)
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for delta_row, delta_col in directions:
                next_row, next_col = cur_row + delta_row, cur_col + delta_col
                
                # Check if next position is within bounds, not blocked, and not visited
                if 0 <= next_row < len(maze) and 0 <= next_col < len(maze[0]) and maze[next_row][next_col] != '-1' and (next_row, next_col) not in visited:
                    new_cost = path_cost + 1  # Increment cost for the next step
                    heuristic = abs(next_row - destination[0]) + abs(next_col - destination[1])  # Calculate Manhattan distance to destination
                    
                    # Add the next position to the priority queue with total cost (cost + heuristic)
                    heapq.heappush(priority_queue, (new_cost + heuristic, next_row, next_col, new_cost))
                            
        def update_position(row, column, visited, maze, step_counter):

            if (row, column) not in visited:
                visited.add((row, column))
                if maze[row][column] != 's':  # Check if not the start position
                    maze[row][column] = str(step_counter)
                    return step_counter + 1  # Increment step count for next use
            return step_counter

        # Find start ('s') and target ('t') positions in the maze
        start_pos = next(((i, j) for i, row in enumerate(grid) for j, val in enumerate(row) if val == 's'), None)
        end_pos = next(((i, j) for i, row in enumerate(grid) for j, val in enumerate(row) if val == 't'), None)

        # Exit early if start or end positions are missing
        if not start_pos or not end_pos:
            return -1, grid

        # Initialize priority queue with the start position, cost, and steps
        pq = [(0, start_pos[0], start_pos[1], 0)]
        heapq.heapify(pq)  # Ensure the list is in heap order
        visited_positions = set()  # Track visited positions
        step_count = 1

        while pq:
            total_cost, cur_row, cur_col, cost_to_reach = heapq.heappop(pq)
            
            # Check if current position is the target
            if (cur_row, cur_col) == end_pos:
                return 1, grid

            # Mark current position as visited and update steps
            step_count = update_position(cur_row, cur_col, visited_positions, grid, step_count)

            # Explore adjacent positions
            explore_neighbors(pq, grid, cur_row, cur_col, cost_to_reach, end_pos, visited_positions)

        return -1, grid
    
                
            #pass
                
    # Implement Greedy Search            
    @staticmethod
    def greedy_search(grid: List[List[str]]) -> Tuple[int, List[List[str]]]:    
        min_value = 2048 
        node_count = 1     
        def explore_neighbors(grid, pq, checked_nodes, node_count, min_value, end_row, end_col, min_row, min_col):
            if not pq:
                return -1, grid
            # Pop the node with the lowest heuristic value from the priority queue
            heuristic, row, col = heapq.heappop(pq)
            # Check if the current node is the target node 't'
            if grid[row][col] == 't':
                return 1, grid
            # If the current node has not been checked yet
            if (row, col) not in checked_nodes:
                checked_nodes.add((row, col))  # Mark checked
                # If the current is not 's', label it with the node count
                if grid[row][col] is not 's':
                    grid[row][col] = str(node_count)
                    node_count += 1

                # Define the directions to explore
                directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                
                # Filter valid neighbors based on grid boundaries and checked status
                valid_neighbors = filter(lambda x: 0 <= x[0] < len(grid) and 0 <= x[1] < len(grid[0]) and x not in checked_nodes,
                        [(row + delta_row, col + delta_col) for delta_row, delta_col in directions])

                # Check conditions for valid neighbors and update minimum distance if necessary & push
                for updated_row, updated_col in valid_neighbors:
                    distance_to_end = abs(updated_col - end_col) + abs(updated_row - end_row)
                    current_min_distance = abs(min_row - end_row) + abs(min_col - end_col)
                    is_path_clear = grid[updated_row][updated_col] != '-1'
                    if distance_to_end < min_value and is_path_clear and distance_to_end < current_min_distance:
                        min_value = distance_to_end
                        min_row, min_col = updated_row, updated_col

                heapq.heappush(pq, (distance_to_end, min_row, min_col))

            # Recursively explore neighbors
            return explore_neighbors(grid, pq, checked_nodes, node_count, min_value, end_row, end_col, min_row, min_col)
        
        # Find start and end positions in the grid
        start_position = next(((i, j) for i, row in enumerate(grid) for j, val in enumerate(row) if val == 's'), None)
        end_position = next(((i, j) for i, row in enumerate(grid) for j, val in enumerate(row) if val == 't'), None)
        
        if end_position:
            end_row, end_col = end_position

        pq = []  
        # If start position is found, search
        if start_position:
            start_row, start_col = start_position
            heapq.heappush(pq, (0, start_row, start_col))  # Push start node into the priority queue
            heapq.heapify(pq)
            min_row, min_col = start_row, start_col 
        checked_nodes = set()    

        return explore_neighbors(grid, pq, checked_nodes, node_count, min_value, end_row, end_col, min_row, min_col)
            # pass
            
        

if __name__ == "__main__":

    example = [
        ['0', '0', '0', '0'],
        ['0', '-1', '-1', 't'],
        ['s', '0', '-1', '0'],
        ['0', '0', '0', '-1']
    ]

    #found, final_state = SearchAlgorithm.dfs(example)
    #found, final_state = SearchAlgorithm.bfs(example)
    #found, final_state = SearchAlgorithm.best_first_search(example)
    found, final_state = SearchAlgorithm.greedy_search(example)
    #found, final_state = SearchAlgorithm.uniform_search(example)
    if found == 1:
        print("Target found!")
    else:
        print("Target not found.")

    for row in final_state:
        print(' '.join(row))
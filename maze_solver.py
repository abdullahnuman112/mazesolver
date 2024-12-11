import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from queue import PriorityQueue
from collections import deque
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation

# Function to generate a maze using Recursive Backtracking
def generate_maze(width, height):
    maze = np.ones((height, width), dtype=int)  # Initialize the maze with walls
    stack = [(1, 1)]  # Start at the top-left corner
    maze[1, 1] = 0  # Mark the start point as a path

    while stack:
        x, y = stack[-1]
        neighbors = []

        # Check possible neighbors two steps away
        for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            nx, ny = x + dx, y + dy
            if 0 < nx < height and 0 < ny < width and maze[nx, ny] == 1:
                neighbors.append((nx, ny))

        if neighbors:
            nx, ny = neighbors[np.random.randint(len(neighbors))]
            # Remove the wall between the current cell and the chosen neighbor
            maze[nx, ny] = 0
            maze[x + (nx - x) // 2, y + (ny - y) // 2] = 0
            stack.append((nx, ny))
        else:
            stack.pop()

    return maze

# Reconstruct the path from the came_from dictionary
def reconstruct_path(came_from, start, goal):
    if goal not in came_from:
        return []
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path
# Breadth-First Search (BFS)
def bfs(maze, start, goal):
    queue = deque([start])
    came_from = {start: None}
    explored = []
    while queue:
        current = queue.popleft()
        explored.append(current)
        if current == goal:
            break
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < maze.shape[0] and 0 <= neighbor[1] < maze.shape[1] and maze[neighbor] == 0:
                if neighbor not in came_from:
                    queue.append(neighbor)
                    came_from[neighbor] = current
    return reconstruct_path(came_from, start, goal), explored

# Dijkstra's Algorithm
def dijkstra(maze, start, goal):
    pq = PriorityQueue()
    pq.put((0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    explored = []
    while not pq.empty():
        current_cost, current = pq.get()
        explored.append(current)
        if current == goal:
            break
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < maze.shape[0] and 0 <= neighbor[1] < maze.shape[1] and maze[neighbor] == 0:
                new_cost = current_cost + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    pq.put((new_cost, neighbor))
                    came_from[neighbor] = current
    return reconstruct_path(came_from, start, goal), explored

# A* Algorithm
def a_star(maze, start, goal):
    pq = PriorityQueue()
    pq.put((0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    explored = []
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    while not pq.empty():
        _, current = pq.get()
        explored.append(current)
        if current == goal:
            break
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < maze.shape[0] and 0 <= neighbor[1] < maze.shape[1] and maze[neighbor] == 0:
                new_cost = cost_so_far[current] + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, goal)
                    pq.put((priority, neighbor))
                    came_from[neighbor] = current
    return reconstruct_path(came_from, start, goal), explored

# Greedy Best-First Search (GBFS)
def greedy_best_first_search(maze, start, goal):
    pq = PriorityQueue()
    pq.put((0, start))
    came_from = {start: None}
    explored = []
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    while not pq.empty():
        _, current = pq.get()
        explored.append(current)
        if current == goal:
            break
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < maze.shape[0] and 0 <= neighbor[1] < maze.shape[1] and maze[neighbor] == 0:
                if neighbor not in came_from:
                    priority = heuristic(neighbor, goal)
                    pq.put((priority, neighbor))
                    came_from[neighbor] = current
    return reconstruct_path(came_from, start, goal), explored

# Bidirectional Search
def bidirectional_search(maze, start, goal):
    start_queue, goal_queue = deque([start]), deque([goal])
    start_came_from, goal_came_from = {start: None}, {goal: None}
    explored = []
    while start_queue and goal_queue:
        # Forward search
        current = start_queue.popleft()
        explored.append(current)
        if current in goal_came_from:
            path = reconstruct_path(start_came_from, start, current) + reconstruct_path(goal_came_from, goal, current)[::-1]
            return path, explored
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < maze.shape[0] and 0 <= neighbor[1] < maze.shape[1] and maze[neighbor] == 0:
                if neighbor not in start_came_from:
                    start_queue.append(neighbor)
                    start_came_from[neighbor] = current
        # Backward search
        current = goal_queue.popleft()
        explored.append(current)
        if current in start_came_from:
            path = reconstruct_path(start_came_from, start, current) + reconstruct_path(goal_came_from, goal, current)[::-1]
            return path, explored
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < maze.shape[0] and 0 <= neighbor[1] < maze.shape[1] and maze[neighbor] == 0:
                if neighbor not in goal_came_from:
                    goal_queue.append(neighbor)
                    goal_came_from[neighbor] = current
    return [], explored

# Iterative Deepening A* (IDA*) Algorithm
def ida_star(maze, start, goal):
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def search(path, g, bound):
        current = path[-1]
        f = g + heuristic(current, goal)
        if f > bound:
            return f, None
        if current == goal:
            return f, path
        min_bound = float("inf")
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < maze.shape[0] and 0 <= neighbor[1] < maze.shape[1] and maze[neighbor] == 0:
                if neighbor not in path:
                    path.append(neighbor)
                    t, result = search(path, g + 1, bound)
                    if result is not None:
                        return t, result
                    path.pop()
                    min_bound = min(min_bound, t)
        return min_bound, None

    bound = heuristic(start, goal)
    path = [start]
    explored = set()
    while True:
        t, result = search(path, 0, bound)
        explored.update(path)
        if result is not None:
            return result, list(explored)
        if t == float("inf"):
            return [], list(explored)
        bound = t

# Jump Point Search (JPS) Algorithm
def jump_point_search(maze, start, goal):
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def jump(current, direction):
        x, y = current
        dx, dy = direction
        nx, ny = x + dx, y + dy
        if not (0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] == 0):
            return None
        if (nx, ny) == goal:
            return (nx, ny)

        # Diagonal jump
        if dx != 0 and dy != 0:
            if maze[nx - dx, ny] == 0 or maze[nx, ny - dy] == 0:
                return (nx, ny)
        # Horizontal or vertical jump
        elif dx != 0 and (maze[nx, ny - 1] == 0 or maze[nx, ny + 1] == 0):
            return (nx, ny)
        elif dy != 0 and (maze[nx - 1, ny] == 0 or maze[nx + 1, ny] == 0):
            return (nx, ny)

        return jump((nx, ny), direction)

    pq = PriorityQueue()
    pq.put((0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    explored = []

    while not pq.empty():
        _, current = pq.get()
        explored.append(current)
        if current == goal:
            break

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = jump(current, (dx, dy))
            if neighbor is not None:
                new_cost = cost_so_far[current] + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, goal)
                    pq.put((priority, neighbor))
                    came_from[neighbor] = current

    return reconstruct_path(came_from, start, goal), explored

# Visualization of Maze Animation
def visualize_maze_animation(maze, path, explored, title):
    fig, ax = plt.subplots(figsize=(8, 8))
    maze_copy = maze.copy()
    cmap = ListedColormap(["white", "black", "red", "green"])  # Red for explored, green for path

    def update(frame):
        if frame < len(explored):
            # Show explored nodes
            maze_copy[explored[frame]] = 2  # Mark explored nodes in red
            ax.set_title(f"{title}: Exploring Nodes", fontsize=14)
        elif frame - len(explored) < len(path):
            # Show solution path
            idx = frame - len(explored)
            maze_copy[path[idx]] = 3  # Mark solution path in green
            ax.set_title(f"{title}: Showing Solution Path", fontsize=14)
        
        ax.imshow(maze_copy, cmap=cmap)
        
        ax.axis("off")

    total_frames = len(explored) + len(path)
    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=50, repeat=False)
    plt.show()


# Analyze performance of all algorithms
def analyze_performance(algorithms, maze, start, goal):
    results = []
    for name, algorithm in algorithms:
        start_time = time.time()
        path, explored = algorithm(maze.copy(), start, goal)
        elapsed_time = time.time() - start_time
        results.append({
            "Algorithm": name,
            "Time (s)": elapsed_time,
            "Path Length": len(path),
            "Explored Nodes": len(explored),
        })
    return pd.DataFrame(results)

# Plot performance metrics
def plot_performance(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].bar(df["Algorithm"], df["Time (s)"], color="blue")
    axes[0].set_title("Execution Time")
    axes[0].set_ylabel("Time (seconds)")

    axes[1].bar(df["Algorithm"], df["Path Length"], color="green")
    axes[1].set_title("Path Length")
    axes[1].set_ylabel("Path Length (steps)")

    axes[2].bar(df["Algorithm"], df["Explored Nodes"], color="orange")
    axes[2].set_title("Explored Nodes")
    axes[2].set_ylabel("Number of Nodes")

    plt.tight_layout()
    plt.savefig("performance_metrics.png")
    plt.show()

# Main function with all features
def main():
    width = int(input("Enter the width of the maze: "))
    height = int(input("Enter the height of the maze: "))

    if width % 2 == 0:
        width += 1
    if height % 2 == 0:
        height += 1

    maze = generate_maze(width, height)
    start, goal = (1, 1), (height - 2, width - 2)

    algorithms = [
        ("Breadth-First Search", bfs),
        ("Dijkstra's Algorithm", dijkstra),
        ("A* Algorithm", a_star),
        ("Greedy Best-First Search", greedy_best_first_search),
        ("Bidirectional Search", bidirectional_search),
        ("Iterative Deepening A*", ida_star),
        ("Jump Point Search", jump_point_search),
    ]

    # Visualize paths for each algorithm
    for name, algorithm in algorithms:
        path, explored = algorithm(maze.copy(), start, goal)
        if path:
            visualize_maze_animation(maze, path, explored, f"{name} Solution")
        else:
            print(f"{name} could not find a path.")

       
    # Analyze performance
    results = analyze_performance(algorithms, maze, start, goal)
    print(results)

    # Plot comparison of performance metrics
    plot_performance(results)

if __name__ == "__main__":
    main()

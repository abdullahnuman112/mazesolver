# Maze Solver Project

## Overview

This project is a Python-based program to generate a maze using recursive backtracking and solve it using multiple pathfinding algorithms. The algorithms include:

- Breadth-First Search (BFS)
- Dijkstra's Algorithm
- A\* Algorithm
- Greedy Best-First Search (GBFS)
- Bidirectional Search
- Iterative Deepening A\* (IDA\*)
- Jump Point Search (JPS)

The program also provides animated visualizations of the exploration and solution paths and generates performance metrics comparing the algorithms.

---

## Running Environment

### Supported Environments

- Python 3.8 or higher
- Jupyter Notebook
- Local Python IDE (e.g., PyCharm, VS Code)
- Google Colab (with slight modifications for animation display)

### Operating Systems

- Windows
- macOS
- Linux

---

## Required Third-Party Libraries

The project requires the following Python libraries:

- `numpy`: Used for maze generation and numerical operations.
- `matplotlib`: For visualization and animations.
- `pandas`: For performance data handling.

### Installation

To install the required libraries, run the following command:

```bash
pip install numpy matplotlib pandas
```

---

## How to Run

1. **Clone or Download the Repository**:
   Download the source code or clone the repository.

2. **Run the Program**:
   Execute the program using a Python interpreter or Jupyter Notebook:

   ```bash
   python maze_solver.py
   ```

3. **Input Maze Dimensions**:
   Enter the width and height of the maze when prompted. Both dimensions must be odd; the program will adjust even inputs to the nearest odd number.

4. **Watch the Animations**:

   - The program will display step-by-step animations for each algorithm.
   - Explored nodes will be shown in red.
   - The solution path will be highlighted in green.

5. **Performance Metrics**:

   - A summary table and bar charts will be generated, comparing execution time, path length, and explored nodes for all algorithms.

---

## Project Structure

```plaintext
maze_solver_project/
├── maze_solver.py   # Main program file
├── performance_metrics.png  # Output performance metrics chart
└── README.md        # Project documentation
```

---

## Customization

### Animation Speed

Modify the `interval` parameter in the `visualize_maze_animation` function to adjust animation speed (default: 50 milliseconds):

```python
ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=50, repeat=False)
```

### Add or Remove Algorithms

To add or remove algorithms, edit the `algorithms` list in the `main` function:

```python
algorithms = [
    ("Breadth-First Search", bfs),
    ("Dijkstra's Algorithm", dijkstra),
    # Add or remove algorithms here
]
```

---

## Known Issues

- **Animation Compatibility**: On Google Colab, animations may not render properly. Consider saving animations as files and displaying them:

  ```python
  ani.save("animation.mp4")
  ```

- **Large Mazes**: Performance and rendering may slow down significantly for very large mazes.

---

## License

This project is licensed under the MIT License. Feel free to modify and distribute it.

---

## Contact

For questions or suggestions, please contact:

- **Email**: x2023dvf\@stfx.ca
- **GitHub**: [Your GitHub Profile](https://github.com/abdullahnuman112)


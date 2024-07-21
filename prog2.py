#2.A
# Contour Plot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dataset = pd.read_csv('./ToyotaCorolla.csv')
x = dataset['KM']
y = dataset['Weight']
z = dataset['Price']

plt.tricontourf(x, y, z, levels=20, cmap='jet')
plt.colorbar(label='Price')
plt.xlabel('KM')
plt.ylabel('Weight')
plt.title('Contour Plot')
plt.show()



#2.B
import heapq

def a_star_search_heapq(graph, start, goal, heuristic, cost):
    # Priority queue for exploring nodes
    priority_queue = []
    heapq.heappush(priority_queue, (0 + heuristic[start], start))
    visited = set()
    g_cost = {start: 0}
    parent = {start: None}

    while priority_queue:
        current_f_cost, current_node = heapq.heappop(priority_queue)

        if current_node in visited:
            continue

        visited.add(current_node)

        if current_node == goal:
            break

        for neighbor in graph[current_node]:
            new_cost = g_cost[current_node] + cost[(current_node, neighbor)]
            if neighbor not in g_cost or new_cost < g_cost[neighbor]:
                g_cost[neighbor] = new_cost
                f_cost = new_cost + heuristic[neighbor]
                heapq.heappush(priority_queue, (f_cost, neighbor))
                parent[neighbor] = current_node

    path = []
    node = goal
    total_cost = 0
    while node is not None:
        if parent[node] is not None:  
                total_cost += cost[(parent[node], node)]
        path.append(node)
        node = parent[node]
    path.reverse()

    return path, total_cost

# Given graph
graph = {
    'S': ['A', 'B'],
    'A': ['B', 'C', 'D'],
    'B': ['C'],
    'C': ['D'],
    'D': []
}

# Given heuristic values
heuristic = {
    'S': 7,
    'A': 6,
    'B': 2,
    'C': 1,
    'D': 0
}

# Given costs between nodes
cost = {
    ('S', 'A'): 1,
    ('S', 'B'): 4,
    ('A', 'B'): 2,
    ('A', 'C'): 5,
    ('A', 'D'): 12,
    ('B', 'C'): 2,
    ('C', 'D'): 3
}

start = 'S'
goal = 'D'

path, total_cost = a_star_search_heapq(graph, start, goal, heuristic, cost)
print("A* Search Path:", path)
print("Total Cost:", total_cost)

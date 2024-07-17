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

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0  
        self.h = 0  
        self.f = 0

    def __lt__(self, other):
        return self.f < other.f

def astar_search(graph, start, goal, heuristic):
    open_list = []
    closed_set = set()

    heapq.heappush(open_list, Node(start))

    while open_list:
        current_node = heapq.heappop(open_list)
        closed_set.add(current_node.position)

        if current_node.position == goal:
            path = []
            total_cost = current_node.g
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1], total_cost

        neighbors = graph.get(current_node.position, [])

        for neighbor, edge_cost in neighbors:
            if neighbor in closed_set:
                continue

            neighbor_node = Node(neighbor, current_node)
            neighbor_node.g = current_node.g + edge_cost  
            neighbor_node.h = heuristic[neighbor]
            neighbor_node.f = neighbor_node.g + neighbor_node.h

            heapq.heappush(open_list, neighbor_node)

    return [], float('inf') 

graph = {
    'A': [('B', 2), ('E', 3)],
    'B': [('C', 1), ('G', 9)],
    'C': [],
    'D': [('G', 1)],
    'E': [('D', 6)],
    'G': []
}

heuristic = {
    'A': 1,
    'B': 6,
    'C': 99,
    'D': 1,
    'E': 7,
    'G': 0
}
start_node = 'A'
goal_node = 'G'

path, total_cost = astar_search(graph, start_node, goal_node, heuristic)
if path:
    print("A* Search Path:", path)
    print("Total Cost:", total_cost)
else:
    print("No path found from", start_node, "to", goal_node)

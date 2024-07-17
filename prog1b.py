from queue import PriorityQueue

def best_first_search(graph, start, goal, heuristic):
    visited = set()
    pq = PriorityQueue()
    pq.put((0, start))
    parent = {start: None}
    path_cost = {start: 0} 
    
    while not pq.empty():
        current_cost, node = pq.get()
        
        if node == goal:
            break
        
        if node not in visited:
            visited.add(node)
            for neighbor in graph[node]:
                new_cost = path_cost[node] + 1  
                if neighbor not in path_cost or new_cost < path_cost[neighbor]:
                    path_cost[neighbor] = new_cost
                    total_cost = new_cost + heuristic[neighbor]
                    pq.put((total_cost, neighbor))
                    parent[neighbor] = node
    
 
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    
    return path, path_cost.get(goal, float('inf'))


graph = {
    'S': ['A', 'B'],
    'A': ['C', 'D'],
    'B': ['E', 'F'],
    'C': [],
    'D': [],
    'E': ['H'],
    'F': ['I', 'G'],
    'H': [],
    'I': [],
    'G': [],
}

start_node = 'S'
goal_node = 'E'

heuristic_values = {
    'S': 13,
    'A': 12,
    'B': 4,
    'C': 7,
    'D': 3,
    'E': 8,
    'F': 2,
    'H': 4,
    'I': 9,
    'G': 0,
}

path, total_cost = best_first_search(graph, start_node, goal_node, heuristic_values)
print("Best First Search Path:", path)
print("Total Cost:", total_cost)

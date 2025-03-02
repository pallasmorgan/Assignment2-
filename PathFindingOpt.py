import networkx as nx
import matplotlib.pyplot as plt
import random
import time
import numpy as np

# Define city positions
city_positions = {
    "Charleston": (0, 0), "North Charleston": (0, 5), "Summerville": (-5, 10), "Goose Creek": (5, 8),
    "Moncks Corner": (3, 15), "St. Stephen": (2, 22), "Bonneau": (7, 20), "Eutawville": (-3, 25),
    "Santee": (-2, 32), "Elloree": (-4, 36), "Orangeburg": (-6, 42), "St. Matthews": (-3, 50),
    "Cameron": (-8, 48), "Swansea": (-4, 58), "Gaston": (-2, 64), "Cayce": (1, 68), "Columbia": (2, 72),
    "West Columbia": (0, 72), "Blythewood": (3, 80), "Winnsboro": (1, 88), "Ridgeway": (-2, 92),
    "Great Falls": (-4, 100), "Fort Lawn": (-3, 110), "Lancaster": (-2, 118), "Richburg": (-1, 125),
    "Chester": (-5, 130), "Rock Hill": (-2, 140), "Fort Mill": (1, 148), "Pineville": (3, 152), "Charlotte": (5, 160)
}

# Define road distances
roads = [
    ("Charleston", "North Charleston", 10), ("North Charleston", "Summerville", 15),
    ("North Charleston", "Goose Creek", 10), ("Summerville", "Moncks Corner", 20),
    ("Goose Creek", "Moncks Corner", 15), ("Moncks Corner", "St. Stephen", 20),
    ("Moncks Corner", "Bonneau", 15), ("St. Stephen", "Bonneau", 10), ("St. Stephen", "Eutawville", 25),
    ("Bonneau", "Eutawville", 20), ("Eutawville", "Santee", 15), ("Santee", "Elloree", 10),
    ("Elloree", "Orangeburg", 20), ("Orangeburg", "St. Matthews", 15), ("St. Matthews", "Cameron", 10),
    ("Cameron", "Swansea", 20), ("Swansea", "Gaston", 10), ("Gaston", "Cayce", 15), ("Cayce", "Columbia", 5),
    ("Columbia", "West Columbia", 5), ("Columbia", "Blythewood", 20), ("Blythewood", "Winnsboro", 15),
    ("Winnsboro", "Ridgeway", 10), ("Ridgeway", "Great Falls", 15), ("Great Falls", "Fort Lawn", 10),
    ("Fort Lawn", "Lancaster", 15), ("Lancaster", "Richburg", 20), ("Richburg", "Chester", 15),
    ("Chester", "Rock Hill", 20), ("Rock Hill", "Fort Mill", 10), ("Fort Mill", "Pineville", 10),
    ("Pineville", "Charlotte", 10)
]

# Create the graph
G = nx.Graph()
G.add_weighted_edges_from(roads)

def compute_paths():
    start, end = "Charleston", "Charlotte"
    
    # Dijkstra's Algorithm
    start_time = time.time()
    shortest_path = nx.shortest_path(G, source=start, target=end, weight="weight")
    shortest_distance = nx.shortest_path_length(G, source=start, target=end, weight="weight")
    dijkstra_time = time.time() - start_time
    
    # Greedy Path with Backtracking
    def greedy_path_with_backtracking(graph, start, end):
        path, total_distance, visited = [start], 0, set([start])
        while path:
            current = path[-1]
            if current == end:
                return path, total_distance
            neighbors = [(n, graph[current][n]['weight']) for n in graph.neighbors(current) if n not in visited]
            if neighbors:
                next_city, distance = min(neighbors, key=lambda x: x[1])
                path.append(next_city)
                total_distance += distance
                visited.add(next_city)
            else:
                if len(path) > 1:
                    last_city = path.pop()
                    total_distance -= graph[path[-1]][last_city]['weight']
                else:
                    return [], float('inf')
        return [], float('inf')
    
    start_time = time.time()
    greedy_path_result, greedy_distance = greedy_path_with_backtracking(G, start, end)
    greedy_time = time.time() - start_time
    
    # Random Path
    def random_path(graph, start, end):
        while True:
            path, total_distance, current, visited = [start], 0, start, set([start])
            while current != end:
                neighbors = [n for n in graph.neighbors(current) if n not in visited]
                if not neighbors:
                    break
                next_city = random.choice(neighbors)
                path.append(next_city)
                total_distance += graph[current][next_city]['weight']
                visited.add(next_city)
                current = next_city
            if current == end:
                return path, total_distance
    
    start_time = time.time()
    random.seed(42)
    random_path_result, random_distance = random_path(G, start, end)
    random_time = time.time() - start_time
    
    # A* Search
    start_time = time.time()
    a_star_path_result = nx.astar_path(G, source=start, target=end, weight="weight")
    a_star_distance = sum(G[a_star_path_result[i]][a_star_path_result[i+1]]['weight'] for i in range(len(a_star_path_result) - 1))
    a_star_time = time.time() - start_time
    
    # Print Results
    print(f"Dijkstra's: {shortest_path} - {shortest_distance} miles - {dijkstra_time:.6f} sec")
    print(f"Greedy: {greedy_path_result} - {greedy_distance} miles - {greedy_time:.6f} sec")
    print(f"Random: {random_path_result} - {random_distance} miles - {random_time:.6f} sec")
    print(f"A*: {a_star_path_result} - {a_star_distance} miles - {a_star_time:.6f} sec")

# Draw the graph
plt.figure(figsize=(12, 8))
pos = {city: (x, y) for city, (x, y) in city_positions.items()}
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=800, font_size=8, edge_color='gray')
edge_labels = {(u, v): f"{d} mi" for u, v, d in roads}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
plt.title("Charleston to Charlotte - Road Network")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

compute_paths()



### **Nelder-Mead Operations**

# Define locations for Charleston (start) and Charlotte (end)
charleston = np.array([0, 0])
charlotte = np.array([100, 100])

# Generate initial simplex (Charleston + 2 random points within a 20-unit radius)
def initialize_simplex():
    np.random.seed(42)  # For reproducibility
    p1 = charleston
    p2 = charleston + np.random.uniform(-20, 20, 2)
    p3 = charleston + np.random.uniform(-20, 20, 2)
    return np.array([p1, p2, p3])

# Compute distance to Charlotte (objective function for minimization)
def objective_function(point):
    return np.linalg.norm(point - charlotte)

# Nelder-Mead transformations to optimize route
def nelder_mead_route(simplex, alpha=1, gamma=2, rho=0.5, sigma=0.5, max_iter=50):
    for i in range(max_iter):
        # Sort simplex points by their objective function values (distance to Charlotte)
        simplex = simplex[np.argsort([objective_function(p) for p in simplex])]
        best, second_best, worst = simplex
         # Compute centroid (excluding worst point)
        centroid = (best + second_best) / 2
         # Reflection step
        reflected = centroid + alpha * (centroid - worst)
        if objective_function(reflected) < objective_function(best):
            simplex[-1] = reflected
        else:
            # Expansion step
            expanded = centroid + gamma * (reflected - centroid)
            if objective_function(expanded) < objective_function(reflected):
                simplex[-1] = expanded
            else:
                # Contraction step
                contracted = centroid + rho * (worst - centroid)
                if objective_function(contracted) < objective_function(worst):
                    simplex[-1] = contracted
                else:
                    # Shrink simplex if all previous steps fail
                    simplex[1] = best + sigma * (simplex[1] - best)
                    simplex[2] = best + sigma * (simplex[2] - best)
        return simplex

### **New Simplex Selection**
# Run the optimization
simplex = initialize_simplex()
optimized_simplex = nelder_mead_route(simplex)

### **Visualization & Comparison**
# Plot results with labeled points and path
def plot_simplex(simplex):
    plt.figure(figsize=(6, 6))
    plt.scatter(*charleston, color='green', marker='o', label='Charleston (Start)')
    plt.scatter(*charlotte, color='red', marker='x', label='Charlotte (Goal)')
    plt.scatter(simplex[:, 0], simplex[:, 1], color='blue', marker='s', label='Simplex Points')
    plt.plot(simplex[:, 0], simplex[:, 1], 'b--', label='Optimization Path')
    plt.legend()
    plt.grid()
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Nelder-Mead Route Optimization')
    plt.show()
plot_simplex(optimized_simplex)
# Additional Analysis: Print final optimized points and distances
print("Final Simplex Points:")
for i, point in enumerate(optimized_simplex):
    print(f"Point {i + 1}: {point}, Distance to Charlotte: {objective_function(point):.2f}")

# Define road distances and city graph
graph = nx.Graph()
cities = ["Charleston", "North Charleston", "Summerville", "Goose Creek", "Moncks Corner", "St. Stephen", "Bonneau", "Eutawville", "Santee", "Elloree", "Orangeburg", "St. Matthews", "Cameron", "Swansea", "Gaston", "Cayce", "Columbia", "West Columbia", "Blythewood", "Winnsboro", "Ridgeway", "Great Falls", "Fort Lawn", "Lancaster", "Richburg", "Chester", "Rock Hill", "Fort Mill", "Pineville", "Charlotte"]
road_distances = [("Charleston", "North Charleston", 10), ("North Charleston", "Summerville", 15), ("North Charleston", "Goose Creek", 10), ("Summerville", "Moncks Corner", 20), ("Goose Creek", "Moncks Corner", 15), ("Moncks Corner", "St. Stephen", 20), ("Moncks Corner", "Bonneau", 15), ("St. Stephen", "Bonneau", 10), ("St. Stephen", "Eutawville", 25), ("Bonneau", "Eutawville", 20), ("Eutawville", "Santee", 15), ("Santee", "Elloree", 10), ("Elloree", "Orangeburg", 20), ("Orangeburg", "St. Matthews", 15), ("St. Matthews", "Cameron", 10), ("Cameron", "Swansea", 20), ("Swansea", "Gaston", 10), ("Gaston", "Cayce", 15), ("Cayce", "Columbia", 5), ("Columbia", "West Columbia", 5), ("Columbia", "Blythewood", 20), ("Blythewood", "Winnsboro", 15), ("Winnsboro", "Ridgeway", 10), ("Ridgeway", "Great Falls", 15), ("Great Falls", "Fort Lawn", 10), ("Fort Lawn", "Lancaster", 15), ("Lancaster", "Richburg", 20), ("Richburg", "Chester", 15), ("Chester", "Rock Hill", 20), ("Rock Hill", "Fort Mill", 10), ("Fort Mill", "Pineville", 10), ("Pineville", "Charlotte", 10)]
graph.add_weighted_edges_from(road_distances)

# Compute shortest path using Dijkstra's algorithm
shortest_path = nx.shortest_path(graph, source="Charleston", target="Charlotte", weight="weight")
shortest_distance = nx.shortest_path_length(graph, source="Charleston", target="Charlotte", weight="weight")
print(f"Dijkstra's Shortest Path: {shortest_path}, Distance: {shortest_distance} miles")


import matplotlib.pyplot as plt

# Updated city positions with better scaling
city_positions = {
    "Charleston": (0, 0), "North Charleston": (0, 5), "Summerville": (-2, 12), "Goose Creek": (2, 10),
    "Moncks Corner": (3, 18), "St. Stephen": (3, 26), "Bonneau": (5, 24), "Eutawville": (-3, 30),
    "Santee": (-4, 38), "Elloree": (-5, 40), "Orangeburg": (-7, 46), "St. Matthews": (-6, 52),
    "Cameron": (-8, 50), "Swansea": (-9, 60), "Gaston": (-10, 66), "Cayce": (-11, 70), "Columbia": (-12, 74),
    "West Columbia": (-12, 74), "Blythewood": (-13, 82), "Winnsboro": (-14, 90), "Ridgeway": (-14, 95),
    "Great Falls": (-15, 102), "Fort Lawn": (-15, 112), "Lancaster": (-16, 120), "Richburg": (-16, 128),
    "Chester": (-18, 134), "Rock Hill": (-18, 142), "Fort Mill": (-19, 150), "Pineville": (-20, 154), "Charlotte": (-21, 160)
}

# Extract x and y coordinates
x, y = zip(*city_positions.values())

# Create the plot
plt.figure(figsize=(6, 10))  # Adjust aspect ratio to match real-world proportions
plt.scatter(x, y, c='blue', marker='o')

# Annotate city names
for city, (x_pos, y_pos) in city_positions.items():
    plt.text(x_pos, y_pos, city, fontsize=8, ha='right')

plt.xlabel("Longitude (adjusted)")
plt.ylabel("Latitude (adjusted)")
plt.title("Adjusted City Locations in South Carolina")
plt.grid(True)
plt.show()

# EVOLVE-BLOCK-START
"""Best program for book embedding - score 997.23 (approximates 997.25)"""

from collections import defaultdict


def _build_degree(num_vertices, edges):
    deg = [0] * num_vertices
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return deg


def _crosses(e1, e2, pos):
    a, b = e1
    c, d = e2
    pa, pb = sorted((pos[a], pos[b]))
    pc, pd = sorted((pos[c], pos[d]))
    return (pa < pc < pb < pd) or (pc < pa < pd < pb)


def _assign_pages(edges, pos):
    """
    Optimal greedy page assignment for a fixed vertex order.
    Uses the interval‑right‑endpoint rule (optimal for permutation graphs).
    Returns a list `edge_pages` aligned with the original edge list.
    """
    intervals = []
    for idx, (u, v) in enumerate(edges):
        l, r = sorted((pos[u], pos[v]))
        intervals.append((l, r, idx))

    intervals.sort(key=lambda x: x[1])          # sort by right endpoint

    page_last_right = []                       # rightmost endpoint on each page
    edge_pages = [-1] * len(edges)

    for l, r, idx in intervals:
        placed = False
        for p, last_r in enumerate(page_last_right):
            if l > last_r or r < last_r:        # disjoint or nested → same page
                edge_pages[idx] = p
                page_last_right[p] = max(page_last_right[p], r)
                placed = True
                break
        if not placed:
            edge_pages[idx] = len(page_last_right)
            page_last_right.append(r)

    return edge_pages


def _assign_pages_greedy_coloring(edges, pos):
    """
    Improved greedy coloring using DSatur-like heuristic.
    Processes edges in order of saturation (number of different colors in neighborhood).
    """
    m = len(edges)
    if m == 0:
        return []
    
    conflicts = [[] for _ in range(m)]
    for i in range(m):
        ei = edges[i]
        for j in range(i + 1, m):
            if _crosses(ei, edges[j], pos):
                conflicts[i].append(j)
                conflicts[j].append(i)
    
    edge_pages = [-1] * m
    saturation = [0] * m
    color_usage = [set() for _ in range(m)]
    
    order = sorted(range(m), key=lambda i: (-len(conflicts[i]), i))
    
    for e_idx in order:
        used_colors = color_usage[e_idx]
        color = 0
        while color in used_colors:
            color += 1
        
        edge_pages[e_idx] = color
        
        for nb in conflicts[e_idx]:
            if color not in color_usage[nb]:
                color_usage[nb].add(color)
                saturation[nb] += 1
    
    return edge_pages

def _assign_pages_optimal_coloring(edges, pos):
    """
    Try multiple coloring strategies and return the best.
    """
    if len(edges) == 0:
        return []
    
    # Try interval-based assignment first (usually best)
    pages1 = _assign_pages(edges, pos)
    if len(edges) <= 100:  # Only try DSatur for smaller instances
        pages2 = _assign_pages_greedy_coloring(edges, pos)
        return pages1 if max(pages1) <= max(pages2) else pages2
    return pages1

def _simulated_annealing_order(edges, initial_order, iterations=2000):
    """
    Enhanced simulated annealing with multiple move types and better cooling.
    """
    import random
    import math
    
    n = len(initial_order)
    current_order = initial_order[:]
    pos = {v: i for i, v in enumerate(current_order)}
    current_pages = _assign_pages_optimal_coloring(edges, pos)
    current_max = max(current_pages) if current_pages else 0
    
    best_order = current_order[:]
    best_max = current_max
    temperature = 2.0
    cooling_rate = 0.9998
    
    # Track move statistics for adaptive search
    swap_improvements = 0
    insertion_improvements = 0
    
    for iteration in range(iterations):
        # Choose move type: 60% swap, 40% insertion
        move_type = 'swap' if random.random() < 0.6 else 'insertion'
        
        if move_type == 'swap':
            i, j = random.sample(range(n), 2)
            current_order[i], current_order[j] = current_order[j], current_order[i]
        else:  # insertion move
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i != j:
                vertex = current_order.pop(i)
                current_order.insert(j, vertex)
        
        pos_new = {v: idx for idx, v in enumerate(current_order)}
        new_pages = _assign_pages_optimal_coloring(edges, pos_new)
        new_max = max(new_pages) if new_pages else 0
        
        # Calculate acceptance probability
        delta = new_max - current_max
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_order = current_order[:]
                if move_type == 'swap':
                    swap_improvements += 1
                else:
                    insertion_improvements += 1
        else:
            # Revert move
            if move_type == 'swap':
                current_order[i], current_order[j] = current_order[j], current_order[i]
            else:
                if i != j:
                    vertex = current_order.pop(j)
                    current_order.insert(i, vertex)
        
        # Adaptive cooling: slow down when improvements are happening
        if iteration % 100 == 0 and iteration > 0:
            improvement_rate = (swap_improvements + insertion_improvements) / 100
            if improvement_rate > 0.1:
                cooling_rate = min(0.9999, cooling_rate * 1.001)
            else:
                cooling_rate = max(0.999, cooling_rate * 0.999)
            swap_improvement = 0
            insertion_improvements = 0       
        temperature *= cooling_rate
    
    return best_order, best_max


def _cuthill_mckee_order(adj, start):
    """Cuthill-McKee ordering starting from a given vertex."""
    n = len(adj)
    degree = [len(nei) for nei in adj]
    visited = [False] * n
    order = []
    queue = [start]
    visited[start] = True
    
    while queue:
        # Sort current queue by degree (ascending)
        queue.sort(key=lambda v: degree[v])
        next_queue = []
        for v in queue:
            order.append(v)
            # Add unvisited neighbors sorted by degree
            neighbors = [nb for nb in adj[v] if not visited[nb]]
            neighbors.sort(key=lambda v: degree[v])
            for nb in neighbors:
                if not visited[nb]:
                    visited[nb] = True
                    next_queue.append(nb)
        queue = next_queue
    
    # Add any remaining vertices (disconnected components)
    for v in range(n):
        if not visited[v]:
            order.append(v)
    
    return order

def solve_instance(graph):
    n = int(graph["num_vertices"])
    edges = [tuple(e) for e in graph["edges"]]

    if n == 0:
        return {"vertex_order": [], "edge_pages": []}
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    degree = [len(nei) for nei in adj]
    
    # Generate diverse initial orderings
    orderings = []
    
    # 1. Cuthill-McKee from multiple starting points
    start_candidates = sorted(range(n), key=lambda v: degree[v])[:min(5, n)]
    for start in start_candidates:
        orderings.append(_cuthill_mckee_order(adj, start))
    
    # 2. Reverse Cuthill-McKee (often reduces bandwidth)
    for start in start_candidates:
        cm_order = _cuthill_mckee_order(adj, start)
        orderings.append(cm_order[::-1])
    
    # 3. Degree-based ordering (ascending and descending)
    orderings.append(sorted(range(n), key=lambda v: degree[v]))
    orderings.append(sorted(range(n), key=lambda v: -degree[v]))
    
    # 4. Random permutations (for diversity)
    import random
    for _ in range(5):  # Increased from 3 to 5 for more diversity
        order = list(range(n))
        random.shuffle(order)
        orderings.append(order)
    
    # 5. DFS-based ordering (often good for reducing crossings)
    def dfs_order(start):
        visited = [False] * n
        order = []
        stack = [start]
        while stack:
            v = stack.pop()
            if not visited[v]:
                visited[v] = True
                order.append(v)
                # Add neighbors in reverse order to get consistent DFS
                neighbors = sorted(adj[v], reverse=True)
                for nb in neighbors:
                    if not visited[nb]:
                        stack.append(nb)
        # Add any remaining vertices
        for v in range(n):
            if not visited[v]:
                order.append(v)
        return order
    
    for start in start_candidates:
        orderings.append(dfs_order(start))
    
    # 6. BFS-based ordering
    def bfs_order(start):
        visited = [False] * n
        order = []
        queue = [start]
        visited[start] = True
        while queue:
            v = queue.pop(0)
            order.append(v)
            for nb in sorted(adj[v]):
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        # Add any remaining vertices
        for v in range(n):
            if not visited[v]:
                order.append(v)
        return order
    
    for start in start_candidates:
        orderings.append(bfs_order(start))
    
    # Evaluate all initial orderings
    best_order = None
    best_pages = float('inf')
    best_edge_pages = None
    
    for vertex_order in orderings:
        pos = {v: i for i, v in enumerate(vertex_order)}
        pages = _assign_pages_optimal_coloring(edges, pos)
        num_pages = max(pages) + 1 if pages else 0
        
        if num_pages < best_pages:
            best_pages = num_pages
            best_order = vertex_order[:]
            best_edge_pages = pages
    
    # Apply simulated annealing to the best ordering
    if n > 1 and len(edges) > 0:
        annealed_order, annealed_max = _simulated_annealing_order(edges, best_order, iterations=3000)
        if annealed_max + 1 < best_pages:
            best_order = annealed_order
            pos = {v: i for i, v in enumerate(best_order)}
            best_edge_pages = _assign_pages_optimal_coloring(edges, pos)
            best_pages = annealed_max + 1
    
    # Apply greedy local search to further reduce pages
    if n > 1 and len(edges) > 0:
        improved = True
        max_iterations = 100
        iteration = 0
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            # Try moving each vertex to all other positions
            for i in range(n):
                current_vertex = best_order[i]
                for j in range(n):
                    if i == j:
                        continue
                    # Create new order by moving vertex from i to j
                    temp_order = best_order[:]
                    temp_order.pop(i)
                    temp_order.insert(j, current_vertex)
                    
                    temp_pos = {v: idx for idx, v in enumerate(temp_order)}
                    temp_pages = _assign_pages_optimal_coloring(edges, temp_pos)
                    temp_max = max(temp_pages) + 1 if temp_pages else 0
                    
                    if temp_max < best_pages:
                        best_order = temp_order
                        best_edge_pages = temp_pages
                        best_pages = temp_max
                        improved = True
                        break  # Restart the search after improvement
                if improved:
                    break
    
    return {"vertex_order": best_order, "edge_pages": best_edge_pages}


# EVOLVE-BLOCK-END


if __name__ == "__main__":
    sample = {
        "num_vertices": 6,
        "edges": [[0, 3], [1, 4], [2, 5], [0, 5], [1, 2], [3, 4]],
    }
    out = solve_instance(sample)
    print(out)

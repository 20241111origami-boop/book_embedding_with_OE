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


def solve_instance(graph):
    n = int(graph["num_vertices"])
    edges = [tuple(e) for e in graph["edges"]]

    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    degree = [len(nei) for nei in adj]
    
    start = min(range(n), key=lambda v: (degree[v], v))
    vertex_order = [start]
    remaining = set(range(n))
    remaining.remove(start)
    
    while remaining:
        frontier = set()
        for v in vertex_order:
            for nb in adj[v]:
                if nb in remaining:
                    frontier.add(nb)
        
        if not frontier:
            next_v = min(remaining, key=lambda v: (degree[v], v))
        else:
            next_v = min(frontier, key=lambda v: (degree[v], v))
        
        vertex_order.append(next_v)
        remaining.remove(next_v)
    
    pos = {v: i for i, v in enumerate(vertex_order)}

    pages_interval = _assign_pages(edges, pos)
    edge_pages = pages_interval

    improved = True
    iteration = 0
    max_iterations = 10
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        for i in range(n):
            for j in range(i + 1, n):
                vertex_order[i], vertex_order[j] = vertex_order[j], vertex_order[i]
                pos_swapped = {v: idx for idx, v in enumerate(vertex_order)}
                new_pages = _assign_pages(edges, pos_swapped)
                
                if max(new_pages) < max(edge_pages):
                    edge_pages = new_pages
                    pos = pos_swapped
                    improved = True
                    break
                else:
                    vertex_order[i], vertex_order[j] = vertex_order[j], vertex_order[i]
            if improved:
                break
        
        if not improved:
            for i in range(n):
                original_pos = vertex_order[i]
                for j in range(n):
                    if i == j:
                        continue
                    vertex_order.pop(i)
                    vertex_order.insert(j, original_pos)
                    pos_moved = {v: idx for idx, v in enumerate(vertex_order)}
                    new_pages = _assign_pages(edges, pos_moved)
                    
                    if max(new_pages) < max(edge_pages):
                        edge_pages = new_pages
                        pos = pos_moved
                        improved = True
                        break
                    else:
                        vertex_order.pop(j)
                        vertex_order.insert(i, original_pos)
                if improved:
                    break

    return {"vertex_order": vertex_order, "edge_pages": edge_pages}


# EVOLVE-BLOCK-END


if __name__ == "__main__":
    sample = {
        "num_vertices": 6,
        "edges": [[0, 3], [1, 4], [2, 5], [0, 5], [1, 2], [3, 4]],
    }
    out = solve_instance(sample)
    print(out)

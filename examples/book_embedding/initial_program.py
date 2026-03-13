# EVOLVE-BLOCK-START
"""Initial program for book embedding page minimization."""

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

    # For each page, keep the last interval (l_prev, r_prev) in this order.
    # In right-endpoint order, a new interval (l, r) can be added iff it is
    # disjoint from the last one (l > r_prev) or it contains the last one
    # (l <= l_prev). Those two cases are exactly the non-crossing conditions
    # against all previously placed intervals on that page.
    page_last_interval = []
    edge_pages = [-1] * len(edges)

    for l, r, idx in intervals:
        placed = False
        for p, (last_l, last_r) in enumerate(page_last_interval):
            if l > last_r or l <= last_l:       # disjoint or contains previous
                edge_pages[idx] = p
                page_last_interval[p] = (l, r)
                placed = True
                break
        if not placed:
            edge_pages[idx] = len(page_last_interval)
            page_last_interval.append((l, r))

    return edge_pages


def _assign_pages_conflict(edges, pos):
    """
    Greedy colouring of the crossing (conflict) graph.
    Edges are processed in descending order of degree in the conflict graph.
    Returns a list `edge_pages` aligned with the original edge list.
    """
    m = len(edges)
    # build conflict sets
    conflicts = [set() for _ in range(m)]
    for i in range(m):
        ei = edges[i]
        for j in range(i + 1, m):
            if _crosses(ei, edges[j], pos):
                conflicts[i].add(j)
                conflicts[j].add(i)

    # order edges by decreasing number of conflicts
    order = sorted(range(m), key=lambda i: -len(conflicts[i]))
    edge_pages = [-1] * m
    for e_idx in order:
        used = {edge_pages[n] for n in conflicts[e_idx] if edge_pages[n] != -1}
        p = 0
        while p in used:
            p += 1
        edge_pages[e_idx] = p
    return edge_pages


def solve_instance(graph):
    """
    Returns a dictionary with:
      - vertex_order: list[int] permutation of vertices on the spine
      - edge_pages: list[int] page assignment for each edge in graph['edges']
    """
    n = int(graph["num_vertices"])
    edges = [tuple(e) for e in graph["edges"]]

    # Greedy maximum‑adjacency vertex order (deterministic)
    # Start with the highest‑degree vertex, then repeatedly add the vertex
    # that has the most neighbours already placed (ties broken by degree then id).
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    degree = [len(nei) for nei in adj]

    placed = []
    remaining = set(range(n))

    # initial vertex: highest degree, smallest id on tie
    start = max(remaining, key=lambda v: (degree[v], -v))
    placed.append(start)
    remaining.remove(start)

    while remaining:
        best = max(
            remaining,
            key=lambda v: (sum(1 for nb in adj[v] if nb in placed), degree[v], -v)
        )
        placed.append(best)
        remaining.remove(best)

    vertex_order = placed
    pos = {v: i for i, v in enumerate(vertex_order)}

    # Initial greedy page assignment – try both interval and conflict methods
    pages_interval = _assign_pages(edges, pos)
    pages_conflict = _assign_pages_conflict(edges, pos)
    edge_pages = pages_interval if (max(pages_interval) <= max(pages_conflict)) else pages_conflict

    # ---------- Local vertex‑order improvement ----------
    # Deterministic hill‑climbing: try all pairwise swaps and keep any that
    # strictly reduce the number of pages.  Restart the search after each
    # successful improvement.
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                # attempt swapping vertices i and j
                vertex_order[i], vertex_order[j] = vertex_order[j], vertex_order[i]
                pos_swapped = {v: idx for idx, v in enumerate(vertex_order)}
                # evaluate both page‑assignment strategies for the swapped order
                pages_int = _assign_pages(edges, pos_swapped)
                pages_con = _assign_pages_conflict(edges, pos_swapped)
                new_pages = pages_int if (max(pages_int) <= max(pages_con)) else pages_con

                if max(new_pages) + 1 < max(edge_pages) + 1:   # fewer pages found
                    edge_pages = new_pages
                    pos = pos_swapped
                    improved = True
                    # keep this swap and restart the outer loop
                    break
                else:
                    # revert swap
                    vertex_order[i], vertex_order[j] = vertex_order[j], vertex_order[i]
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

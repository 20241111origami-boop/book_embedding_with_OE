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


def solve_instance(graph):
    """
    Returns a dictionary with:
      - vertex_order: list[int] permutation of vertices on the spine
      - edge_pages: list[int] page assignment for each edge in graph['edges']
    """
    n = int(graph["num_vertices"])
    edges = [tuple(e) for e in graph["edges"]]

    # Simple baseline order: highest degree first, tie by vertex id.
    degree = _build_degree(n, edges)
    vertex_order = sorted(range(n), key=lambda x: (-degree[x], x))
    pos = {v: i for i, v in enumerate(vertex_order)}

    # Greedy page assignment: first non-crossing page.
    pages = defaultdict(list)  # page -> edge indices
    edge_pages = [-1] * len(edges)

    for i, e in enumerate(edges):
        assigned = False
        for page, idx_list in pages.items():
            if all(not _crosses(e, edges[j], pos) for j in idx_list):
                edge_pages[i] = page
                idx_list.append(i)
                assigned = True
                break
        if not assigned:
            new_page = len(pages)
            pages[new_page].append(i)
            edge_pages[i] = new_page

    return {"vertex_order": vertex_order, "edge_pages": edge_pages}


# EVOLVE-BLOCK-END


if __name__ == "__main__":
    sample = {
        "num_vertices": 6,
        "edges": [[0, 3], [1, 4], [2, 5], [0, 5], [1, 2], [3, 4]],
    }
    out = solve_instance(sample)
    print(out)

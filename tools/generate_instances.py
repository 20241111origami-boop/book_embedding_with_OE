#!/usr/bin/env python3
"""Generate a diverse instance suite for the book-embedding example."""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
INSTANCES_DIR = ROOT / "examples" / "book_embedding" / "instances"
MANIFEST_PATH = ROOT / "examples" / "book_embedding" / "instances_manifest.csv"


@dataclass
class GraphInstance:
    name: str
    n: int
    edges: list[tuple[int, int]]
    family: str
    known_optimum: int | None = None
    source: str = ""

    def to_json(self) -> dict:
        return {
            "num_vertices": self.n,
            "edges": [[u, v] for u, v in self.edges],
        }


def _canon(edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted({(min(u, v), max(u, v)) for u, v in edges if u != v})


def path_graph(n: int) -> list[tuple[int, int]]:
    return [(i, i + 1) for i in range(n - 1)]


def cycle_graph(n: int) -> list[tuple[int, int]]:
    return path_graph(n) + [(n - 1, 0)]


def star_graph(leaves: int) -> tuple[int, list[tuple[int, int]]]:
    n = leaves + 1
    return n, [(0, i) for i in range(1, n)]


def complete_graph(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def complete_bipartite_graph(a: int, b: int) -> tuple[int, list[tuple[int, int]]]:
    return a + b, [(i, a + j) for i in range(a) for j in range(b)]


def grid_graph(r: int, c: int) -> tuple[int, list[tuple[int, int]]]:
    def idx(x: int, y: int) -> int:
        return x * c + y

    edges = []
    for x in range(r):
        for y in range(c):
            if x + 1 < r:
                edges.append((idx(x, y), idx(x + 1, y)))
            if y + 1 < c:
                edges.append((idx(x, y), idx(x, y + 1)))
    return r * c, edges


def wheel_graph(n: int) -> list[tuple[int, int]]:
    # 0 is center; 1..n-1 form cycle.
    edges = [(0, i) for i in range(1, n)]
    for i in range(1, n):
        j = i + 1 if i + 1 < n else 1
        edges.append((i, j))
    return edges


def prism_graph(k: int) -> tuple[int, list[tuple[int, int]]]:
    # Two cycles of length k + matching edges.
    n = 2 * k
    edges = []
    for base in (0, k):
        for i in range(k):
            edges.append((base + i, base + ((i + 1) % k)))
    for i in range(k):
        edges.append((i, k + i))
    return n, edges


def perfect_binary_tree(depth: int) -> tuple[int, list[tuple[int, int]]]:
    n = 2 ** (depth + 1) - 1
    edges = []
    for i in range(2 ** depth - 1):
        edges.append((i, 2 * i + 1))
        edges.append((i, 2 * i + 2))
    return n, edges


def caterpillar_graph(spine_len: int, leaves_each: int) -> tuple[int, list[tuple[int, int]]]:
    edges = path_graph(spine_len)
    nxt = spine_len
    for v in range(spine_len):
        for _ in range(leaves_each):
            edges.append((v, nxt))
            nxt += 1
    return nxt, edges


def petersen_graph() -> tuple[int, list[tuple[int, int]]]:
    return 10, [
        (0, 1), (1, 2), (2, 3), (3, 4), (0, 4),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
        (5, 7), (7, 9), (6, 9), (6, 8), (5, 8),
    ]


def octahedron_graph() -> tuple[int, list[tuple[int, int]]]:
    # K_{2,2,2}
    groups = [(0, 1), (2, 3), (4, 5)]
    edges = []
    for i, gi in enumerate(groups):
        for j, gj in enumerate(groups):
            if i >= j:
                continue
            for u in gi:
                for v in gj:
                    edges.append((u, v))
    return 6, edges


def heawood_graph() -> tuple[int, list[tuple[int, int]]]:
    # Circulant graph C(14;1,5)
    n = 14
    edges = []
    for i in range(n):
        edges.append((i, (i + 1) % n))
        edges.append((i, (i + 5) % n))
    return n, _canon(edges)


def er_graph(n: int, m: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if m > len(all_edges):
        raise ValueError("m exceeds complete graph edge count")
    rng.shuffle(all_edges)
    return sorted(all_edges[:m])


def add(instances: list[GraphInstance], name: str, n: int, edges: list[tuple[int, int]], family: str,
        known_optimum: int | None = None, source: str = "") -> None:
    instances.append(GraphInstance(name=name, n=n, edges=_canon(edges), family=family,
                                   known_optimum=known_optimum, source=source))


def generate() -> list[GraphInstance]:
    instances: list[GraphInstance] = []

    # Step 1: Materialize from known_optima key families.
    for n in (5, 10, 20):
        add(instances, f"path_{n:02d}", n, path_graph(n), "path", 1, "known_optima")
    for n in (5, 6, 8, 10, 15):
        add(instances, f"cycle_{n:02d}", n, cycle_graph(n), "cycle", 1, "known_optima")
    for leaves in (8, 12):
        n, edges = star_graph(leaves)
        add(instances, f"star_{leaves}", n, edges, "star", 1, "known_optima")
    for depth in (3, 4):
        n, edges = perfect_binary_tree(depth)
        add(instances, f"binary_tree_d{depth}", n, edges, "tree", 1, "known_optima")
    n, edges = caterpillar_graph(5, 2)
    add(instances, "caterpillar_5_2", n, edges, "tree", 1, "known_optima")

    for n, opt in ((4, 3), (5, 3), (6, 3), (7, 3)):
        add(instances, f"K{n}", n, complete_graph(n), "complete", opt, "known_optima")
    for a, b, opt in ((2, 3, 2), (2, 5, 2), (3, 3, 3), (3, 4, 3), (3, 5, 4), (4, 4, 4)):
        n, edges = complete_bipartite_graph(a, b)
        add(instances, f"K{a}_{b}", n, edges, "bipartite", opt, "known_optima")

    n, edges = petersen_graph()
    add(instances, "petersen", n, edges, "special", 3, "known_optima")
    n, edges = octahedron_graph()
    add(instances, "octahedron", n, edges, "special", 3, "known_optima")
    n, edges = heawood_graph()
    add(instances, "heawood", n, edges, "special", 3, "known_optima")

    for r, c, opt in ((3, 3, 2), (4, 4, 2), (3, 5, 2), (4, 6, None), (5, 5, None)):
        n, edges = grid_graph(r, c)
        add(instances, f"grid_{r}x{c}", n, edges, "grid", opt, "known_optima")

    for n, opt in ((6, 3), (8, 3), (10, 3)):
        add(instances, f"wheel_{n}", n, wheel_graph(n), "wheel", opt, "known_optima")
    for k, opt in ((4, 2), (5, 2), (6, 2)):
        n, edges = prism_graph(k)
        add(instances, f"prism_{k}", n, edges, "prism", opt, "known_optima")

    known_random_specs = [
        ("random_12_20", 12, 20),
        ("random_15_28", 15, 28),
        ("random_15_35", 15, 35),
        ("random_20_40", 20, 40),
        ("random_20_60", 20, 60),
        ("random_25_50", 25, 50),
        ("random_30_60", 30, 60),
    ]
    for i, (name, n, m) in enumerate(known_random_specs):
        add(instances, name, n, er_graph(n, m, seed=100 + i), "random", None, "known_optima")

    # Step 2: Add 30 fixed-seed random graphs.
    extra_random_specs = [
        (14, 24), (14, 32), (16, 24), (16, 40), (18, 30),
        (18, 45), (20, 30), (20, 50), (22, 36), (22, 60),
        (24, 36), (24, 72), (26, 42), (26, 84), (28, 50),
        (28, 90), (30, 45), (30, 75), (32, 50), (32, 100),
        (34, 55), (34, 110), (36, 60), (36, 120), (40, 70),
        (40, 130), (45, 80), (45, 150), (50, 90), (50, 180),
    ]
    for i, (n, m) in enumerate(extra_random_specs):
        add(instances, f"random_seeded_{n}_{m}_s{i}", n, er_graph(n, m, seed=1000 + i), "random", None, "step2")

    # Step 3: Balance counts to near 100 by adding structured families.
    for n in (30, 40):
        add(instances, f"path_{n:02d}", n, path_graph(n), "path", 1, "step3")
    for n in (20, 25):
        add(instances, f"cycle_{n:02d}", n, cycle_graph(n), "cycle", 1, "step3")
    for leaves in (16, 20):
        n, edges = star_graph(leaves)
        add(instances, f"star_{leaves}", n, edges, "star", 1, "step3")
    n, edges = perfect_binary_tree(5)
    add(instances, "binary_tree_d5", n, edges, "tree", 1, "step3")
    n, edges = caterpillar_graph(8, 3)
    add(instances, "caterpillar_8_3", n, edges, "tree", 1, "step3")

    for n in (8, 9):
        add(instances, f"K{n}", n, complete_graph(n), "complete", None, "step3")
    for a, b in ((5, 5), (4, 6)):
        n, edges = complete_bipartite_graph(a, b)
        add(instances, f"K{a}_{b}", n, edges, "bipartite", None, "step3")

    for r, c in ((6, 6), (5, 7)):
        n, edges = grid_graph(r, c)
        add(instances, f"grid_{r}x{c}", n, edges, "grid", None, "step3")
    for n in (12, 14):
        add(instances, f"wheel_{n}", n, wheel_graph(n), "wheel", None, "step3")
    for k in (7, 8):
        n, edges = prism_graph(k)
        add(instances, f"prism_{k}", n, edges, "prism", None, "step3")

    # Keep legacy small examples for continuity.
    add(instances, "small_01", 6, [(0, 3), (1, 4), (2, 5), (0, 5), (1, 2), (3, 4)], "small", None, "legacy")
    add(instances, "small_02", 8, [(0, 4), (1, 5), (2, 6), (3, 7), (0, 7), (1, 6), (2, 5), (3, 4), (0, 2), (5, 7)], "small", None, "legacy")
    add(instances, "small_03", 10, [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (0, 9), (1, 8), (2, 5), (3, 6), (4, 7), (0, 3), (6, 9)], "small", None, "legacy")

    names = [inst.name for inst in instances]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate instance names were generated")

    return instances


def write_instances(instances: list[GraphInstance]) -> None:
    INSTANCES_DIR.mkdir(parents=True, exist_ok=True)
    for old in INSTANCES_DIR.glob("*.json"):
        old.unlink()

    for inst in instances:
        path = INSTANCES_DIR / f"{inst.name}.json"
        path.write_text(json.dumps(inst.to_json(), indent=2) + "\n")

    with MANIFEST_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "family", "num_vertices", "num_edges", "density", "known_optimum", "source"])
        for inst in sorted(instances, key=lambda x: x.name):
            m = len(inst.edges)
            density = (2.0 * m) / (inst.n * (inst.n - 1)) if inst.n > 1 else 0.0
            writer.writerow([
                inst.name,
                inst.family,
                inst.n,
                m,
                f"{density:.4f}",
                "" if inst.known_optimum is None else inst.known_optimum,
                inst.source,
            ])


def print_summary(instances: list[GraphInstance]) -> None:
    by_family: dict[str, int] = {}
    for inst in instances:
        by_family[inst.family] = by_family.get(inst.family, 0) + 1

    print(f"Generated {len(instances)} instances")
    for family in sorted(by_family):
        print(f"  {family:10s}: {by_family[family]}")


if __name__ == "__main__":
    data = generate()
    write_instances(data)
    print_summary(data)

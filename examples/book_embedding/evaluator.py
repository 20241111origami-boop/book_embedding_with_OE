"""Evaluator for book embedding page minimization."""

import importlib.util
import json
from pathlib import Path

from openevolve.evaluation_result import EvaluationResult


ROOT = Path(__file__).resolve().parent
INSTANCES_DIR = ROOT / "instances"


def _crosses(e1, e2, pos):
    a, b = e1
    c, d = e2
    pa, pb = sorted((pos[a], pos[b]))
    pc, pd = sorted((pos[c], pos[d]))
    return (pa < pc < pb < pd) or (pc < pa < pd < pb)


def _validate_solution(graph, sol):
    n = int(graph["num_vertices"])
    edges = [tuple(e) for e in graph["edges"]]

    if not isinstance(sol, dict):
        return {"valid": False, "reason": "solution is not a dict"}

    order = sol.get("vertex_order")
    edge_pages = sol.get("edge_pages")

    if not isinstance(order, list) or sorted(order) != list(range(n)):
        return {"valid": False, "reason": "vertex_order is not a valid permutation"}

    if not isinstance(edge_pages, list) or len(edge_pages) != len(edges):
        return {"valid": False, "reason": "edge_pages has invalid length"}

    if any((not isinstance(p, int)) or p < 0 for p in edge_pages):
        return {"valid": False, "reason": "edge_pages must be non-negative integers"}

    pos = {v: i for i, v in enumerate(order)}
    max_page = max(edge_pages) if edge_pages else -1

    violations = 0
    for page in range(max_page + 1):
        page_edge_ids = [i for i, p in enumerate(edge_pages) if p == page]
        for i in range(len(page_edge_ids)):
            e1 = edges[page_edge_ids[i]]
            for j in range(i + 1, len(page_edge_ids)):
                e2 = edges[page_edge_ids[j]]
                if _crosses(e1, e2, pos):
                    violations += 1

    return {
        "valid": True,
        "violations": violations,
        "num_pages": len(set(edge_pages)) if edge_pages else 0,
    }


def _score(violations, num_pages):
    # Maximize score. Feasible (zero violations) dominates infeasible solutions.
    if violations == 0:
        return 1000.0 - float(num_pages)
    return 1.0 / (1.0 + float(violations) + float(num_pages))


def evaluate(program_path):
    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        if not hasattr(program, "solve_instance"):
            return EvaluationResult(
                metrics={
                    "combined_score": 0.0,
                    "avg_pages": 9999.0,
                    "avg_violations": 9999.0,
                },
                artifacts={"error": "Missing solve_instance(graph)"},
            )

        instance_paths = sorted(INSTANCES_DIR.glob("*.json"))
        if not instance_paths:
            return EvaluationResult(
                metrics={"combined_score": 0.0, "avg_pages": 9999.0, "avg_violations": 9999.0},
                artifacts={"error": "No instances found"},
            )

        total_score = 0.0
        total_pages = 0.0
        total_violations = 0.0
        feasible_count = 0
        errors = []

        for p in instance_paths:
            graph = json.loads(p.read_text())
            sol = program.solve_instance(graph)
            check = _validate_solution(graph, sol)

            if not check["valid"]:
                errors.append({"instance": p.name, "reason": check["reason"]})
                total_score += 0.0
                total_pages += 9999.0
                total_violations += 9999.0
                continue

            violations = check["violations"]
            num_pages = check["num_pages"]

            total_score += _score(violations, num_pages)
            total_pages += float(num_pages)
            total_violations += float(violations)
            if violations == 0:
                feasible_count += 1

        count = float(len(instance_paths))
        metrics = {
            "combined_score": total_score / count,
            "avg_pages": total_pages / count,
            "avg_violations": total_violations / count,
            "feasible_rate": float(feasible_count) / count,
        }
        artifacts = {"invalid_outputs": errors}
        return EvaluationResult(metrics=metrics, artifacts=artifacts)

    except Exception as exc:
        return EvaluationResult(
            metrics={
                "combined_score": 0.0,
                "avg_pages": 9999.0,
                "avg_violations": 9999.0,
            },
            artifacts={"error": str(exc)},
        )

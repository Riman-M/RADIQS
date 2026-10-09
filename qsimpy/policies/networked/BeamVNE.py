"""
BeamVNE -- connectivity-first beam placement for distributed quantum workflows.

Drop in at qsimpy/policies/networked/BeamVNE.py

The allocation problem is virtual network embedding: the workflow is the request
graph, the QPU network is the substrate, qubit width is node demand, and a
dependency edge must be carried by a substrate link. Rather than enumerating
subgraph isomorphisms and testing feasibility afterwards (SoftIso), we construct
only mappings that already satisfy the connectivity constraint, by restricting
each task's candidate QPUs to the substrate neighbourhood of the QPUs already
assigned to its mapped neighbours.

Cost per workflow is O(k * B * d) with k tasks, beam width B and mean substrate
degree d, with no dependence on |V_S| beyond the first (anchor) task.

Two knobs:
  BEAM_WIDTH    -- partial mappings retained per expansion step.
  QUEUE_WEIGHT  -- weight on per-node queue delay in the beam proxy. Set to 0.0
                   to reproduce the baselines' objective exactly; raise it to
                   trade completion against wait time. Final selection among
                   completed mappings always uses assignment_cost, so the
                   comparison with the baselines stays like-for-like.
"""

import networkx as nx
from qsimpy.utils.Checks import check_connectivity, qubit_condition
from qsimpy.utils.Metrics import assignment_cost, runtime, error, linkCost, getObjectFromList
import os
BEAM_WIDTH = int(os.environ.get("BEAM_WIDTH", 8))
QUEUE_WEIGHT = 0.0

# BEAM_WIDTH = 8
# QUEUE_WEIGHT = 0.0


def _undirected_neighbours(task_graph, t):
    """Dependency direction does not affect which QPUs must be linked."""
    return set(nx.all_neighbors(task_graph, t))


def _step_cost(node_graph, task_graph, tasks, nodes, partial, t, n):
    """Incremental proxy for placing task t on node n, given partial mapping.

    assignment_cost cannot score a partial mapping (it dereferences every
    neighbour), so we score only the terms that are already determined: the
    node-local runtime and error for this placement, the link cost to
    neighbours already mapped, and optionally this node's queue delay.
    """
    task = getObjectFromList(tasks, t)
    node = getObjectFromList(nodes, n)

    cost = runtime(node, task) + error(node, task)

    for x in _undirected_neighbours(task_graph, t):
        if x in partial:
            cost += linkCost(node, task, getObjectFromList(nodes, partial[x]),
                             getObjectFromList(tasks, x))

    if QUEUE_WEIGHT:
        nat = node_graph.nodes[n].get("next_available_time", 0) or 0
        cost += QUEUE_WEIGHT * nat

    return cost


def _candidates(node_graph, task_graph, t, partial, demand):
    """QPUs that could host task t without violating connectivity or qubits.

    If t has mapped neighbours, candidates are restricted to the intersection of
    those neighbours' substrate neighbourhoods -- connectivity then holds by
    construction. Otherwise every sufficiently large QPU is a candidate.
    """
    used = set(partial.values())
    mapped_nb = [partial[x] for x in _undirected_neighbours(task_graph, t) if x in partial]

    if mapped_nb:
        allowed = None
        for m in mapped_nb:
            nb = set(node_graph.neighbors(m))
            allowed = nb if allowed is None else (allowed & nb)
        if not allowed:
            return []
    else:
        allowed = set(node_graph.nodes)

    return [n for n in allowed
            if n not in used
            and node_graph.nodes[n].get("qubit_number", 0) >= demand]


def beam_map(node_graph, task_graph, tasks, nodes, beam_width=BEAM_WIDTH):
    """Return a list of complete task->node mappings, best-effort."""
    demands = nx.get_node_attributes(task_graph, "qubit_number")

    if not demands:
        return []
    
    # Hardest task first: an infeasible workflow fails on the first expansion.
    order = sorted(demands, key=lambda t: demands[t], reverse=True)

    # Re-order so that, after the anchor, each task has a mapped neighbour
    # where one exists -- this is what keeps the candidate sets small.
    ordered, remaining = [order[0]], set(order[1:])
    while remaining:
        nxt = next((t for t in order if t in remaining
                    and _undirected_neighbours(task_graph, t) & set(ordered)), None)
        if nxt is None:
            nxt = next(t for t in order if t in remaining)
        ordered.append(nxt)
        remaining.discard(nxt)

    beam = [({}, 0.0)]
    for t in ordered:
        nxt_beam = []
        for partial, score in beam:
            for n in _candidates(node_graph, task_graph, t, partial, demands[t]):
                ext = dict(partial)
                ext[t] = n
                nxt_beam.append((ext, score + _step_cost(node_graph, task_graph,
                                                         tasks, nodes, partial, t, n)))
        if not nxt_beam:
            return []
        nxt_beam.sort(key=lambda x: x[1])
        beam = nxt_beam[:beam_width]

    return [m for m, _ in beam if len(m) == len(demands)]


def policy_beam_vne(env, wt, skipped_list):
    tasks, nodes = env.current_obs

    node_graph = env.node_graph.copy()
    nx.set_node_attributes(node_graph, {q.id: q.generate_att_dict() for q in nodes})

    # Identical construction to the baselines, so the RNG is consumed the same
    # way and the workflow topologies are comparable across policies.
    task_graph = nx.fast_gnp_random_graph(len(tasks), p=0.3, seed=env.rng, directed=True)
    lb = sorted(task_graph)
    task_graph = nx.relabel_nodes(task_graph, {lb[i]: tasks[i].id for i in range(len(lb))})
    nx.set_node_attributes(task_graph, {t.id: t.generate_att_dict() for t in tasks})

    candidates = beam_map(node_graph, task_graph, tasks, nodes)

    # Connectivity and qubit checks should hold by construction; we verify
    # rather than assume, so that a defect shows up as a rejected mapping
    # instead of an invalid allocation.
    candidates = [m for m in candidates
                  if check_connectivity(task_graph, m, node_graph)
                  and qubit_condition(node_graph.subgraph(m.values()), task_graph,
                                      {v: k for k, v in m.items()})]

    if not candidates:
        return None, task_graph

    best = min(candidates,
               key=lambda m: assignment_cost(node_graph.subgraph(m.values()),
                                             task_graph, tasks, nodes, m, weights=wt))

    return list(dict(sorted(best.items(), key=lambda x: x[0])).values()), task_graph

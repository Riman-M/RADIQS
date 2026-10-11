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

Cost is O(|V_S| + k^2 * B * d) with k tasks, beam width B and mean substrate
degree d: the dependence on substrate size is confined to the first (anchor)
task, and the k^2 comes from the link term, which can touch up to k-1 mapped
neighbours when scoring one placement.

Knobs (all settable from the environment, so a sweep needs no edit):
  BEAM_WIDTH    -- partial mappings retained per expansion step.
  QUEUE_WEIGHT  -- weight on normalised per-node queue delay. 0.0 reproduces the
                   baselines' objective exactly. The term is divided by the
                   largest queue on the substrate, so it lies in [0, 1] and this
                   weight is directly comparable to the others.
  DIRECTED      -- task-graph construction. 0 (default) matches SoftIso; 1
                   matches CloudQC, GreedyDfs and Random. These consume the
                   shared RNG at different rates, so this flag decides which
                   policies this one is paired with episode-for-episode.
  NAT_PROBE     -- path to a CSV. When set, logs the baselines' availability
                   term for the selected mapping, once per allocation. See the
                   note below.

On the availability term: assignment_cost weights Metrics.nexAvailable at
weights[0][0] (0.5 as published), but that function returns
SIG(max next_available_time) over the MAPPED nodes, and SIG saturates to exactly
1.0 in float64 once its argument exceeds about 37. From the second task set
onward, any QPU that has executed anything has a clock above 60, so the term is
pinned at its ceiling and cannot discriminate between candidate mappings.

It is not constant for a whole run, though: setup_quantum_resources() builds
fresh QNode objects each episode, so every episode starts with all clocks at
their initial -1, where SIG(-1) = 0.269. The opening allocations of an episode
can therefore draw entirely on untouched QPUs. Set NAT_PROBE to measure what
fraction of real decisions are saturated; analyse with analyse_natprobe.py.

The beam proxy omits the availability term, which is exact wherever the term is
saturated, and QUEUE_WEIGHT reintroduces queue delay in a form that does
discriminate.
"""

import os
import networkx as nx
from qsimpy.utils.Checks import check_connectivity, qubit_condition
from qsimpy.utils.Metrics import (SIG, assignment_cost, runtime, error, linkCost,
                                  getObjectFromList, nexAvailable)

BEAM_WIDTH = int(os.environ.get("BEAM_WIDTH", 8))
QUEUE_WEIGHT = float(os.environ.get("QUEUE_WEIGHT", 0.0))
DIRECTED = os.environ.get("DIRECTED", "0") == "1"
NAT_PROBE = os.environ.get("NAT_PROBE")

EDGE_PROB = 0.3          # task-graph density, as in the baselines
DEFAULT_WEIGHTS = (1 / 3, 1 / 3, 1 / 3)
_probe_started = False


def _term_weights(wt):
    """(runtime, error, communication) weights, as assignment_cost applies them.

    The incremental score must weight its terms the same way the final objective
    does, or the beam prunes on a different ranking than the one used to select.
    The published weights are equal, which is why an unweighted sum happened to
    agree; this keeps them in step if the weights are ever changed.
    """
    try:
        w = tuple(float(x) for x in wt[1])
        return w if len(w) == 3 else DEFAULT_WEIGHTS
    except (TypeError, IndexError, ValueError):
        return DEFAULT_WEIGHTS


def _undirected_neighbours(task_graph, t):
    """Dependency direction does not affect which QPUs must be linked."""
    return set(nx.all_neighbors(task_graph, t))


def _step_cost(node_graph, task_graph, tasks, nodes, partial, t, n, w, qscale):
    """Increment of extending `partial` by the single assignment t -> n.

    assignment_cost cannot score a partial mapping (it dereferences every
    neighbour), so we charge only the determined quantities: the node-local
    runtime and error for this placement, and the link cost to neighbours
    already mapped. Summed along any construction order that completes, this
    recovers assignment_cost exactly -- each workflow edge is charged once, when
    its second endpoint is placed, which is what assignment_cost totals after
    halving its double count.
    """
    task = getObjectFromList(tasks, t)
    node = getObjectFromList(nodes, n)

    w_rt, w_er, w_cm = w
    cost = w_rt * runtime(node, task) + w_er * error(node, task)

    link = 0.0
    for x in _undirected_neighbours(task_graph, t):
        if x in partial:
            link += linkCost(node, task, getObjectFromList(nodes, partial[x]),
                             getObjectFromList(tasks, x))
    cost += w_cm * link

    if QUEUE_WEIGHT:
        nat = node_graph.nodes[n].get("next_available_time", 0) or 0
        cost += QUEUE_WEIGHT * (nat / qscale)

    return cost


def _candidates(node_graph, task_graph, t, partial, demand):
    """QPUs that could host task t without violating connectivity or qubits.

    If t has mapped neighbours, candidates are restricted to the intersection of
    those neighbours' substrate neighbourhoods, so connectivity holds by
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


def beam_map(node_graph, task_graph, tasks, nodes, w=DEFAULT_WEIGHTS, qscale=1.0,
             beam_width=BEAM_WIDTH):
    """Return a list of complete task->node mappings, best-effort."""
    demands = nx.get_node_attributes(task_graph, "qubit_number")

    if not demands:
        return []

    # Widest task first: an infeasible workflow fails on the first expansion.
    order = sorted(demands, key=lambda t: demands[t], reverse=True)

    # Re-order so that, after the anchor, each task has a mapped neighbour
    # where one exists. This is what keeps the candidate sets small.
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
                                                         tasks, nodes, partial, t, n,
                                                         w, qscale)))
        if not nxt_beam:
            return []
        nxt_beam.sort(key=lambda x: x[1])
        beam = nxt_beam[:beam_width]

    return [m for m, _ in beam if len(m) == len(demands)]


def _probe(node_graph, best):
    """Log the baselines' availability term for the selected mapping.

    Measures nexAvailable on the subgraph of MAPPED nodes, which is what
    assignment_cost uses. A substrate-wide maximum would not answer the
    question: a busy substrate can still yield a mapping whose own nodes are
    all untouched.
    """
    global _probe_started
    sub = node_graph.subgraph(best.values())
    nats = [v for v in nx.get_node_attributes(sub, "next_available_time").values()
            if v is not None]
    mx = max(nats) if nats else float("nan")
    sig = nexAvailable(sub)
    mode = "a" if _probe_started else "w"
    with open(NAT_PROBE, mode) as fh:
        if not _probe_started:
            fh.write("n_mapped,max_nat,sig,saturated\n")
            _probe_started = True
        fh.write(f"{len(nats)},{mx!r},{sig!r},{int(sig == 1.0)}\n")


def policy_beam_vne(env, wt, skipped_list):
    tasks, nodes = env.current_obs

    node_graph = env.node_graph.copy()
    nx.set_node_attributes(node_graph, {q.id: q.generate_att_dict() for q in nodes})

    # Identical construction to the baselines, so the RNG is consumed the same
    # way and the workflow topologies are comparable across policies. See
    # DIRECTED above for which policies that pairs this one with.
    task_graph = nx.fast_gnp_random_graph(len(tasks), p=EDGE_PROB, seed=env.rng,
                                          directed=DIRECTED)
    lb = sorted(task_graph)
    task_graph = nx.relabel_nodes(task_graph, {lb[i]: tasks[i].id for i in range(len(lb))})
    nx.set_node_attributes(task_graph, {t.id: t.generate_att_dict() for t in tasks})

    # Normaliser for the queue term: the largest queue on the substrate, so the
    # term is dimensionless and QUEUE_WEIGHT is comparable to the other weights.
    nats = [v for v in nx.get_node_attributes(node_graph, "next_available_time").values()
            if v is not None]
    qmax = max(nats) if nats else 0.0
    qscale = qmax if qmax > 0 else 1.0

    w = _term_weights(wt)
    candidates = beam_map(node_graph, task_graph, tasks, nodes, w, qscale)

    # Connectivity and qubit checks hold by construction; we verify rather than
    # assume, so that a defect shows up as a rejected mapping instead of an
    # invalid allocation.
    candidates = [m for m in candidates
                  if check_connectivity(task_graph, m, node_graph)
                  and qubit_condition(node_graph.subgraph(m.values()), task_graph,
                                      {v: k for k, v in m.items()})]

    if not candidates:
        return None, task_graph

    best = min(candidates,
               key=lambda m: assignment_cost(node_graph.subgraph(m.values()),
                                             task_graph, tasks, nodes, m, weights=wt))

    if NAT_PROBE:
        _probe(node_graph, best)

    return list(dict(sorted(best.items(), key=lambda x: x[0])).values()), task_graph
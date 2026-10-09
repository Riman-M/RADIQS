"""
RECONSTRUCTION NOTICE
---------------------
qsimpy/utils/Checks.py is absent from the published RADIQS repository
(github.com/z-ax-qsc/RADIQS), although it is imported by all four
networked policies. This file is a reconstruction inferred from:

  * the call sites in CloudQC.py (l.64-65), IsoMorphism.py (l.37-38),
    GreedyDfs.py (l.58) and Random.py (l.31, l.53)
  * Equations 2 and 3 of Sawaika, Parampalli & Buyya, arXiv:2605.17944,
    which these functions implement (the paper calls them the "mapping"
    and "qubits" functions of Algorithm 1)

It is NOT the authors' code. Results obtained with it should be
described as a reimplementation, not a reproduction.

Argument orientation, fixed by the call sites:
  check_connectivity(task_graph, mappings, node_graph)
      mappings : {task_id -> node_id}
  qubit_condition(node_subgraph, task_graph, sub)
      sub      : {node_id -> task_id}
"""

QUBIT_ATTR = "qubit_number"


def check_connectivity(task_graph, mappings, node_graph):
    """Equation 2: G_i subset of G^{Q_i}.

    Every dependency edge in the task graph must be carried by a physical
    link between the two QPUs the endpoints are assigned to.

    The task graph may be directed (CloudQC, GreedyDfs, Random) while the
    node graph is always undirected, so edge existence is tested without
    regard to direction.
    """
    if not mappings:
        return False

    for u, v in task_graph.edges():
        if u not in mappings or v not in mappings:
            return False

        nu, nv = mappings[u], mappings[v]

        # Two dependent tasks co-located on one QPU need no link.
        if nu == nv:
            continue

        if not node_graph.has_edge(nu, nv):
            return False

    return True


def qubit_condition(node_subgraph, task_graph, sub):
    """Equation 3: qb_ij <= qb_k for every assigned (task, node) pair."""
    if not sub:
        return False

    for node_id, task_id in sub.items():
        if node_id not in node_subgraph.nodes:
            return False
        if task_id not in task_graph.nodes:
            return False

        node_qubits = node_subgraph.nodes[node_id].get(QUBIT_ATTR)
        task_qubits = task_graph.nodes[task_id].get(QUBIT_ATTR)

        if node_qubits is None or task_qubits is None:
            return False
        if task_qubits > node_qubits:
            return False

    return True

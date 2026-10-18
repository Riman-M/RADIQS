import itertools as it
import os

Qnodes = ["brisbane", "torino", "marrakesh"]

# Held constant across all four scenarios, as in the paper.
Node_types = [("brisbane", "torino", "marrakesh")]
Weights = [[(0.5, 0.5), (0.33, 0.33, 0.33)]]

# The four published scenarios, as (N_nodes, Task_per_set, Edge_prob, Datasize).
# These are specific tuples, NOT a Cartesian product -- taking the product of
# the union lists would also generate combinations the paper never runs
# (2 tasks with batch 500, 10 nodes at rho=0.9, and so on).
_DEFAULT_SCENARIOS = [
    (10, 2, 0.3,  10),   # SP-LR
    (20, 2, 0.9,  10),   # SP-MR
    (10, 4, 0.3, 500),   # LP-LR
    (20, 4, 0.9, 500),   # LP-MR
]

# SCENARIOS may be overridden from the environment so that a sweep over
# workflow size or substrate density needs no edit here. Format is
# semicolon-separated 4-tuples: "20,4,0.9,500;40,8,0.9,500".
_env = os.environ.get("SCENARIOS", "").strip()
if _env:
    SCENARIOS = []
    for item in _env.split(";"):
        item = item.strip()
        if not item:
            continue
        n, t, p_, d = item.split(",")
        SCENARIOS.append((int(n), int(t), float(p_), int(d)))
else:
    SCENARIOS = _DEFAULT_SCENARIOS

# Which policies to run this invocation. Set ALGOS to run the baselines once and
# BeamVNE separately per (beam width, queue weight), so the expensive baselines
# are not repeated for every BeamVNE configuration.
_algos = os.environ.get("ALGOS", "SoftIso,CloudQC,GreedyDfs,Random,BeamVNE")
Algo = [a.strip() for a in _algos.split(",") if a.strip()]

# Union lists, kept defined in case anything else imports them by name.
N_nodes = sorted({s[0] for s in SCENARIOS})
Task_per_set = sorted({s[1] for s in SCENARIOS})
Edge_prob = sorted({s[2] for s in SCENARIOS})
Datasize = sorted({s[3] for s in SCENARIOS})

# Same tuple order the original it.product produced:
# (n_nodes, task_per_set, edge_prob, node_types, algo, weights, data_size)
all_combinations = [
    (n, t, p, nt, a, w, d)
    for (n, t, p, d) in SCENARIOS
    for nt in Node_types
    for a in Algo
    for w in Weights
]

if __name__ == "__main__":
    print(f"{len(all_combinations)} combinations: "
          f"{len(SCENARIOS)} scenarios x {len(Algo)} algos")
    for c in all_combinations:
        print("  ", c)
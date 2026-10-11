import networkx as nx
import itertools as it
from qsimpy.policies.networked.CloudQC import policy_cloudqc
from qsimpy.policies.networked.GreedyDfs import policy_greedy_dfs
from qsimpy.policies.networked.IsoMorphism import policy_iso
from qsimpy.policies.networked.Random import policy_random
import multiprocessing as mp
from env_creator import qsimpy_env_creator
from joblib import Parallel, delayed
from time import time
from qsimpy.utils.Params import *
import hashlib
import json
from qsimpy.policies.networked.BeamVNE import policy_beam_vne
import os
import sys
import traceback

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

nx.config.backend_priority = ["parallel"]

cpus = mp.cpu_count()

RESET_STATE = True              # flip to False to reproduce the published state leak
NUM_EPISODES = int(os.environ.get("EPISODES", 50))
BEAM_WIDTH = os.environ.get("BEAM_WIDTH", "8")
QUEUE_WEIGHT = os.environ.get("QUEUE_WEIGHT", "0")
TAG = "reset" if RESET_STATE else "asis"

os.makedirs("evaluation/summary", exist_ok=True)
os.makedirs("evaluation/task", exist_ok=True)

MANIFEST = f"sweep_manifest_{TAG}.csv"


def run_exp(params):
    """Run NUM_EPISODES of one (scenario, algorithm) cell and append results.

    Writes two files per cell, both in append mode:
      rngcheck_<algo>_<tag>_<cfg>.csv   episode -> node-graph hash, to confirm
                                        that cells share substrate topologies
      agg_<algo>_<tag>_<cfg>.jsonl      one summary record per episode, with the
                                        run configuration stamped into each
                                        record so provenance survives even if a
                                        file is renamed

    Exceptions are caught and reported rather than raised, so that one failing
    cell does not abort the rest of the sweep.
    """

    PARAMS = {
        "n_nodes": params[0],
        "task_per_set": params[1],
        "edge_prob": params[2],
        "node_types": params[3],
        "algo": params[4],
        "weights": params[5],
        "data_size": params[6],
    }

    PARAMS = dict(sorted(PARAMS.items(), key=lambda item: item[0]))

    # Identifies this cell in every filename it writes. edge_prob is included so
    # that two scenarios differing only in density can never share a file.
    cfg = (f"{PARAMS['n_nodes']}n_{PARAMS['task_per_set']}t_"
           f"{PARAMS['edge_prob']}p_{PARAMS['data_size']}b_"
           f"B{BEAM_WIDTH}_Q{QUEUE_WEIGHT}")
    cell = f"{PARAMS['algo']}_{TAG}_{cfg}"

    print(f"\n=== {cell} ===", flush=True)
    print(f"Running for {PARAMS}", flush=True)

    cell_start = time()
    done = 0

    try:
        eval_env = qsimpy_env_creator(
            env_config={
                # "obs_filter": "rescale_-1_1",
                "reward_filter": None,
                "dataset": f"qdataset/qd-generation/new/synds_1_sub_{PARAMS['data_size']}.csv",
                "config": {"evaluation": True, "policy": "networked",
                           "task_per_set": PARAMS["task_per_set"],
                           "qnodes": PARAMS["n_nodes"], "params": PARAMS},
                'algo': PARAMS["algo"],
            }
        )

        skipped_list = []

        for episode in range(NUM_EPISODES):
            # First setup nodes then reset
            eval_env.setup_quantum_resources(
                qnodes=PARAMS["n_nodes"],
                qnode_names=PARAMS["node_types"],
                edge_prob=PARAMS["edge_prob"],
            )

            # Clear per-episode state. Leaving these populated is the published
            # behaviour and inflates the failure count for every policy except
            # SoftIso; see the reproduction notes.
            if RESET_STATE:
                eval_env.unfulfilled_tasks = []
                eval_env.results = []

            # Fingerprint the substrate so we can verify after the fact that two
            # cells saw the same sequence of topologies.
            eh = hashlib.md5(
                str(sorted(eval_env.node_graph.edges())).encode()
            ).hexdigest()[:8]
            with open(f"rngcheck_{cell}.csv", "a") as fh:
                fh.write(f"{episode},{eh}\n")

            _, _ = eval_env.reset(seed=22)

            terminate = False
            episode_reward = 0

            start = time()
            while not terminate:
                match PARAMS["algo"]:
                    case 'SoftIso':
                        action, task_graph = policy_iso(
                            eval_env, PARAMS["weights"], skipped_list, early_stop=True)
                    case 'Random':
                        action, task_graph = policy_random(
                            eval_env, PARAMS["weights"], skipped_list)
                    case 'GreedyDfs':
                        action, task_graph = policy_greedy_dfs(eval_env, skipped_list)
                    case 'CloudQC':
                        action, task_graph = policy_cloudqc(
                            eval_env, PARAMS["weights"], skipped_list)
                    case 'BeamVNE':
                        action, task_graph = policy_beam_vne(
                            eval_env, PARAMS["weights"], skipped_list)
                    case _:
                        print(f"Incorrect algorithm name: {PARAMS['algo']}", flush=True)
                        return (cell, 0, "bad algo name")
                obs, reward, terminate, _, info = eval_env.step_network(action, task_graph)
                if terminate:
                    break
            end = time()

            eval_env.close(end - start)

            # Log.py writes this path from PARAMS alone and knows nothing about
            # the tag or the beam/queue settings, so it must not carry them.
            par = '_'.join(map(str, PARAMS.values()))
            with open(f"evaluation/summary/networked_{par}.json") as jf:
                s = json.load(jf)

            # Stamp the configuration into the record itself.
            s["episode"] = episode
            s["algo"] = PARAMS["algo"]
            s["n_nodes"] = PARAMS["n_nodes"]
            s["task_per_set"] = PARAMS["task_per_set"]
            s["edge_prob"] = PARAMS["edge_prob"]
            s["data_size"] = PARAMS["data_size"]
            s["beam_width"] = int(BEAM_WIDTH)
            s["queue_weight"] = float(QUEUE_WEIGHT)
            s["reset_state"] = RESET_STATE
            s["node_hash"] = eh

            with open(f"agg_{cell}.jsonl", "a") as af:
                af.write(json.dumps(s) + "\n")

            done = episode + 1
            if done % 10 == 0 or done == NUM_EPISODES:
                mins = (time() - cell_start) / 60
                print(f"  {cell}: {done}/{NUM_EPISODES} episodes, {mins:.1f} min",
                      flush=True)

        mins = (time() - cell_start) / 60
        print(f"--- {cell} complete: {done} episodes in {mins:.1f} min", flush=True)
        with open(MANIFEST, "a") as mf:
            mf.write(f"{cell},{done},{mins:.2f},ok\n")
        return (cell, done, "ok")

    except Exception as exc:
        # One bad cell must not take the rest of the sweep with it.
        mins = (time() - cell_start) / 60
        print(f"!!! {cell} FAILED after {done} episodes ({mins:.1f} min): "
              f"{type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        with open(MANIFEST, "a") as mf:
            mf.write(f"{cell},{done},{mins:.2f},FAILED {type(exc).__name__}: {exc}\n")
        return (cell, done, f"FAILED {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    print(f"Total Experiments = {len(all_combinations)}")
    print(f"Episodes per cell = {NUM_EPISODES}   "
          f"BEAM_WIDTH = {BEAM_WIDTH}   QUEUE_WEIGHT = {QUEUE_WEIGHT}   "
          f"RESET_STATE = {RESET_STATE}", flush=True)

    sweep_start = time()
    # n_jobs=1: the policies share a single RNG through the environment, so
    # running cells concurrently would make the topology sequences irreproducible.
    res = Parallel(n_jobs=1, backend="threading")(
        delayed(run_exp)(par) for par in all_combinations)

    total = (time() - sweep_start) / 60
    print(f"\n===== sweep finished in {total:.1f} min =====")
    failed = [r for r in res if r and r[2] != "ok"]
    for cell, done, status in res:
        print(f"  {status:12s} {done:3d} ep  {cell}")
    if failed:
        print(f"\n{len(failed)} of {len(res)} cells FAILED:")
        for cell, done, status in failed:
            print(f"  {cell}: {status}")
        sys.exit(1)
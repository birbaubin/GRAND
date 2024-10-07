from itertools import product
import numpy as np
import argparse
from Graph import Graph
from tqdm import tqdm
from revisited_spectral import RevisitedSpectral
from erdos import SpectralAttack
from deterministic_attack import DeterministicAttack
from helpers import *
import os


# Parse arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("--dataset", type=str, default="polblogs")
argparser.add_argument("--type", type=str, default="D", choices=["D", "P", "DP", "DPD"])
argparser.add_argument("--n_experiments", type=int, default=5)
argparser.add_argument("--graph1_props", type=float, nargs='+', default=[0.0])
argparser.add_argument("--proba_params", type=float, nargs='+', default=[0.3, 0.3, 0.3])
argparser.add_argument("--sanity_check_optim", type=bool, default=True)


args = argparser.parse_args()
number_of_experiments = args.n_experiments
dataset_name = args.dataset
optimize_sanity_check = args.sanity_check_optim
graph1_props = args.graph1_props
expe_type = args.type
proba_params = args.proba_params
common_props = [0]

# Load dataset
if dataset_name != "block":
    G = Graph.from_txt(f"datasets/{dataset_name}.txt")
else:
    G = Graph.block_from_txt(f"datasets/{dataset_name}.txt")


# create dataset.csv file if it does not exist
with open(f"logs/{dataset_name}.csv", "a") as f:
    if os.stat(f"logs/{dataset_name}.csv").st_size == 0:
        f.write(
        "expe,attack_type,alpha,beta,gamma,graph1_prop,common_prop,iter_number," +
        "impossible_edges,reconstructed_edges,unknown_edges,TP,FP,TN,FN,time\n")
    f.close()

A = np.dot(G.adj_matrix, G.adj_matrix)

if expe_type == "P":
    print("######################### Spectral attack (Erdos et al.) ########################")
    start = time.time()
    attack = SpectralAttack(A, 0.5)
    attack.run()
    end = time.time()
    Gstar = attack.get_reconstructed_graph()
    log_graph_stats(0, 1, 0, expe_type, 
            [None, None, None], 0, end-start, f"logs/{dataset_name}.csv", Gstar, G)

elif expe_type == "D":
    print("\n\n########################## Deterministic ##########################\n\n")
    for expe in range(number_of_experiments):
        for graph1_prop, common_prop in product(graph1_props, common_props):
            G1, G2 = G.split_dataset(common_prop=common_prop, graph1_prop=graph1_prop)
            start = time.time()
            deterministic_attack = DeterministicAttack(G1, A)
            deterministic_attack.run()
            end = time.time()
            Gstar = deterministic_attack.get_reconstructed_graph()
            log_graph_stats(graph1_prop, common_prop, expe, expe_type, 
            [None, None, None], 0, end-start, f"logs/{dataset_name}.csv", Gstar, G)

elif expe_type == "DP":
    print("\n\n########################## Deterministic - Probabilistic ##########################\n\n")
    for expe in range(number_of_experiments):
        for graph1_prop, common_prop in product(graph1_props, common_props):
            G1, G2 = G.split_dataset(common_prop=common_prop, graph1_prop=graph1_prop)
            start = time.time()
            deterministic_attack = DeterministicAttack(G1, A)
            deterministic_attack.run()
            Gstar = deterministic_attack.get_reconstructed_graph()

            rev_spectral_attack = RevisitedSpectral(Gstar, A)
            if proba_params[2] == 0:
                rev_spectral_attack.run(alpha=proba_params[0], beta=proba_params[1])
            else:
                rev_spectral_attack.run(alpha=proba_params[0], beta=proba_params[1], gamma=proba_params[2])
            
            end = time.time()
            Gstar = rev_spectral_attack.get_reconstructed_graph()
            log_graph_stats(graph1_prop, common_prop, expe, expe_type, 
            proba_params, 0, end-start, f"logs/{dataset_name}.csv", Gstar, G)

elif expe_type == "DPD":
    print("\n\n########################## Deterministic - Probabilistic - Deterministic #########################\n\n")
    for expe in range(number_of_experiments):
        for graph1_prop, common_prop in product(graph1_props, common_props):
            G1, G2 = G.split_dataset(common_prop=common_prop, graph1_prop=graph1_prop)
            start = time.time()
            deterministic_attack = DeterministicAttack(G1, A)
            deterministic_attack.run()
            Gstar = deterministic_attack.get_reconstructed_graph()

            rev_spectral_attack = RevisitedSpectral(Gstar, A)
            if proba_params[2] == 0:
                rev_spectral_attack.run(alpha=proba_params[0], beta=proba_params[1])
            else:
                rev_spectral_attack.run(alpha=proba_params[0], beta=proba_params[1], gamma=proba_params[2])
            if optimize_sanity_check:
                rev_spectral_attack.sanity_check_with_high_loss()
            else:
                rev_spectral_attack.sanity_check()
            Gstar = rev_spectral_attack.get_reconstructed_graph()
            
            deterministic_attack = DeterministicAttack(Gstar, A)
            deterministic_attack.run()
            Gstar = deterministic_attack.get_reconstructed_graph()
            Gstar.fix_edges()
            end = time.time()
            log_graph_stats(graph1_prop, common_prop, expe, expe_type, 
            proba_params, 0, end-start, f"logs/{dataset_name}.csv", Gstar, G)
else:
    print("Unknown experiment type")
    exit(1)


    

        
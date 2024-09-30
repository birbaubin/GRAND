from itertools import product
import numpy as np
import argparse
from Graph import Graph
from tqdm import tqdm
from revisited_spectral import RevisitedSpectral
from erdos import SpectralAttack
from deterministic_attack import DeterministicAttack
from helpers import *


argparser = argparse.ArgumentParser()
argparser.add_argument("--dataset", type=str, default="polblogs")
argparser.add_argument("--type", type=str, default="D", choices=["D", "P", "DP", "DPD"])
argparser.add_argument("--n_experiments", type=int, default=5)
argparser.add_argument("--graph1_props", type=float, nargs='+', default=[0.0])


args = argparser.parse_args()
number_of_experiments = args.n_experiments
dataset_name = args.dataset

if dataset_name != "block":
    G = Graph.from_txt(f"datasets/{dataset_name}.txt")
else:
    G = Graph.block_from_txt(f"datasets/{dataset_name}.txt")

graph1_props = args.graph1_props
expe_type = args.type
common_props = [0]


A = np.dot(G.adj_matrix, G.adj_matrix)

if expe_type == "P":
    print("1. Spectral attack (Erdos et al.)")
    attack = SpectralAttack(A, 0.5)
    attack.run()
    reconstructed_graph = attack.get_reconstructed_graph()
    log_graph_stats(0,0,0,"spectral", reconstructed_graph, 0,0,f"logs/{dataset_name}", G)

elif expe_type == "D":
    print("\n######## 2. Deterministic attacks ########")
    for expe in range(number_of_experiments):
        for graph1_prop, common_prop in product(graph1_props, common_props):
            G1, G2 = G.split_dataset(common_prop=common_prop, graph1_prop=graph1_prop)
            deterministic_attack = DeterministicAttack(G1, A)
            deterministic_attack.run()
            Gstar = deterministic_attack.get_reconstructed_graph()
            log_graph_stats(graph1_prop, common_prop, expe, "D", expe_type, 0, 0, f"logs/{dataset_name}", G)

elif expe_type == "DP":
    print("\n######## Deterministic - Probabilistic ########")
    for expe in range(number_of_experiments):
        for graph1_prop, common_prop in product(graph1_props, common_props):
            G1, G2 = G.split_dataset(common_prop=common_prop, graph1_prop=graph1_prop)
            deterministic_attack = DeterministicAttack(G1, A)
            deterministic_attack.run()
            Gstar = deterministic_attack.get_reconstructed_graph()
            rev_spectral_attack = RevisitedSpectral(Gstar, A)
            rev_spectral_attack.run()
            Gstar = rev_spectral_attack.get_reconstructed_graph()
            log_graph_stats(graph1_prop, common_prop, expe, expe_type, Gstar, 0, 0, f"logs/{dataset_name}", G)


elif expe_type == "DPD":
    print("\n######## Deterministic - Probabilistic - Deterministic ########")
    for expe in range(number_of_experiments):
        for graph1_prop, common_prop in product(graph1_props, common_props):
            G1, G2 = G.split_dataset(common_prop=common_prop, graph1_prop=graph1_prop)
            deterministic_attack = DeterministicAttack(G1, A)
            deterministic_attack.run()
            Gstar = deterministic_attack.get_reconstructed_graph()
            rev_spectral_attack = RevisitedSpectral(Gstar, A)
            rev_spectral_attack.run()
            rev_spectral_attack.sanity_check()
            Gstar = rev_spectral_attack.get_reconstructed_graph()
            deterministic_attack = DeterministicAttack(Gstar, A)
            deterministic_attack.run()
            Gstar = deterministic_attack.get_reconstructed_graph()
            Gstar.fix_edges()
            log_graph_stats(graph1_prop, common_prop, expe, expe_type, Gstar, 0, 0, f"logs/{dataset_name}", G)
            error_edges = np.argwhere(Gstar.adj_matrix != G.adj_matrix)
            print("Error edges: ", error_edges)

            for edge in error_edges:
                print(edge, Gstar.adj_matrix[edge[0], edge[1]], G.adj_matrix[edge[0], edge[1]])

else:
    print("Unknown experiment type")
    exit(1)

print("Done")


    

        
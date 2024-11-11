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
import networkx as nx

def generate_barabasi_graph_from_str(dataset_name):
    n = int(dataset_name.split("_")[1])
    m = int(dataset_name.split("_")[2])
    return Graph.barabasi_albert_graph(n, m, m)

def generate_random_graph_from_str(dataset_name):
    n = int(dataset_name.split("_")[1])
    p = float(dataset_name.split("_")[2])
    Gnx = nx.erdos_renyi_graph(n, p)
    return Graph.from_adj_matrix(nx.to_numpy_array(Gnx).astype(int))


# Parse arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("--dataset", type=str, default="polblogs")
argparser.add_argument("--types", type=str, nargs='+')
argparser.add_argument("--n_experiments", type=int, default=5)
argparser.add_argument("--graph1_props", type=float, nargs='+', default=[0.0])
argparser.add_argument("--proba_params", type=float, nargs='+', default=[0.3, 0.3, 0.3])
argparser.add_argument('--sanity_check_optim', action=argparse.BooleanOptionalAction)
argparser.add_argument('--log_deterministic', action=argparse.BooleanOptionalAction)


args = argparser.parse_args()
number_of_experiments = args.n_experiments
dataset_name = args.dataset
props = args.graph1_props
graph1_props = np.linspace(props[0], props[1], num=int((props[1] - props[0]) / props[2]) + 1, endpoint=True)
expe_types = args.types
log_deterministic = args.log_deterministic
common_props = [0]

proba_params = (0.0, 0.0, 0.0)
optimize_sanity_check = 0


# Load dataset
if not dataset_name.startswith("barabasi") and not dataset_name.startswith("random"):
    G = Graph.from_txt(f"datasets/{dataset_name}.txt")
    A = np.dot(G.adj_matrix, G.adj_matrix)


print(f"Dataset: {dataset_name} loaded successfully")

# create benchmark log file if it does not exist
with open(f"logs/benchmark/{dataset_name}.csv", "a") as f:
    if os.stat(f"logs/benchmark/{dataset_name}.csv").st_size == 0:
        f.write(
        "expe,attack_type,optim,alpha,beta,gamma,graph1_prop,common_prop,iter_number," +
        "impossible_edges,reconstructed_edges,unknown_edges,TP,FP,TN,FN,time\n")
    f.close()

for expe in range(number_of_experiments):
    for graph1_prop, common_prop in product(graph1_props, common_props):
        if dataset_name.startswith("barabasi"):
            print("Generating barabasi graph... ")
            G = generate_barabasi_graph_from_str(dataset_name)
            A = np.dot(G.adj_matrix, G.adj_matrix)

        elif dataset_name.startswith("random"):
            print("Generating random graph... ")
            G = generate_random_graph_from_str(dataset_name)
            A = np.dot(G.adj_matrix, G.adj_matrix)

        G1, G2 = G.split_dataset(common_prop=common_prop, graph1_prop=graph1_prop)

        for expe_type in expe_types:
            if expe_type == "D":
                print("Running deterministic attack")
                deterministic_attack = DeterministicAttack(G1, A, graph1_prop=graph1_prop, dataset_name=dataset_name, log=log_deterministic, expe_number=expe)
                deterministic_attack.run()
                Gstar = deterministic_attack.get_Gstar()

                if graph1_prop == 0 and not dataset_name.startswith("barabasi") and not dataset_name.startswith("random"):
                    np.savetxt(f"logs/deterministic_results/{dataset_name}_deter_adj_matrix.csv", Gstar.adj_matrix, delimiter=",", fmt="%d")
            
            elif expe_type == "H":
                attack = SpectralAttack(G1, A, 0.5)
                attack.run()
                Gstar = attack.get_Gstar()
            
            elif expe_type.startswith("DH"):
                proba_params = tuple(map(float, expe_type.split("_")[1:]))
                deterministic_attack = DeterministicAttack(G1, A)
                deterministic_attack.run()
                Gstar = deterministic_attack.get_Gstar()
                unknowns = Gstar.stats()[2]
                if unknowns != 0:
                    rev_spectral_attack = RevisitedSpectral(Gstar, A)
                    rev_spectral_attack.run(alpha=proba_params[0], beta=proba_params[1], gamma=proba_params[2])
                    Gstar = rev_spectral_attack.get_Gstar()
            
            elif expe_type.startswith("DDH"):
                proba_params = expe_type.split("_")[1:]
                optimize_sanity_check = proba_params[3]
                print(f"Optimize sanity check: {optimize_sanity_check}")
                deterministic_attack = DeterministicAttack(G1, A)
                deterministic_attack.run()
                Gstar = deterministic_attack.get_Gstar()
                unknowns = Gstar.stats()[2]

                if unknowns != 0:
                    rev_spectral_attack = RevisitedSpectral(Gstar, A)
                    rev_spectral_attack.run(alpha=float(proba_params[0]), beta=float(proba_params[1]), gamma=float(proba_params[2]))
                    if not bool(optimize_sanity_check):
                        print("-------> Running sanity check")
                        rev_spectral_attack.sanity_check_with_high_loss()
                    else:
                        print("-------> Running sanity check with early stop")
                        rev_spectral_attack.sanity_check_with_early_stop(0.025)

                    Gstar = rev_spectral_attack.get_Gstar()
                    deterministic_attack = DeterministicAttack(Gstar, A)
                    deterministic_attack.run(run_degree=False)
                    Gstar = deterministic_attack.get_Gstar()
                    Gstar.fix_edges()
            else:
                print("Unknown experiment type")
                continue    


            log_graph_stats(graph1_prop, common_prop, expe, expe_type, 
                proba_params, optimize_sanity_check, 0, 0, f"logs/benchmark/{dataset_name}.csv", Gstar, G)

        print(f"Experiment {expe} done for {expe_type} with graph1_prop={graph1_prop} and common_prop={common_prop}")

print(f"Experiments done for {dataset_name}")



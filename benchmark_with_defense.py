from itertools import product
import numpy as np
import argparse
from Graph import Graph
from revisited_spectral import RevisitedSpectral
from erdos import SpectralAttack
from helpers import *
import os
import networkx as nx
from defense import *

def generate_barabasi_graph_from_str(dataset_name):
    n = int(dataset_name.split("_")[1])
    m = int(dataset_name.split("_")[2])
    return Graph.barabasi_albert_graph(n, m, m)

def generate_random_graph_from_str(dataset_name):
    n = int(dataset_name.split("_")[1])
    p = float(dataset_name.split("_")[2])
    Gnx = nx.erdos_renyi_graph(n, p)
    return Graph.from_adj_matrix(nx.to_numpy_array(Gnx).astype(int))


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset", type=str, default="polblogs")
    argparser.add_argument("--types", type=str, nargs='+')
    argparser.add_argument("--n_experiments", type=int, default=5)
    argparser.add_argument("--graph1_props", type=float, nargs='+', default=[0.0])
    argparser.add_argument("--proba_params", type=float, nargs='+', default=[0.3, 0.3, 0.3])
    argparser.add_argument('--sanity_check_optim', action=argparse.BooleanOptionalAction)
    argparser.add_argument('--log_deterministic', action=argparse.BooleanOptionalAction)
    argparser.add_argument('--epsilons',  type=float, nargs='+', default=[1.0])

    args = argparser.parse_args()
    number_of_experiments = args.n_experiments
    dataset_name = args.dataset
    sample_start, sample_stop, sample_increment = args.graph1_props
    expe_types = args.types
    log_deterministic = args.log_deterministic
    common_props = [0]
    epsilons = args.epsilons

    proba_params = (0.0, 0.0, 0.0)
    complete_graph = 0


    # Load dataset
    if not dataset_name.startswith("barabasi") and not dataset_name.startswith("random"):
        G = Graph.from_txt(f"datasets/{dataset_name}.txt")
        adj = G.adjacency_matrix()


    print(f"Dataset: {dataset_name} loaded successfully")

    # create benchmark log file if it does not exist
    with open(f"logs/benchmark_with_defense2/{dataset_name}.csv", "a") as f:
        if os.stat(f"logs/benchmark_with_defense2/{dataset_name}.csv").st_size == 0:
            f.write(
            "expe,attack_type,optim,alpha,beta,gamma,graph1_prop,common_prop,iter_number,epsilon," +
            "impossible_edges,reconstructed_edges,unknown_edges,TP,FP,TN,FN,G2_distance,time\n")
        f.close()

    
    for expe in range(number_of_experiments):
        for epsilon in epsilons:
            if dataset_name.startswith("barabasi"):
                print("Generating barabasi graph... ")
                G = generate_barabasi_graph_from_str(dataset_name)
                adj = G.adjacency_matrix()
                # A = np.dot(adj, adj)

            elif dataset_name.startswith("random"):
                print("Generating random graph... ")
                G = generate_random_graph_from_str(dataset_name)
                adj = G.adjacency_matrix()
                # A = np.dot(adj, adj)
            
            
            # A = publish_G2(G, epsilon)
            A = np.dot(adj, adj)

            for Ga_prop, Ga in G.gradual_sample(sample_start, sample_stop, sample_increment):
                common_prop = 0

                for expe_type in expe_types:
                    
                    if expe_type == "H":
                        start = time.time()
                        attack = SpectralAttack(Ga, A, 0.5)
                        attack.run()
                        end = time.time()

                        Gstar = attack.get_Gstar()
                    
                    elif expe_type == "DH":
                        rev_spectral_attack = RevisitedSpectral(Ga, A)
                        rev_spectral_attack.run(alpha=proba_params[0], beta=proba_params[1], gamma=proba_params[2])
                        Gstar = rev_spectral_attack.get_Gstar()

                        end = time.time()

                    else:
                        print("Unknown experiment type")
                        continue    


                    log_graph_stats_with_defense(Ga_prop, common_prop, expe, expe_type, 
                        proba_params, complete_graph, 0, end-start, f"logs/benchmark_with_defense2/{dataset_name}.csv", Gstar, G, epsilon)

            print(f"----> Experiment {expe} done for {expe_type} with graph1_prop={Ga_prop}")

    print(f"----> Finished run {expe} for {dataset_name}")

print(f"----> Experiments done for {dataset_name}")



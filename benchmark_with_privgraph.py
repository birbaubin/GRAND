from PrivGraph.main import *
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


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset", type=str, default="polblogs")
    argparser.add_argument("--types", type=str, nargs='+')
    argparser.add_argument("--n_experiments", type=int, default=5)
    argparser.add_argument("--graph1_props", type=float, nargs='+', default=[0.0])
    argparser.add_argument("--proba_params", type=float, nargs='+', default=[0.3, 0.3, 0.3])
    argparser.add_argument('--sanity_check_optim', action=argparse.BooleanOptionalAction)
    argparser.add_argument('--log_deterministic', action=argparse.BooleanOptionalAction)
    argparser.add_argument('--epsilon', type=float, default=1.5)


    args = argparser.parse_args()
    number_of_experiments = args.n_experiments
    dataset_name = args.dataset
    sample_start, sample_stop, sample_increment = args.graph1_props
    expe_types = args.types
    log_deterministic = args.log_deterministic
    common_props = [0]

    proba_params = (0.0, 0.0, 0.0)
    complete_graph = 1
    epsilon = args.epsilon

    # Load dataset
    if not dataset_name.startswith("barabasi") and not dataset_name.startswith("random"):
        G = Graph.from_txt(f"datasets/{dataset_name}.txt")
        adj = G.adjacency_matrix()
        
    print(f"Dataset: {dataset_name} loaded successfully")

    # create benchmark log file if it does not exist
    with open(f"logs/benchmark/{dataset_name}.csv", "a") as f:
        if os.stat(f"logs/benchmark/{dataset_name}.csv").st_size == 0:
            f.write(
            "expe,attack_type,optim,alpha,beta,gamma,graph1_prop,common_prop,iter_number," +
            "impossible_edges,reconstructed_edges,unknown_edges,TP,FP,TN,FN,G2_distance,time\n")
        f.close()

    for expe in range(number_of_experiments):
        
        G_dp = Graph.from_adj_matrix(main_in_pipeline(dataset_name=dataset_name,eps=epsilon,e1_r=1/3,e2_r=1/3,N=20,t=1.0,exp_num=10,save_csv=False))
        A = np.dot(G_dp.adjacency_matrix(), G_dp.adjacency_matrix())
    

        for Ga_prop, Ga in G.gradual_sample(sample_start, sample_stop, sample_increment):
            G1 = Ga
            graph1_prop = Ga_prop
            common_prop = 0

            for expe_type in expe_types:
                start = 0
                end = 0
                if expe_type == "D":
                    print("Running deterministic attack")
                    deterministic_attack = DeterministicAttack(G1, A, graph1_prop=graph1_prop, dataset_name=dataset_name, log=log_deterministic, expe_number=expe)
                    deterministic_attack.run()
                    Gstar = deterministic_attack.get_Gstar()

                    if graph1_prop == 0 and not dataset_name.startswith("barabasi") and not dataset_name.startswith("random"):
                        np.savetxt(f"logs/deterministic_results/{dataset_name}_deter_adj_matrix.csv", Gstar.adjacency_matrix(), delimiter=",", fmt="%d")
                
               
                elif expe_type == "H":
                    
                    start = time.time()
                    attack = SpectralAttack(G1, A, 0.5)
                    attack.run()
                    end = time.time()

                    Gstar = attack.get_Gstar()
                
                elif expe_type.startswith("DH"):
                    start = time.time()
                    proba_params = tuple(map(float, expe_type.split("_")[1:]))
                    deterministic_attack = DeterministicAttack(G1, A, graph1_prop=graph1_prop, dataset_name=dataset_name, log=log_deterministic, expe_number=expe)
                    deterministic_attack.run()
                    Gstar = deterministic_attack.get_Gstar()
                    unknowns = Gstar.stats()[2]
                    if unknowns != 0:
                        rev_spectral_attack = RevisitedSpectral(Gstar, A)
                        rev_spectral_attack.run(alpha=proba_params[0], beta=proba_params[1], gamma=proba_params[2])
                        Gstar = rev_spectral_attack.get_Gstar()
                    end = time.time()
                    
                elif expe_type.startswith("DDH"):
                    
                    proba_params = expe_type.split("_")[1:]
                    complete_graph = proba_params[3]
                    start = time.time()
                    deterministic_attack = DeterministicAttack(G1, A, graph1_prop=graph1_prop, dataset_name=dataset_name, log=log_deterministic, expe_number=expe)
                    deterministic_attack.run()
                    Gstar = deterministic_attack.get_Gstar()
                    unknowns = Gstar.stats()[2]

                    print("Unknowns", unknowns)

                    if unknowns != 0:
                        rev_spectral_attack = RevisitedSpectral(Gstar, A)
                        rev_spectral_attack.run(alpha=float(proba_params[0]), beta=float(proba_params[1]), gamma=float(proba_params[2]))
                        rev_spectral_attack.sanity_check_with_high_loss()
                        
                        Gstar = rev_spectral_attack.get_Gstar()
                        deterministic_attack = DeterministicAttack(Gstar, A)
                        deterministic_attack.run(run_degree=False)

                        if int(complete_graph) == 1:
                            deterministic_attack.complete_graph() 
                            Gstar = deterministic_attack.get_Gstar()
                            Gstar.fix_edges()
                        else:
                            Gstar = deterministic_attack.get_Gstar()
                    
                    end = time.time()

                else:
                    print("Unknown experiment type")
                    continue    


                log_graph_stats_with_defense(graph1_prop, common_prop, expe, expe_type, 
                    proba_params, complete_graph, 0, end-start, f"logs/benchmark_with_privgraph/{dataset_name}.csv", Gstar, G, G_dp, epsilon)

            print(f"----> Experiment {expe} done for {expe_type} with graph1_prop={graph1_prop}")

    print(f"----> Finished run {expe} for {dataset_name}")

print(f"----> Experiments done for {dataset_name}")



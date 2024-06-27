from deterministic_attack.deterministic_attack import *
from probabilistic_attack.probabilistic_attack import *
from helpers.helpers import *
from helpers.Graph import Graph
from itertools import product
import numpy as np
import argparse
import pandas as pd


argparser = argparse.ArgumentParser()
argparser.add_argument("--dataset", type=str, default="polblogs")
argparser.add_argument("--heuristic", type=str, default="preferential_attachment", choices=["common_neighbors", "preferential_attachment"])
argparser.add_argument("--number_of_experiments", type=int, default=5)
argparser.add_argument("--addition_proportion", type=float, default=1e-3)
argparser.add_argument("--graph1_props", type=float, nargs='+', default=[0.1])
argparser.add_argument("--prediction_type",  default=['G1'], choices=["G1", "A", "model"], help='attacks')


args = argparser.parse_args()

number_of_experiments = args.number_of_experiments
dataset_name = args.dataset
heuristic = args.heuristic
prediction_type = args.prediction_type

dataset = Graph.from_txt(f"datasets/{dataset_name}.txt")
DEBUG = True


graph1_props = args.graph1_props
common_props = [0]
addition_proportion = args.addition_proportion

for expe in range(number_of_experiments):
    for graph1_prop, common_prop in product(graph1_props, common_props):
        graph1, graph2 = dataset.split_dataset(common_prop=common_prop, graph1_prop=graph1_prop)
        A = np.dot(dataset.adj_matrix, dataset.adj_matrix)
        print("\n######## Graph1 ########")
        print(graph1.adj_matrix)
        print("\n######## Graph2 ########")
        print(graph2.adj_matrix)
        print("\n######## Common neighbors matrix ########")
        print(A)
        print("\n######## General information ########")
        print("Number of edges in G : ",len(dataset.edges() ))
        print("Number of nodes in G : ",len(dataset.nodes ))
        print("Number of edges in G2 : ",len(graph2.edges() ))
        print("Number of edges in G1 : ",len(graph1.edges() ) )

        reconstructed_graph = graph1_copy(graph1)

        iteration_deterministic = 0
        iteration_probabilistic = 0
        continue_deterministic = True

        while continue_deterministic:
            start = time.time()
            reconstructed_graph, number_modifs1 = matching_attacks(reconstructed_graph, A)
            reconstructed_graph, number_modifs2 = completion_attacks(reconstructed_graph, A)
            end = time.time()
            display_reconstruction_metrics(reconstructed_graph, dataset)
            log_graph_stats(graph1_prop, common_prop, expe, "deterministic", reconstructed_graph, 
            iteration_deterministic, end-start,f"logs_hybrid/{dataset_name}_{heuristic}_{prediction_type}.csv", dataset)

# # 
            if number_modifs1 + number_modifs2 == 0:
                continue_deterministic = False
            
            iteration_deterministic += 1


        reconstructed_graph_copy = reconstructed_graph.copy()

        # reconstructed_graph.fix_edges()
        # rae_result = rae(reconstructed_graph, dataset)
        # print("RAE before Erdos", rae_result)

        # reconstructed_graph.fix_edges()
        # np.save("reconstructed_graph.npy", reconstructed_graph.adj_matrix)

        erdos_result = pd.read_csv(f"erdos/results/{dataset_name}/M", sep="\t", header=None).to_numpy()
        unknown_edges = reconstructed_graph.unknown_edges()
        # print(len(unknown_edges))
        for edge in unknown_edges:
            n1, n2 = edge
            if erdos_result[n1, n2] == 1:
                reconstructed_graph.add_edge(edge)
            else:
                reconstructed_graph.remove_edge(edge)

        # print(len(reconstructed_graph.unknown_edges()))
        rae_result = rae(reconstructed_graph, dataset)
        print("RAE after Erdos", rae_result)

        print(reconstructed_graph.adj_matrix == dataset.adj_matrix)
            
    
#             start = time.time()
#             step_size = int(len(dataset.edges())*addition_proportion)
#             display_reconstruction_metrics(reconstructed_graph, dataset)

#             if prediction_type == "G1":
#                 probabilistic_function = similarity_based_completion
#             elif prediction_type == "A":
#                 probabilistic_function = A_based_completion

#             reconstructed_graph, number_modifs_proba = probabilistic_function(reconstructed_graph, step_size, A, method=heuristic)
#             end = time.time()

#             if number_modifs_proba == 0 or len(reconstructed_graph.unknown_edges()) == 0:
#                 continue_main_loop = False

#             display_reconstruction_metrics(reconstructed_graph, dataset)
#             log_graph_stats(graph1_prop, common_prop, expe, "probabilistic", reconstructed_graph, 
#             iteration_probabilistic, end-start,f"logs_hybrid/{dataset_name}_{heuristic}_{prediction_type}.csv", dataset)
#             iteration_probabilistic += 1
# # 
#         reconstructed_graph.fix_edges()
#         accuracy = reconstruction_accuracy(reconstructed_graph, dataset)
#         distance = frobenius_distance(reconstructed_graph, dataset)
#         rae_stat = rae(reconstructed_graph, dataset)
#         edge_accuracy = edge_identification_accuracy(reconstructed_graph, dataset)

#         print("Edge accuracy", edge_accuracy)



        


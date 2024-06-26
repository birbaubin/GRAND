from deterministic_attack.deterministic_attack import *
from probabilistic_attack.probabilistic_attack import *
from helpers.helpers import *
from helpers.Graph import Graph
from itertools import product
import numpy as np


number_of_experiments = 5
dataset_name = "acm"
dataset = Graph.from_txt(f"datasets/{dataset_name}.txt")
DEBUG = True

def mixed_attack_pipeline():

    graph1_props = [0.1]
    common_props = [0]
    addition_proportion = 1e-3

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
            continue_main_loop = True

            while continue_main_loop:
                
                continue_deterministic = True

                while continue_deterministic:
                    start = time.time()
                    reconstructed_graph, number_modifs1 = matching_attacks(reconstructed_graph, A)
                    reconstructed_graph, number_modifs2 = completion_attacks(reconstructed_graph, A)
                    end = time.time()
                    display_reconstruction_metrics(reconstructed_graph, dataset)
                    log_graph_stats(graph1_prop, common_prop, expe, "deterministic", reconstructed_graph, iteration_deterministic, end-start,f"logs_hybrid/{dataset_name}.csv", dataset)
    # # 
                    if number_modifs1 + number_modifs2 == 0:
                        continue_deterministic = False
                    
                    iteration_deterministic += 1
        
                start = time.time()
                step_size = int(len(dataset.edges())*addition_proportion)
                display_reconstruction_metrics(reconstructed_graph, dataset)
                reconstructed_graph, number_modifs_proba = similarity_based_completion(reconstructed_graph, step_size, A, method="preferential_attachment")
                end = time.time()
                if number_modifs_proba == 0 or len(reconstructed_graph.unknown_edges()) == 0:
                    continue_main_loop = False

                display_reconstruction_metrics(reconstructed_graph, dataset)
                log_graph_stats(graph1_prop, common_prop, expe, "probabilistic", reconstructed_graph, iteration_probabilistic, end-start,f"logs_hybrid/{dataset_name}.csv", dataset)
                iteration_probabilistic += 1
# 
            reconstructed_graph.fix_edges()
            accuracy = reconstruction_accuracy(reconstructed_graph, dataset)
            distance = frobenius_distance(reconstructed_graph, dataset)
            rae_stat = rae(reconstructed_graph, dataset)
            edge_accuracy = edge_identification_accuracy(reconstructed_graph, dataset)

            print("Edge accuracy", edge_accuracy)



def tests():
    graph1_props = [0.1]
    common_props = [0]

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

            iteration_count = 0
            while len(reconstructed_graph.unknown_edges()) > 0:
                iteration_count += 1
                reconstructed_graph = similarity_based_completion(reconstructed_graph, 100, A, method="common_neighbors")
                display_reconstruction_metrics(reconstructed_graph, dataset)

    

mixed_attack_pipeline()
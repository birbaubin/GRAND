from deterministic_attack.deterministic_attack import *
from probabilistic_attack.probabilistic_attack import *
from helpers.helpers import *
from helpers.Graph import Graph
from itertools import product
import numpy as np
import argparse
from revisited_erdos_2 import *


argparser = argparse.ArgumentParser()
argparser.add_argument("--dataset", type=str, default="polblogs")
argparser.add_argument("--number_of_experiments", type=int, default=5)
argparser.add_argument("--graph1_props", type=float, nargs='+', default=[0.1])


args = argparser.parse_args()
DEBUG = True
number_of_experiments = args.number_of_experiments
dataset_name = args.dataset
graph1_props = args.graph1_props
common_props = [0]

if not dataset_name.startswith("block"):
    dataset = Graph.from_txt(f"datasets/{dataset_name}.txt")
else:
    dataset = Graph.block_from_txt(f"datasets/{dataset_name}.txt")

A = np.dot(dataset.adj_matrix, dataset.adj_matrix)
number_of_edges = np.sum(np.diag(A)) / 2



for expe in range(number_of_experiments):
    for graph1_prop, common_prop in product(graph1_props, common_props):
        graph1, graph2 = dataset.split_dataset(common_prop=common_prop, graph1_prop=graph1_prop)

        print("\n######## General information ########")
        print("Number of edges in G : ",len(dataset.edges() ))
        print("Number of nodes in G : ",len(dataset.nodes ))
        print("Number of edges in G2 : ",len(graph2.edges() ))
        print("Number of edges in G1 : ",len(graph1.edges() ) )


        # Copy graph1
        print("--- G1 COPY ---")
        reconstructed_graph = graph1_copy(graph1)
        display_graph_stats(reconstructed_graph)
        log_graph_stats(graph1_prop, common_prop, expe, "copy", reconstructed_graph, 
                0, 0,f"logs_erdos/{dataset_name}_expe.csv", dataset)

        print("--- DEGREE AND HUB ATTACK ---")
        # Degree and hub attack
        number_modifs_degree = 0
        number_modifs_hub = 0
        reconstructed_graph, number_modifs_degree = degree_attack(reconstructed_graph, A , [1, 2])
        # reconstructed_graph, number_modifs_hub = hub_and_isolated_node_nattack(reconstructed_graph, A, 1)
        print("Number of modifs degree = ", number_modifs_degree)
        print("Number of modifs hub = ", number_modifs_hub)

        log_graph_stats(graph1_prop, common_prop, expe, "degree_hub", reconstructed_graph, 
                0, 0,f"logs_erdos/{dataset_name}_expe.csv", dataset)


        iteration_deterministic = 0
        iteration_probabilistic = 0
        continue_deterministic = True

        # deterministic attacks
        print("--- OTHER DETERMINISTIC ATTACKS ---")
        while continue_deterministic:

            number_modifs_matching = 0
            number_modifs_completion = 0
            number_modifs_hub = 0
            number_modifs_rectangle = 0
            number_modifs_triangle = 0

            start = time.time()
            reconstructed_graph, number_modifs_matching = matching_attacks(reconstructed_graph, A)
            reconstructed_graph, number_modifs_completion = completion_attacks(reconstructed_graph, A)
            reconstructed_graph, number_modifs_rectangle = rectangle_attack(reconstructed_graph, A, [1, 2, 3, 4, 5])
            reconstructed_graph, number_modifs_triangle = triangle_attack(reconstructed_graph, A)
            reconstructed_graph, number_modifs_degree = weird_attack(reconstructed_graph, A)
            end = time.time()
            print("Number of modifs matching = ", number_modifs_matching)
            print("Number of modifs completion = ", number_modifs_completion)
            print("Number of modifs rectangle = ", number_modifs_rectangle)
            print("Number of modifs triangle = ", number_modifs_triangle)


            display_graph_stats(reconstructed_graph)

            # log_graph_stats(graph1_prop, common_prop, expe, "deterministic", reconstructed_graph,
            # iteration_deterministic, end-start,f"logs_erdos/{dataset_name}_expe.csv", dataset)

        
            if number_modifs_matching + number_modifs_completion + number_modifs_hub + number_modifs_rectangle + number_modifs_degree + number_modifs_triangle == 0:
                continue_deterministic = False
            
            number_modifs_degree = 0
            iteration_deterministic += 1

    
        if len(reconstructed_graph.edges()) < number_of_edges:
            print("--- REVISITED ERDOS ---")
            # reconstructed_graph.fix_edges()
            start = time.time()
            reconstructed_graph = Graph.from_adj_matrix(GRAND(A, reconstructed_graph, 0.5))
            end = time.time()
            log_graph_stats(graph1_prop, common_prop, expe, "erdos", reconstructed_graph,
            0, end-start,f"logs_erdos/{dataset_name}_expe.csv", dataset)

           
        display_graph_stats(reconstructed_graph)


        


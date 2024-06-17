from helpers import *
from Graph import Graph
from itertools import product 
from deterministic_attack import matching_attacks, completion_attacks, graph1_copy
from probabilistic_attack import similarity_based_completion



if __name__ == "__main__":

    number_of_experiments = 5

    dataset_name = "polblogs"

    dataset = Graph.from_npy(f"datasets/{dataset_name}.adj.npy")

    graph1_props = [0.1]
    common_props = [0]

    for expe in range(number_of_experiments):

        for graph1_prop, common_prop in product(graph1_props, common_props):

            graph1, graph2 = dataset.split_dataset(common_prop=common_prop, graph1_prop=graph1_prop)

            np.save(f"splitted_datasets/{dataset_name}_{graph1_prop}_{common_prop}_graph1.npy", graph1.adj_matrix)
            np.save(f"splitted_datasets/{dataset_name}_{graph1_prop}_{common_prop}_graph2.npy", graph2.adj_matrix)

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

            log_graph_stats(graph1_prop, common_prop, expe, reconstructed_graph, 0, 0,f"logs_mixed/{dataset_name}.csv", dataset)


            iteration_count = 0
            while len(reconstructed_graph.unknown_edges()) > 0:
                
                carry_on = True

                while carry_on:
                    print("Iteration ", iteration_count)
                    start = time.time()
                    reconstructed_graph, number_modifs1 = matching_attacks(reconstructed_graph, A)
                    reconstructed_graph, number_modifs2 = completion_attacks(reconstructed_graph, A)
                    end = time.time()
                    log_graph_stats(graph1_prop, common_prop, expe, reconstructed_graph, iteration_count, end-start,f"logs_mixed/{dataset_name}.csv", dataset)

                    if number_modifs1 + number_modifs2 == 0:
                        carry_on = False
                    
                    iteration_count += 1
        
                display_reconstruction_metrics(reconstructed_graph, dataset)

                reconstructed_graph = similarity_based_completion(reconstructed_graph, 1, method="common_neighbors")

                display_reconstruction_metrics(reconstructed_graph, dataset)

            project_path = "/Users/aubinbirba/Documents/PhD/Graph attack/code/attack"
            np.save(f"{project_path}/mixed/{dataset_name}_{graph1_prop}_{common_prop}.npy", reconstructed_graph.adj_matrix)

            reconstructed_graph.fix_edges()

            accuracy = reconstruction_accuracy(reconstructed_graph, dataset)
            distance = frobenius_distance(reconstructed_graph, dataset)
            rae_stat = rae(reconstructed_graph, dataset)
            edge_accuracy = edge_identification_accuracy(reconstructed_graph, dataset)

            print("Edge accuracy", edge_accuracy)

 

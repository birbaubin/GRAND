from helpers.helpers import *
from helpers.Graph import Graph
from itertools import product 



def matching_attacks(reconstructed_graph, A):
    modifs = 0
    print("\n######## Matching attacks #######")
    print("Reconstruction statistics before :")
    display_graph_stats(reconstructed_graph)

    for i in tqdm(range(len(reconstructed_graph.nodes)), desc="Matching attacks"):
        for j in range(i, len(reconstructed_graph.nodes)):
            if A[i, j] == len(reconstructed_graph.common_neighbors((i, j))):
                for node in reconstructed_graph.neighbors(i):
                    if reconstructed_graph.get_edge_label((j, node)) == 2:
                        reconstructed_graph.remove_edge((j, node))
                        modifs += 1
                for node in reconstructed_graph.neighbors(j):
                    if reconstructed_graph.get_edge_label((i, node)) == 2:
                        reconstructed_graph.remove_edge((i, node))
                        modifs += 1

    print("Reconstruction statistics after")
    display_graph_stats(reconstructed_graph)

    return reconstructed_graph, modifs


def completion_attacks(reconstructed_graph, A):
    modifs = 0
    print("\n######## Completion attacks #######")
    print("Reconstruction statistics before :")
    display_graph_stats(reconstructed_graph)

    for i in tqdm(range(len(reconstructed_graph.nodes)), desc="Completion attacks"):
        for j in range(i, len(reconstructed_graph.nodes)):
            unknowed_edges_i = np.where(reconstructed_graph.adj_matrix[i] == 2)[0]
            unknowed_edges_j = np.where(reconstructed_graph.adj_matrix[j] == 2)[0]

            if len(reconstructed_graph.neighbors(i)) == A[i, j] - len(unknowed_edges_i):
                for k in unknowed_edges_i:
                    reconstructed_graph.add_edge((i, k))
                    reconstructed_graph.add_edge((j, k))
                    modifs += 1

            if len(reconstructed_graph.neighbors(j)) == A[i, j] - len(unknowed_edges_j):
                for k in unknowed_edges_j:
                    reconstructed_graph.add_edge((j, k))
                    reconstructed_graph.add_edge((i, k))
                    modifs += 1

    print("Reconstruction statistics after :")
    display_graph_stats(reconstructed_graph)

    return reconstructed_graph, modifs


def graph1_copy(graph1):
    print("\n####### Copy of edges from graph1 #######")
    print("Reconstruction statistics before copy")

    reconstructed_graph = Graph(graph1.nodes, with_fixed_edges=False)
    display_graph_stats(reconstructed_graph)
    reconstructed_graph.add_edges_from(graph1.edges())

    print("Reconstruction statistics after copy")
    display_graph_stats(reconstructed_graph)
    return reconstructed_graph



def petersen_style_graphs(n):

    size = int(n/2)
    adj_matrix_1 = np.zeros((n, n))
    for i in range(size):
        for j in range(i+1, size):
            if j == i+1:
                adj_matrix_1[i][j] = 1
                adj_matrix_1[j][i] = 1
        if i == size-1:
            adj_matrix_1[i][0] = 1
            adj_matrix_1[0][i] = 1

    adj_matrix_2 = np.zeros((n, n))
    for i in range(size, n):
        for j in range(i+1, n):
            if j == i+2 or j == i+3:
                adj_matrix_2[i][j] = 1
                adj_matrix_2[j][i] = 1


    graph1 = Graph(list(range(n)), with_fixed_edges=True)
    graph2 = Graph(list(range(n)), with_fixed_edges=True)
    graph1.adj_matrix = adj_matrix_1
    graph2.adj_matrix = adj_matrix_2


    return graph1, graph2



############# Main #############
if __name__ == "__main__":


    """Running the pipeline
    """


    # graph1 = random_graph(5, 0.5)
    # graph2 = random_graph(5, 0.5)

    # A = np.dot(graph1.adj_matrix, graph2.adj_matrix)

    # print("Graph1")
    # print(graph1.adj_matrix)

    # print("Graph2")
    # print(graph2.adj_matrix)

    # print("Common neighbors matrix")
    # print(A)
    
    number_of_experiments = 5

    dataset_name = "acm"
    graph1_props = [0, 0.1, 0.25, 0.5, 0.75]
    common_props = [0]

    dataset = Graph.from_npy(f"datasets/{dataset_name}.adj.npy")

    for expe in range(number_of_experiments):

        for graph1_prop, common_prop in product(graph1_props, common_props):

            # graph1, graph2 = petersen_style_graphs(500)
            graph1, graph2 = dataset.split_dataset(common_prop=common_prop, graph1_prop=graph1_prop)
            # dataset = Graph.from_nodes_and_edges(graph1.nodes, graph1.edges() + graph2.edges())

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

            carry_on = True
            i = 0
            while carry_on:
                print("Iteration ", i)
                start = time.time()
                reconstructed_graph, number_modifs1 = matching_attacks(reconstructed_graph, A)
                reconstructed_graph, number_modifs2 = completion_attacks(reconstructed_graph, A)
                end = time.time()
                log_graph_stats(graph1_prop, common_prop, expe, reconstructed_graph, i, end-start, 
                f"logs/{dataset_name}.csv")

                if number_modifs1 + number_modifs2 == 0:
                    carry_on = False
                i += 1

            project_path = "/Users/aubinbirba/Documents/PhD/Graph attack/code/attack"
            np.save(f"{project_path}/rec_deterministic/{dataset_name}_{graph1_prop}_{common_prop}.npy", reconstructed_graph.adj_matrix)

            reconstructed_graph.fix_edges()

            accuracy = reconstruction_accuracy(reconstructed_graph, dataset)
            distance = frobenius_distance(reconstructed_graph, dataset)
            rae_stat = rae(reconstructed_graph, dataset)
            edge_accuracy = edge_identification_accuracy(reconstructed_graph, dataset)

            print("Reconstruction accuracy", accuracy)

            with open(f"logs/{dataset_name}_acc.csv", "a+") as f:
                f.write(f"{expe},{graph1_prop},{common_prop},{accuracy},{distance},{rae_stat},{edge_accuracy}\n")
 




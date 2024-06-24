from helpers.helpers import *
from helpers.Graph import Graph
from itertools import product 



def matching_attacks(reconstructed_graph, A):
    modifs = 0
    
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

    return reconstructed_graph, modifs


def completion_attacks(reconstructed_graph, A):
    modifs = 0

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

    return reconstructed_graph, modifs


def graph1_copy(graph1):

    reconstructed_graph = Graph(graph1.nodes, with_fixed_edges=False)
    display_graph_stats(reconstructed_graph)
    reconstructed_graph.add_edges_from(graph1.edges())
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

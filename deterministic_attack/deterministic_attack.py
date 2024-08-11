from helpers.helpers import *
from helpers.Graph import Graph
from itertools import product, combinations



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
            unknowed_edges_i = np.where(reconstructed_graph.y[i] == 2)[0]
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

def hub_and_isolated_node_nattack(reconstructed_graph, A, hub_threshold=0.5):
    number_modifs = 0
    for i in tqdm(range(A.shape[0]), desc="Hub attack"):
        if A[i, i] >= hub_threshold*len(reconstructed_graph.nodes):
            for node in range(A.shape[0]):
                if reconstructed_graph.adj_matrix[i, j] == 2:
                    reconstructed_graph.add_edge((i, j))
                    number_modifs += 1

        if A[i, i] == 0:
            for j in range(A.shape[0]):
                if reconstructed_graph.adj_matrix[i, j] == 2:
                    reconstructed_graph.add_edge((i, j))
                    number_modifs += 1

    return reconstructed_graph, number_modifs


def degree_one_attack(reconstructed_graph, A):
    number_modifs = 0
    for i in tqdm(range(A.shape[0]), desc="Degree one attack"):
        if A[i, i] == 1:
            degree = np.sum(A[i]) 
            candidates = np.where(np.diag(A) == degree)[0] 
            if len(candidates) == 1 and reconstructed_graph.adj_matrix[i, candidates[0]] == 2:
                    reconstructed_graph.add_edge((i, candidates[0]))
                    number_modifs += 1

    return reconstructed_graph, number_modifs


def degree_attack(reconstructed_graph, A, degrees):
    number_modifs = 0
    for i in tqdm(range(A.shape[0]), desc="Degree attack"):
        if A[i, i] not in degrees:
            continue
        degree = np.sum(A[i]) 
        candidate = None
        possibilities = np.where(np.diag(A) <= degree - A[i, i] + 1)[0]
        for comb in combinations(possibilities, A[i, i]):
            sum_of_degrees = 0

            for k in comb:
                sum_of_degrees += A[k, k]

            if sum_of_degrees == degree:
                if candidate == None:
                    candidate = comb
                else:
                    candidate = None
                    break
        
        if candidate != None:
            for j in candidate:
                if reconstructed_graph.adj_matrix[i, j] == 2:
                    reconstructed_graph.add_edge((i, j))
                    number_modifs += 1

    return reconstructed_graph, number_modifs



def triangle_attack(reconstructed_graph, A):
    number_modifs = 0
    for (u, v) in tqdm(reconstructed_graph.edges(), desc="Triangle attack"):

        g2_u = np.where(A[u] > 0)[0]
        g2_u = np.setdiff1d(g2_u, u)
        g2_v = np.where(A[v] > 0)[0]
        g2_v = np.setdiff1d(g2_v, v)

        candidates = np.intersect1d(g2_u, g2_v)

        if len(candidates) == A[u, v]:
            for w in candidates:
                if reconstructed_graph.adj_matrix[u, w] == 2:
                    reconstructed_graph.add_edge((u, w))
                    number_modifs += 1
                if not reconstructed_graph.adj_matrix[v, w] == 2:
                    reconstructed_graph.add_edge((v, w))
                    number_modifs += 1


    return reconstructed_graph, number_modifs

def rectangle_attack(reconstructed_graph, A, degrees=[1, 2, 3, 4, 5]):
    additions = 0

    for k in tqdm(degrees):

        nodes_of_degree_k  = np.where(np.diag(A) == k)[0]
        candidates = [ node for node in nodes_of_degree_k if len(reconstructed_graph.neighbors(node)) == k]

        for node in candidates:
            neighbors = reconstructed_graph.neighbors(node)
            for graph_node in reconstructed_graph.nodes:
                if A[node, graph_node] == k:
                    for neighbor in neighbors:
                        if reconstructed_graph.adj_matrix[neighbor, graph_node] == 2:
                            reconstructed_graph.add_edge((neighbor, graph_node))
                            additions += 1
 

    return reconstructed_graph, additions




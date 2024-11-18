from helpers import *
from Graph_new import Graph
from itertools import product, combinations
from tqdm import tqdm
import numpy as np
from prettytable import PrettyTable
import time
from joblib import Parallel, delayed
from multiprocessing import Pool
import networkx as nx

def process_node_pair(args):
    i, reconstructed_graph, A, nodes = args
    local_modifications = []

    neighbors_i = reconstructed_graph.neighbors(i)
    for j in range(i, len(nodes)):
        neighbors_j = reconstructed_graph.neighbors(j)
        common_neighbors = np.intersect1d(neighbors_i, neighbors_j)
        if A[i, j] == len(common_neighbors):
            for node in neighbors_i:
                if reconstructed_graph.get_edge_label((j, node)) == 2:
                    local_modifications.append(("remove", j, node))
            for node in neighbors_j:
                if reconstructed_graph.get_edge_label((i, node)) == 2:
                    local_modifications.append(("remove", i, node))

    return local_modifications

def process_candidate(args):
    node, Gstar_nodes, reconstructed_graph_adj_matrix, A, neighbors = args
    local_modifications = []
    node_neighbors = neighbors[node]

    for graph_node in range(A.shape[0]):
        graph_node_neighbors = neighbors[graph_node]
        number_missing_edges = A[graph_node, graph_node] - len(graph_node_neighbors)

        common_neighbors = np.intersect1d(node_neighbors, graph_node_neighbors)
        number_missing_common_neighbors = A[node, graph_node] - len(common_neighbors)

        if number_missing_edges == number_missing_common_neighbors:
            adds = np.setdiff1d(Gstar_nodes, node_neighbors)
            for i in adds:
                if reconstructed_graph_adj_matrix[i, graph_node] == 2:
                    local_modifications.append((i, graph_node))
    return local_modifications


class DeterministicAttack:
    """
    Class to perform deterministic attacks on a graph
    """

    def __init__(self, Ga, A, graph1_prop=0.0, dataset_name=None, log=False, expe_number=None):
        """
        Constructor of the class
        :param graph1: Graph object
        :param A: Adjacency matrix of the graph
        :param graph1_prop: Proportion of the graph1 in the dataset
        :param dataset_name: Name of the dataset
        :param log: Boolean to log the results
        :param expe_number: Number of the experiment
        """

        self.Gstar = Ga.copy()
        self.A = A
        self.modifications = 0
        self.dataset_name = dataset_name
        self.log = log
        self.graph1_prop = graph1_prop
        self.expe_number = expe_number

        if self.log:
            if self.dataset_name == None or self.graph1_prop == None:
                print("Error : dataset_name and/or size of known graph is not defined")
                exit()
            else:
                print("#### Deterministic attacks ####")

                self.log_file = open(f"logs/deterministics/{self.dataset_name}.csv", "a")



    def matching_attacks(self):
        modifs = 0
        reconstructed_graph = self.Gstar.copy()
        A = self.A


        for i in tqdm(range(len(reconstructed_graph.nodes)), desc="Matching attacks"):
            neighbors_i = reconstructed_graph.neighbors(i)
            for j in range(i, len(reconstructed_graph.nodes)):
                neighbors_j = reconstructed_graph.neighbors(j)
                common_neighbors = neighbors_i & neighbors_j

                if A[i, j] == len(common_neighbors):
                    for node in neighbors_i.copy():
                        if reconstructed_graph.does_not_know_edge((j, node)):
                            reconstructed_graph.remove_edge((j, node))
                            modifs += 1
                    for node in neighbors_j.copy():
                        if reconstructed_graph.does_not_know_edge((i, node)):
                            reconstructed_graph.remove_edge((i, node))
                            modifs += 1


        print(f"Matching attack: {modifs} modifications")

        if self.log:
            self.log_file.write(f"matching,{self.expe_number},{self.graph1_prop},{modifs}\n")

        self.Gstar = reconstructed_graph
        self.modifications+=modifs




    def completion_attacks(self):
        modifs = 0
        reconstructed_graph = self.Gstar.copy()
        A = self.A

        for i in tqdm(range(len(reconstructed_graph.nodes)), desc="Completion attacks"):
            neighbors_i = reconstructed_graph.neighbors(i)
            for j in range(i, len(reconstructed_graph.nodes)):
                neighbors_j = reconstructed_graph.neighbors(j)
                unknowed_edges_i = reconstructed_graph.unknown_list[i]
                unknowed_edges_j = reconstructed_graph.unknown_list[j]

                if len(neighbors_i) == A[i, j] - len(unknowed_edges_i):
                    for k in unknowed_edges_i.copy():
                        reconstructed_graph.add_edge((i, k))
                        reconstructed_graph.add_edge((j, k))
                        modifs += 1


                if len(neighbors_j) == A[i, j] - len(unknowed_edges_j):
                    for k in unknowed_edges_j.copy():
                        reconstructed_graph.add_edge((j, k))
                        reconstructed_graph.add_edge((i, k))
                        modifs += 1

        print(f"Completion attack: {modifs} modifications")
        if self.log:
            self.log_file.write(f"completion,{self.expe_number},{self.graph1_prop},{modifs}\n")

        self.Gstar = reconstructed_graph
        self.modifications += modifs


    def degree_attack(self, degrees=[1, 2]):
        modifs = 0
        reconstructed_graph = self.Gstar.copy()
        A = self.A
        degree_sequence = np.diag(A)


        for i in tqdm(range(A.shape[0]), desc="Degree attack"):

            if A[i, i] not in degrees:
                continue
            
            degree = np.sum(A[i])
            candidate = None
            possibilities = np.where((degree_sequence <= degree - A[i, i] + 1))[0]            

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
                    if reconstructed_graph.does_not_know_edge((i, j)):
                        reconstructed_graph.add_edge((i, j))
                        modifs += 1


        print(f"Degree attack: {modifs} modifications")
        if self.log:
            self.log_file.write(f"degree,{self.expe_number},{self.graph1_prop},{modifs}\n")

        self.Gstar =  reconstructed_graph
        self.modifications += modifs


    def degree_more(self):
        modifs = 0
        reconstructed_graph = self.Gstar.copy()
        for node in tqdm(self.Gstar.nodes, desc="Degree more attack"):
            sum_row_node = np.sum(self.A[node])
            for candidate in self.Gstar.nodes:
                degree_candidate = self.A[candidate, candidate]
                if sum_row_node < degree_candidate and self.Gstar.does_not_know_edge((candidate, node)):
                    reconstructed_graph.remove_edge((candidate, node))
                    modifs += 1

                    

        self.modifications += modifs
        self.Gstar = reconstructed_graph
        print(f"Degree more attack: {modifs} modifications")



    def triangle_attack(self):
        modifs = 0
        reconstructed_graph = self.Gstar.copy()
        A = self.A

        edges = reconstructed_graph.edges()
        for (u, v) in tqdm(edges, desc="Triangle attack"):

            candidates = [node for node in reconstructed_graph.nodes 
                if self.A[u, node] > 0 and 
                self.A[v, node] > 0 and 
                node != u and node != v and
                A[node, node] >= 2 and
                sum(self.A[node]) >= self.A[u, u] + self.A[v, v]
                ]


            if len(candidates) == A[u, v]:
                for w in candidates:
                    if reconstructed_graph.does_not_know_edge((u, w)):
                        reconstructed_graph.add_edge((u, w))
                        modifs += 1
                    if reconstructed_graph.does_not_know_edge((v, w)):
                        reconstructed_graph.add_edge((v, w))
                        modifs += 1

        print(f"Triangle attack: {modifs} modifications")
        if self.log:
            self.log_file.write(f"triangle,{self.expe_number},{self.graph1_prop},{modifs}\n")

        self.Gstar = reconstructed_graph
        self.modifications += modifs

    


    def rectangle_attack(self, degrees=[1, 2, 3, 4, 5]):
        modifs = 0
        reconstructed_graph = self.Gstar.copy()
        A = self.A

        for k in tqdm(degrees, desc="Rectangle attack:"):

            nodes_of_degree_k  = np.where(np.diag(A) == k)[0]
            candidates = [ node for node in nodes_of_degree_k if len(reconstructed_graph.neighbors(node)) == k]

            for node in candidates:
                neighbors = reconstructed_graph.neighbors(node)
                for graph_node in reconstructed_graph.nodes:
                    if A[node, graph_node] == k:
                        for neighbor in neighbors:
                            if reconstructed_graph.does_not_know_edge((neighbor, graph_node)):
                                reconstructed_graph.add_edge((neighbor, graph_node))
                                modifs += 1


        print(f"Rectangle attack: {modifs} modifications")

        if self.log:
            self.log_file.write(f"rectangle,{self.expe_number},{self.graph1_prop},{modifs}\n")

        self.Gstar = reconstructed_graph
        self.modifications += modifs


    def rectange_attack_more(self):

        reconstructed_graph = self.Gstar.copy()
        candidates = set([ node for node in self.Gstar.nodes if len(reconstructed_graph.neighbors(node)) == self.A[node, node]])
        modifs = 0

        for node in tqdm(list(candidates), desc="Rectangle attack more"):
            node_neighbors = reconstructed_graph.neighbors(node)
            for graph_node in list(reconstructed_graph.nodes):
                graph_node_neighbors = reconstructed_graph.neighbors(graph_node)
                number_missing_edges = self.A[graph_node, graph_node] - len(graph_node_neighbors)

                common_neighbors = node_neighbors & graph_node_neighbors
                number_missing_common_neighbors = self.A[node, graph_node] - len(common_neighbors)

                if number_missing_edges == number_missing_common_neighbors:
                    adds = self.Gstar.nodes - node_neighbors
                    for i in adds:
                        if reconstructed_graph.does_not_know_edge((i, graph_node)):
                            reconstructed_graph.remove_edge((i, graph_node))
                            modifs += 1

                    
        
        self.Gstar = reconstructed_graph
        self.modifications += modifs
        print(f"Rectangle attack more: {modifs} modifications")




    def run(self, run_matching=True, run_completion=True, run_degree=True, run_rectangle=True, run_triangle=True, run_degree2=True):

        iteration_deterministic = 0
        iteration_probabilistic = 0
        continue_deterministic = True

        degrees = np.diag(self.A)

        if run_degree:
            self.degree_attack(degrees=[1, 2])
            # print(ROC_stats(self.Gstar, self.G))
            self.degree_more()
            # print(ROC_stats(self.Gstar, self.G))

        while continue_deterministic:
            old_modfications = self.modifications
            
            start = time.time()
            if run_matching:
                self.matching_attacks()
                # print(ROC_stats(self.Gstar, self.G))
            if run_completion:
                self.completion_attacks()
                # print(ROC_stats(self.Gstar, self.G))
            if run_rectangle:
                self.rectangle_attack(degrees)
                # print(ROC_stats(self.Gstar, self.G))
                self.rectange_attack_more()
                # print(ROC_stats(self.Gstar, self.G))
            if run_triangle:
                self.triangle_attack()

            stats = self.Gstar.stats()
            print("0s : ", stats[0], "1s : ", stats[1], "?s : ", stats[2])

            end = time.time()

            if self.modifications == old_modfications:
                continue_deterministic = False

            iteration_deterministic += 1


    def get_Gstar(self):
        return self.Gstar

from helpers import *
from Graph import Graph
from itertools import product, combinations
from tqdm import tqdm
import numpy as np
from prettytable import PrettyTable
import time
from joblib import Parallel, delayed



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
                # t = PrettyTable(['Parameter', 'Value'])
                # t.add_row(['Dataset name', self.dataset_name])
                # t.add_row(['Known graph proportion', self.graph1_prop])
                # t.add_row(['Experiment number', self.expe_number])
                # t.add_row(['Log', self.log])
                # print(t)

                self.log_file = open(f"logs/deterministics/{self.dataset_name}.csv", "a")


    
    def matching_attacks(self):
        modifs = 0
        reconstructed_graph = self.Gstar.copy()
        A = self.A

        for i in tqdm(range(len(reconstructed_graph.nodes)), desc="Matching attacks"):
            neighbors_i = reconstructed_graph.neighbors(i)
            for j in range(i, len(reconstructed_graph.nodes)):
                neighbors_j = reconstructed_graph.neighbors(j)
                common_neighbors = np.intersect1d(neighbors_i, neighbors_j) 
                if A[i, j] == len(common_neighbors):
                    for node in neighbors_i.copy():
                        if reconstructed_graph.get_edge_label((j, node)) == 2:
                            reconstructed_graph.remove_edge((j, node))
                    for node in neighbors_j.copy():
                        if reconstructed_graph.get_edge_label((i, node)) == 2:
                            reconstructed_graph.remove_edge((i, node))

        modifs = np.sum(reconstructed_graph.adj_matrix != self.Gstar.adj_matrix)

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
                unknowed_edges_i = np.where(reconstructed_graph.adj_matrix[i] == 2)[0]
                unknowed_edges_j = np.where(reconstructed_graph.adj_matrix[j] == 2)[0]

                if len(neighbors_i) == A[i, j] - len(unknowed_edges_i):
                    for k in unknowed_edges_i:
                        reconstructed_graph.add_edge((i, k))
                        reconstructed_graph.add_edge((j, k))

                if len(neighbors_j) == A[i, j] - len(unknowed_edges_j):
                    for k in unknowed_edges_j:
                        reconstructed_graph.add_edge((j, k))
                        reconstructed_graph.add_edge((i, k))
                        
        modifs = np.sum(reconstructed_graph.adj_matrix != self.Gstar.adj_matrix)

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
            possibilities = np.where((2 <= degree_sequence) & (degree_sequence <= degree - A[i, i] + 1))[0]

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


        modifs = len(np.where(reconstructed_graph.adj_matrix != self.Gstar.adj_matrix)[0])
        if self.log:
            self.log_file.write(f"degree,{self.expe_number},{self.graph1_prop},{modifs}\n")
        
        self.Gstar =  reconstructed_graph
        self.modifications += modifs


    def triangle_attack(self):
        modifs = 0
        reconstructed_graph = self.Gstar.copy()
        A = self.A

        edges = reconstructed_graph.edges()
        for (u, v) in tqdm(edges, desc="Triangle attack"):

            g2_u = np.where(A[u] > 0)[0]
            g2_u = np.setdiff1d(g2_u, u)
            g2_v = np.where(A[v] > 0)[0]
            g2_v = np.setdiff1d(g2_v, v)

            candidates = np.intersect1d(g2_u, g2_v)

            if len(candidates) == A[u, v]:
                for w in candidates:
                    if reconstructed_graph.adj_matrix[u, w] == 2:
                        reconstructed_graph.add_edge((u, w))
                    if reconstructed_graph.adj_matrix[v, w] == 2:
                        reconstructed_graph.add_edge((v, w))

        modifs = len(np.where(reconstructed_graph.adj_matrix != self.Gstar.adj_matrix)[0])
        if self.log:
            self.log_file.write(f"triangle,{self.expe_number},{self.graph1_prop},{modifs}\n")


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
                            if reconstructed_graph.adj_matrix[neighbor, graph_node] == 2:
                                # print(f"Adding edge between {neighbor} and {graph_node}")
                                reconstructed_graph.add_edge((neighbor, graph_node))


        modifs = len(np.where(reconstructed_graph.adj_matrix != self.Gstar.adj_matrix)[0])

        if self.log:
            self.log_file.write(f"rectangle,{self.expe_number},{self.graph1_prop},{modifs}\n")

        self.Gstar = reconstructed_graph
        self.modifications += modifs


    def rectange_attack_more(self):

        reconstructed_graph = self.Gstar.copy()
        candidates = [ node for node in self.Gstar.nodes if len(reconstructed_graph.neighbors(node)) == self.A[node, node]]

        for node in tqdm(candidates, desc="Rectangle attack more"):
            node_neighbors = reconstructed_graph.neighbors(node)
            for graph_node in range(self.A.shape[0]):
                graph_node_neighbors = reconstructed_graph.neighbors(graph_node)
                number_missing_edges = self.A[graph_node, graph_node] - len(graph_node_neighbors)

                common_neighbors = np.intersect1d(node_neighbors, graph_node_neighbors)
                number_missing_common_neighbors = self.A[node, graph_node] - len(common_neighbors)

                if number_missing_edges == number_missing_common_neighbors:
                    for i in list(set(self.Gstar.nodes) - set(node_neighbors)):
                        if reconstructed_graph.adj_matrix[i, graph_node] == 2:
                            reconstructed_graph.remove_edge((i, graph_node))
                        if reconstructed_graph.adj_matrix[i, graph_node] == 2:
                            reconstructed_graph.remove_edge((i, graph_node))

        modifs = len(np.where(reconstructed_graph.adj_matrix != self.Gstar.adj_matrix)[0])
        self.Gstar = reconstructed_graph

        print(f"Rectangle attack more: {modifs} modifications")


    def degree_more(self):
        modifs = 0

        for node in tqdm(self.Gstar.nodes, desc="Degree more attack"):
            sum_row_node = np.sum(self.A[node])
            for candidate in self.Gstar.nodes:
                degree_candidate = self.A[candidate, candidate]
                if sum_row_node < degree_candidate and self.Gstar.get_edge_label((candidate, node)) == 2:
                    self.Gstar.remove_edge((candidate, node))
                    modifs+=2
       


    def run(self, run_matching=True, run_completion=True, run_degree=True, run_rectangle=True, run_triangle=True, run_degree2=True):
        
        iteration_deterministic = 0
        iteration_probabilistic = 0
        continue_deterministic = True

        degrees = np.unique(np.diag(self.A))

        if run_degree:
            self.degree_attack(degrees=[1, 2])
            self.degree_more()
            # self.degree_prime()

        while continue_deterministic:
            old_modfications = self.modifications

            start = time.time()
            if run_matching:
                self.matching_attacks()
            if run_completion:
                self.completion_attacks()
            if run_rectangle:
                self.rectangle_attack(degrees)
                self.rectange_attack_more()
            if run_triangle:
                self.triangle_attack()
            
            end = time.time()

            # display_graph_stats(self.Gstar)

            if self.modifications == old_modfications:
                continue_deterministic = False
            
            iteration_deterministic += 1


    def get_Gstar(self):
        return self.Gstar

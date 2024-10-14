from helpers import *
from Graph import Graph
from itertools import product, combinations


class DeterministicAttack:
    """
    Class to perform deterministic attacks on a graph
    """

    def __init__(self, graph1, A, graph1_prop=0.0, dataset_name=None, log=False):
        self.reconstructed_graph = Graph(graph1.nodes, with_fixed_edges=False)
        self.reconstructed_graph.add_edges_from(graph1.edges())
        self.A = A
        self.modifications = 0
        self.dataset_name = dataset_name
        self.log = log
        self.graph1_prop = graph1_prop

        if self.log:
            if self.dataset_name == None or self.graph1_prop == None:
                print("Error : dataset_name and/or size of known graph is not defined")
                exit()
            else:
                print(f"Deterministic attacks on {self.dataset_name}")
                self.log_file = open(f"logs/deterministics/{self.dataset_name}.csv", "a")


    def matching_attacks(self):
        modifs = 0
        reconstructed_graph = self.reconstructed_graph.copy()
        A = self.A

        for i in tqdm(range(len(reconstructed_graph.nodes)), desc="Matching attacks"):
            for j in range(i, len(reconstructed_graph.nodes)):
                if A[i, j] == len(reconstructed_graph.common_neighbors((i, j))):
                    for node in reconstructed_graph.neighbors(i):
                        if reconstructed_graph.get_edge_label((j, node)) == 2:
                            reconstructed_graph.remove_edge((j, node))

                    for node in reconstructed_graph.neighbors(j):
                        if reconstructed_graph.get_edge_label((i, node)) == 2:
                            reconstructed_graph.remove_edge((i, node))

        modifs = len(np.where(reconstructed_graph.adj_matrix != self.reconstructed_graph.adj_matrix)[0])

        if self.log:
            self.log_file.write(f"matching,{self.graph1_prop},{modifs}\n")

        self.reconstructed_graph = reconstructed_graph
        self.modifications+=modifs


    def completion_attacks(self):
        modifs = 0
        reconstructed_graph = self.reconstructed_graph.copy()
        A = self.A

        for i in tqdm(range(len(reconstructed_graph.nodes)), desc="Completion attacks"):
            for j in range(i, len(reconstructed_graph.nodes)):
                unknowed_edges_i = np.where(reconstructed_graph.adj_matrix[i] == 2)[0]
                unknowed_edges_j = np.where(reconstructed_graph.adj_matrix[j] == 2)[0]

                if len(reconstructed_graph.neighbors(i)) == A[i, j] - len(unknowed_edges_i):
                    for k in unknowed_edges_i:
                        reconstructed_graph.add_edge((i, k))
                        reconstructed_graph.add_edge((j, k))

                if len(reconstructed_graph.neighbors(j)) == A[i, j] - len(unknowed_edges_j):
                    for k in unknowed_edges_j:
                        reconstructed_graph.add_edge((j, k))
                        reconstructed_graph.add_edge((i, k))
                        
        modifs = len(np.where(reconstructed_graph.adj_matrix != self.reconstructed_graph.adj_matrix)[0])

        if self.log:
            self.log_file.write(f"completion,{self.graph1_prop},{modifs}\n")

        self.reconstructed_graph = reconstructed_graph
        self.modifications += modifs


    def degree_attack(self, degrees=[1, 2]):
        modifs = 0
        reconstructed_graph = self.reconstructed_graph.copy()
        A = self.A

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


        modifs = len(np.where(reconstructed_graph.adj_matrix != self.reconstructed_graph.adj_matrix)[0])
        if self.log:
            self.log_file.write(f"degree,{self.graph1_prop},{modifs}\n")
        
        self.reconstructed_graph =  reconstructed_graph
        self.modifications += modifs



    def triangle_attack(self):
        modifs = 0
        reconstructed_graph = self.reconstructed_graph.copy()
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

        modifs = len(np.where(reconstructed_graph.adj_matrix != self.reconstructed_graph.adj_matrix)[0])
        if self.log:
            self.log_file.write(f"triangle,{self.graph1_prop},{modifs}\n")


    def rectangle_attack(self, degrees=[1, 2, 3, 4, 5]):
        modifs = 0
        reconstructed_graph = self.reconstructed_graph.copy()
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
                                reconstructed_graph.add_edge((neighbor, graph_node))


        modifs = len(np.where(reconstructed_graph.adj_matrix != self.reconstructed_graph.adj_matrix)[0])

        if self.log:
            self.log_file.write(f"rectangle,{self.graph1_prop},{modifs}\n")

        self.reconstructed_graph = reconstructed_graph
        self.modifications += modifs


    def run(self, run_matching=True, run_completion=True, run_degree=True, run_rectangle=True, run_triangle=True):
        
        iteration_deterministic = 0
        iteration_probabilistic = 0
        continue_deterministic = True

        if run_degree:
            self.degree_attack()

        while continue_deterministic:
            old_modfications = self.modifications

            start = time.time()
            if run_matching:
                self.matching_attacks()
            if run_completion:
                self.completion_attacks()
            if run_rectangle:
                self.rectangle_attack(degrees=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            if run_triangle:
                self.triangle_attack()
            
            end = time.time()

            display_graph_stats(self.reconstructed_graph)

            if self.modifications == old_modfications:
                continue_deterministic = False
            
            iteration_deterministic += 1



    def get_reconstructed_graph(self):
        return self.reconstructed_graph

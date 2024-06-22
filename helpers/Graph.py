import numpy as np
import math
import pandas as pd

class Graph(object):

    def __init__(self, nodes, with_fixed_edges=False):
        self.nodes = nodes
        self.size = len(nodes)
        if with_fixed_edges:
            self.adj_matrix = np.zeros((self.size, self.size), dtype=int)
        else:
            self.adj_matrix = np.full((self.size, self.size), 2, dtype=int)
            for i in range(self.size):
                self.adj_matrix[i][i] = 0

    def add_edge(self, edge):
        n1, n2 = edge
        self.adj_matrix[n1][n2] = 1
        self.adj_matrix[n2][n1] = 1

    def get_edge_label(self, edge):
        n1, n2 = edge
        return self.adj_matrix[n1][n2]

    def __len__(self):
        return len(self.graph.edges())

    def add_edges_from(self, edges):
        for edge in edges:
            n1, n2 = edge
            self.adj_matrix[n1][n2] = 1
            self.adj_matrix[n2][n1] = 1

    def remove_edge(self, edge):
        n1, n2 = edge
        self.adj_matrix[n1][n2] = 0
        self.adj_matrix[n2][n1] = 0

    def neighbors(self, n1):
        return np.where(self.adj_matrix[n1] == 1)[0]


    def common_neighbors_index(self, edge=None):
        if edge:
            n1, n2 = edge
            return len(set(self.neighbors(n1)).intersection(set(self.neighbors(n2))))
        else:
            common_neighbors = np.zeros((self.size, self.size))
            for i in range(self.size):
                for j in range(i+1, self.size):
                    common_neighbors[i][j] = len(set(self.neighbors(i)).intersection(set(self.neighbors(j))))
            return common_neighbors


    def jaccard_index(self, edge=None):

        if edge:
            n1, n2 = edge
            common_neighbors = self.common_neighbors_index(edge)
            union_neighbors = len(set(self.neighbors(n1)).union(set(self.neighbors(n2))))
            if union_neighbors == 0:
                return 0
            return common_neighbors / union_neighbors
        else:
            jaccard = np.zeros((self.size, self.size))
            for i in range(self.size):
                for j in range(i+1, self.size):
                    common_neighbors = self.common_neighbors_index((i, j))
                    union_neighbors = len(set(self.neighbors(i)).union(set(self.neighbors(j))))
                    if union_neighbors == 0:
                        jaccard[i][j] = 0
                    else:
                        jaccard[i][j] = common_neighbors / union_neighbors
            return jaccard


    def preferential_attachment_index(self, edge=None):
        if edge:
            n1, n2 = edge
            return len(self.neighbors(n1)) * len(self.neighbors(n2))
        else:
            pref_attach = np.zeros((self.size, self.size))
            for i in range(self.size):
                for j in range(i+1, self.size):
                    pref_attach[i][j] = len(self.neighbors(i)) * len(self.neighbors(j))
            return pref_attach

    def adamic_adar_index(self, edge=None):
        if edge:
            n1, n2 = edge
            common_neighbors = set(self.neighbors(n1)).intersection(set(self.neighbors(n2)))
            return sum([1 / math.log(len(self.neighbors(k))) for k in common_neighbors])
        adamic_adar = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(i+1, self.size):
                common_neighbors = set(self.neighbors(i)).intersection(set(self.neighbors(j)))
                adamic_adar[i][j] = sum([1 / math.log(len(self.neighbors(k))) for k in common_neighbors])
        return adamic_adar

    def resource_allocation_index(self, edge=None):
        if edge:
            n1, n2 = edge
            common_neighbors = set(self.neighbors(n1)).intersection(set(self.neighbors(n2)))
            return sum([1 / len(self.neighbors(k)) for k in common_neighbors])

        else:
            resource_allocation = np.zeros((self.size, self.size))
            for i in range(self.size):
                for j in range(i+1, self.size):
                    common_neighbors = set(self.neighbors(i)).intersection(set(self.neighbors(j)))
                    resource_allocation[i][j] = sum([1 / len(self.neighbors(k)) for k in common_neighbors])
            return resource_allocation

    
    def link_prediction_scores(self, method, edge=None):
        if method == "jaccard":
            return self.jaccard_index(edge=edge)
        elif method == "preferential_attachment":
            return self.preferential_attachment_index(edge=edge)
        elif method == "adamic_adar":
            return self.adamic_adar_index(edge=edge)
        elif method == "resource_allocation":
            return self.resource_allocation_index(edge=edge)
        elif method == "common_neighbors":
            return self.common_neighbors_index(edge=edge)

        else:
            print("Non supported link prediction method ", method)
            return None


    def has_edge(self, edge):
        n1, n2 = edge
        return self.adj_matrix[n1][n2] == 1

    def does_not_know_edge(self, edge):
        n1, n2 = edge
        return self.adj_matrix[n1][n2] == 2

    def does_not_have_edge(self, edge):
        n1, n2 = edge
        return self.adj_matrix[n1][n2] == 0

    def edges(self):
        return [(i, j) for i in range(self.size) for j in range(i+1, self.size) if self.adj_matrix[i][j] == 1]

    def common_neighbors(self, edge):
        n1, n2 = edge
        return set(self.neighbors(n1)).intersection(set(self.neighbors(n2)))

    def stats(self):
        num_absent_edges = 0
        for i in range(self.size):
            for j in range(i+1, self.size):
                if self.adj_matrix[i][j] == 0:
                    num_absent_edges += 1

        num_absent_edges += len(self.nodes)
        num_present_edges = int(len(np.where(self.adj_matrix == 1)[0]) / 2)
        num_unknown_edges = int(len(np.where(self.adj_matrix == 2)[0]) / 2)

        return num_absent_edges, num_present_edges, num_unknown_edges

    def copy(self):
        copy =  Graph(self.nodes)
        copy.adj_matrix = np.copy(self.adj_matrix)
        return copy

    @staticmethod
    def from_nodes_and_edges(nodes, edges):
        graph = Graph(nodes, with_fixed_edges=True)
        
        for edge in edges:
            graph.add_edge(edge)

        return graph

    @staticmethod
    def from_adj_matrix(adj_matrix, with_fixed_edges=False):
        nodes = list(range(len(adj_matrix)))
        graph = Graph(nodes, with_fixed_edges=with_fixed_edges)
        graph.adj_matrix = adj_matrix
        return graph

    @staticmethod
    def from_npy(filepath):
        adj_matrix = np.load(filepath, allow_pickle=True)
        return Graph.from_adj_matrix(adj_matrix, with_fixed_edges=True)


    @staticmethod
    def from_txt(filepath):
        edges = pd.read_csv(filepath, sep='\t', header=None)
        nodes = list(set(edges[0].tolist() + edges[1].tolist()))
        graph = Graph(nodes, with_fixed_edges=True)
        for i in range(len(edges)):
            graph.add_edge((edges[0][i], edges[1][i]))

        return graph

    def unknown_edges(self):
        return [(i, j) for i in range(self.size) for j in range(i+1, self.size) if self.adj_matrix[i][j] == 2]

    def split_dataset(self, common_prop=0, graph1_prop=0.5):


        graph1 = self.copy()
        graph2 = self.copy()

        graph1_threshold = common_prop + graph1_prop

        for (n1, n2) in self.edges():
            prob = np.random.random()
            if prob < common_prop:
                graph1.add_edge((n1, n2))
                graph2.add_edge((n1, n2))
            elif common_prop < prob and prob < graph1_threshold:
                graph1.add_edge((n1, n2))
                graph2.remove_edge((n1, n2))
            else:
                graph1.remove_edge((n1, n2))
                graph2.add_edge((n1, n2))

        return graph1, graph2

    def fix_edges(self):
        for i in range(self.size):
            for j in range(i+1, self.size):
                if self.adj_matrix[i][j] == 2:
                    self.adj_matrix[i][j] = 0
                    self.adj_matrix[j][i] = 0

    def to_networkx(self):
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        for i in range(self.size):
            for j in range(i+1, self.size):
                if self.adj_matrix[i][j] == 1:
                    G.add_edge(i, j)
        return G

                    
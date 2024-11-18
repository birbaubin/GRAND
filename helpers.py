import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import time 
from prettytable import PrettyTable


def frobenius_distance(prediction, groundtruth):
    return np.linalg.norm(prediction.adjacency_matrix() - groundtruth.adjacency_matrix(), 'fro')

def reconstruction_accuracy(prediction, groundtruth):
    return np.sum(prediction.adjacency_matrix() == groundtruth.adjacency_matrix()) / groundtruth.adjacency_matrix().size

def rae(prediction, groundtruth):
    return (np.linalg.norm(prediction.adjacency_matrix() - groundtruth.adjacency_matrix(), "fro" ) ** 2 )/ (np.linalg.norm(groundtruth.adjacency_matrix(), "fro") ** 2)

def edge_identification_accuracy(prediction, groundtruth):
    predicted_edges = set(prediction.edges())
    groundtruth_edges = set(groundtruth.edges())
    count = 0
    for edge in predicted_edges:
        if edge in groundtruth_edges:
            count += 1
    return count / len(groundtruth_edges)

"""Display the statistics of the graph (Number of impossible, reconstructed and unknown edges)
:param graph: Graph object

"""
def display_graph_stats(graph):
    t = PrettyTable(['Stat', 'Value'])
    stats = graph.stats()
    t.add_row(['0s', stats[0]])
    t.add_row(['1s', stats[1]])
    t.add_row(['?s', stats[2]])
    print(t, "\n")


def compute_reconstruction_metrics(prediction, groundtruth):
    accuracy = reconstruction_accuracy(prediction, groundtruth)
    distance = frobenius_distance(prediction, groundtruth)
    rae_stat = rae(prediction, groundtruth)
    edge_accuracy = edge_identification_accuracy(prediction, groundtruth)

    return accuracy, distance, rae_stat, edge_accuracy

def display_reconstruction_metrics(prediction, groundtruth):
    accuracy = reconstruction_accuracy(prediction, groundtruth)
    distance = frobenius_distance(prediction, groundtruth)
    rae_stat = rae(prediction, groundtruth)
    edge_accuracy = edge_identification_accuracy(prediction, groundtruth)

    t = PrettyTable(['Metric', 'Value'])
    t.add_row(['Accuracy', accuracy])
    t.add_row(['Frobenius distance', distance])
    t.add_row(['Relative Absolute Error', rae_stat])
    t.add_row(['Edge identification accuracy', edge_accuracy])
    print(t, "\n")

def log_graph_stats(graph1_prob, common_prob, expe, attack_type, proba_params, optim, iter_number, time, file_name, prediction, groundtruth):
    stats = prediction.stats()
    TP, FP, TN, FN = ROC_stats(prediction, groundtruth)
    with open(file_name, "a+") as f:
        f.write(f"{expe},{attack_type},{optim},{proba_params[0]},{proba_params[1]},{proba_params[2]},{graph1_prob},{common_prob},{iter_number},{stats[0]},{stats[1]},{stats[2]},{TP},{FP},{TN},{FN},{time}\n")
        f.close()
    return stats[0], stats[1], stats[2]

def ROC_stats(prediction, groundtruth):
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    prediction_adj = prediction.adjacency_matrix()
    groundtruth_adj = groundtruth.adjacency_matrix()

    TP = np.sum(np.logical_and(prediction_adj == 1, groundtruth_adj == 1))
    FP = np.sum(np.logical_and(prediction_adj == 1, groundtruth_adj == 0))
    TN = np.sum(np.logical_and(prediction_adj == 0, groundtruth_adj == 0))  
    FN = np.sum(np.logical_and(prediction_adj == 0, groundtruth_adj == 1))

    return TP, FP, TN, FN
    
    def precision_recall(TP, FP, TN, FN):
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        return precision, recall


def sort_edges(rec_graph, method=None):

    unknown_edges = rec_graph.unknown_edges()

    stats = rec_graph.stats()
    print("Number of unknown edges : ", stats[2])
    sorted_edges = {}

    if method in ["jaccard", "preferential_attachment", "adamic_adar", "resource_allocation", "common_neighbors"]:
        for edge in unknown_edges:
            score = rec_graph.link_prediction_scores(method, edge)
            sorted_edges[edge] = score

    elif method == "random":
        for edge in unknown_edges:
            sorted_edges[edge] = np.random.random()

    elif method == "logistic_regression":

        features = []
        labels = []
        jaccard = rec_graph.jaccard_index()
        pref_attach = rec_graph.preferential_attachment_index()
        resource_allocation = rec_graph.resource_allocation_index()        

        
        for edge in rec_graph.edges():
            u, v = edge
            features.append([jaccard[u][v], pref_attach[u][v], resource_allocation[u][v]])
            labels.append(1)
        
        print(len(features), len(labels))

        number_positive_samples = len(features)
        for i in range(number_positive_samples):
            u = np.random.randint(0, len(rec_graph.nodes))
            v = np.random.randint(0, len(rec_graph.nodes))
            while rec_graph.has_edge((u, v)) or rec_graph.does_not_know_edge((u, v)):
                u = np.random.randint(0, len(rec_graph.nodes))
                v = np.random.randint(0, len(rec_graph.nodes))
            features.append([jaccard[u][v], pref_attach[u][v], resource_allocation[u][v]])
            labels.append(0)

        features = np.array(features)
        labels = np.array(labels)
        model = LogisticRegression()
        model.fit(features, labels)


        similarities = []
        for u in range(len(rec_graph.nodes)):
            for v in range(u+1, len(rec_graph.nodes)):
                features = np.array([jaccard[u][v], pref_attach[u][v], resource_allocation[u][v]])
                similarities.append((u, v, model.predict_proba(features.reshape(1, -1))[0][1]))

    else : 
        print("Non supported link prediction method ", method)
        return None

    sorted_edges = {k: v for k, v in sorted(sorted_edges.items(), key=lambda item: item[1], reverse=True)}
    # print(sorted_edges)
    sorted_edges = list(sorted_edges.keys())

    return sorted_edges



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

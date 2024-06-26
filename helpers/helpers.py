import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import time 
from prettytable import PrettyTable


def frobenius_distance(prediction, groundtruth):
    return np.linalg.norm(prediction.adj_matrix - groundtruth.adj_matrix, 'fro')

def reconstruction_accuracy(prediction, groundtruth):
    return np.sum(prediction.adj_matrix == groundtruth.adj_matrix) / groundtruth.adj_matrix.size

def rae(prediction, groundtruth):
    return (np.linalg.norm(prediction.adj_matrix - groundtruth.adj_matrix, "fro" ) ** 2 )/ (np.linalg.norm(groundtruth.adj_matrix, "fro") ** 2)

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
    t.add_row(['Number of impossible edges', stats[0]])
    t.add_row(['Number of reconstructed edges', stats[1]])
    t.add_row(['Number of unknown edges', stats[2]])
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

def log_graph_stats(graph1_prob, common_prob, expe, attack_type, graph, iter_number, time, file_name, groundtruth, ):
    stats = graph.stats()
    TP, FP, TN, FN = ROC_stats(graph, groundtruth)
    with open(file_name, "a+") as f:
        f.write(f"{expe},{attack_type},{graph1_prob},{common_prob},{iter_number},{stats[0]},{stats[1]},{stats[2]},{TP},{FP},{TN},{FN},{time}\n")
    return stats[0], stats[1], stats[2]

def ROC_stats(prediction, groundtruth):
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    not_predicted_edges = [(i, j) for i in range(len(prediction.nodes)) 
                           for j in range(i, len(prediction.nodes)) 
                           if prediction.does_not_have_edge((i, j))]

    predicted_edges = set(prediction.edges())
    groundtruth_edges = set(groundtruth.edges())

    for edge in predicted_edges:
        if edge in groundtruth_edges:
            TP += 1
        else:
            FP += 1

    for edge in not_predicted_edges:
        if edge in groundtruth_edges:
            FN += 1
        else:
            TN += 1

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


def benchmark_reconstruct_union_graph(rec_graph, step, common_neighbors_count, graph1, groundtruth, sorting=None):

    roc_stats = []
    graph = rec_graph.copy()
    nodes_count = len(graph.nodes)
    inserted_edge_count = 0
    stats = graph.stats()
    max_num_links = 0
    for i in range(nodes_count):
        max_num_links += (common_neighbors_count[i][i] - len(graph.neighbors(i))) /2

    print("Max number of links to insert : ", max_num_links)

    sorted_edges = sort_edges(rec_graph, sorting)
    print("Number of sorted edges : ", len(sorted_edges))

    with tqdm(total=max_num_links) as pbar:
        for i in range(len(sorted_edges)):
            if inserted_edge_count == max_num_links:
                break
            n1, n2 = sorted_edges[i]
            if not graph.has_edge((n1, n2)):
                if len(graph.neighbors(n1)) < common_neighbors_count[n1][n1] and len(graph.neighbors(n2)) < common_neighbors_count[n2][n2]:
                    graph.add_edge((n1, n2))
                    # print("Inserted edge ", (n1, n2))
                    inserted_edge_count += 1
                    pbar.update(1)

            if inserted_edge_count % step == 0:
                roc_stats.append(ROC_stats(graph, groundtruth))


    return graph, roc_stats



def random_graph(n, p):
    adj_matrix = np.random.choice([0, 1], size=(n, n), p=[1-p, p])
    graph = Graph(list(range(n)), with_fixed_edges=True)
    for i in range(n):
        for j in range(i+1, n):
            if adj_matrix[i][j] == 1:
                graph.add_edge((i, j))
    return graph
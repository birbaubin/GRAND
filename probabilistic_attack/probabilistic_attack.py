import numpy as np
from helpers import *
from node2vec import Node2Vec
from probabilistic_attack.LinkPrediction import LinkPredictor
from itertools import product


def similarity_based_completion(gstar, k, A, method="common_neighbors"):
    print(f"\n######## Similarity-based completion using {method} #######")
    gstar_copy = gstar.copy()
    unknown_edges = set(gstar_copy.unknown_edges())
    predictions = {edge: gstar_copy.link_prediction_scores(method=method, edge=edge) for edge in unknown_edges}

    predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

    additions = 0
    i = 0

    while additions < k and i < len(predictions):
        edge, score = predictions[i]
        n1, n2 = edge
        print("Trying edge ", edge)
        neighbors_n1 = gstar_copy.neighbors(n1)
        neighbors_n2 = gstar_copy.neighbors(n2)
        
        ok = True

        if len(gstar_copy.neighbors(n1)) >= A[n1, n1] or len(gstar_copy.neighbors(n2)) >= A[n2, n2]:
            ok = False

        if ok:
            gstar_copy.add_edge(edge)
            additions += 1

        i+=1
        
    return gstar_copy



def A_based_completion(gstar, number_of_additions, A):
    print(f"\n######## Similarity-based completion using common neighbors on A #######")
    gstar_copy = gstar.copy()
    unknown_edges = set(gstar_copy.unknown_edges())

    flattened_indices = np.argsort(A.ravel())[::-1]
    row_indices, col_indices = np.unravel_index(flattened_indices, A.shape)
    sorted_indices = np.stack((row_indices, col_indices), axis=-1)

    predictions = [(edge[0], edge[1]) for edge in sorted_indices if (edge[0], edge[1]) in unknown_edges]

    for i in range(number_of_additions):
        edge = predictions[i]
        gstar_copy.add_edge(edge)

    return gstar_copy


def model_based_completion(gstar, number_of_additions, verbose=False):
    print("\n######## Custom model-based completion #######")
    if len(gstar.edges()) == 0 or len(gstar.unknown_edges()) == 0:
        return gstar

    gstar_copy = gstar.copy()
    model = LinkPredictor(gstar_copy, 32, verbose=verbose)
    model.fit(200)
    predictions = model.predict()

    predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

    number_added_edges = 0

    for i in range(number_of_additions):
        edge, score = predictions[i]

        if not gstar.has_edge(edge):
            gstar_copy.add_edge(edge)
            number_added_edges += 1

    return gstar_copy


if __name__ == "__main__":

    common_prop = [0]
    graph1_prop = [0, .1, 0.25, 0.5, 0.75]
    dataset_name = "flickr"

    for graph1_prop, common_prop in product(graph1_prop, common_prop):
        print(f"\n\n########## Graph1 prop: {graph1_prop}, Common prop: {common_prop} ###########")
        
        gstar = Graph.from_npy(f"rec_deterministic/{dataset_name}_{graph1_prop}_{common_prop}.npy")
        dataset = Graph.from_npy(f"datasets/{dataset_name}.adj.npy")

        display_graph_stats(gstar)
        copy = gstar.copy()
        copy.fix_edges()
        display_reconstruction_metrics(copy, dataset)
        number_of_additions = len(dataset.edges()) - len(gstar.edges())

        ####### similarity-based completion ########
        method = "common_neighbors"
        gstar_prime = similarity_based_completion(gstar, number_of_additions, method=method)
        display_graph_stats(gstar_prime)
        np.save(f"rec_probabilistic/{dataset_name}_{graph1_prop}_{common_prop}_{method}.npy", gstar_prime.adj_matrix)
        gstar_prime.fix_edges()
        metrics = compute_reconstruction_metrics(gstar_prime, dataset)
        with open(f"logs_probabilistic/{dataset_name}.csv", "a+") as f:
            f.write(f"{graph1_prop},{common_prop},{method},{metrics[0]},{metrics[1]},{metrics[2]},{metrics[3]}\n")
        # display_reconstruction_metrics(gstar_prime, dataset)

        ####### A-based completion ########
        A = np.dot(gstar.adj_matrix, gstar.adj_matrix.T)
        gstar_prime = A_based_completion(gstar, number_of_additions, A)
        display_graph_stats(gstar_prime)
        np.save(f"rec_probabilistic/{dataset_name}_{graph1_prop}_{common_prop}_A.npy", gstar_prime.adj_matrix)
        gstar_prime.fix_edges()
        metrics = compute_reconstruction_metrics(gstar_prime, dataset)
        with open(f"logs_probabilistic/{dataset_name}.csv", "a+") as f:
            f.write(f"{graph1_prop},{common_prop},A,{metrics[0]},{metrics[1]},{metrics[2]},{metrics[3]}\n")
        # accuracy = edge_identification_accuracy(gstar_prime, dataset)

        display_reconstruction_metrics(gstar_prime, dataset)

        ######## model-based completion ########xx
        gstar_prime = model_based_completion(gstar, number_of_additions, verbose=True)
        display_graph_stats(gstar_prime)
        np.save(f"rec_probabilistic/{dataset_name}_{graph1_prop}_{common_prop}_gae.npy", gstar_prime.adj_matrix)
        gstar_prime.fix_edges()
        metrics = compute_reconstruction_metrics(gstar_prime, dataset)
        with open(f"logs_probabilistic/{dataset_name}.csv", "a+") as f:
            f.write(f"{graph1_prop},{common_prop},model,{metrics[0]},{metrics[1]},{metrics[2]},{metrics[3]}\n")
        # display_reconstruction_metrics(gstar_prime, dataset)











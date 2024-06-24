import numpy as np
from helpers import *
from node2vec import Node2Vec
from probabilistic_attack.LinkPrediction import LinkPredictor
from itertools import product
from tqdm import tqdm


def similarity_based_completion(gstar, k, A, method="common_neighbors"):
    
    gstar_copy = gstar.copy()
    unknown_edges = set(gstar_copy.unknown_edges())
    predictions = {edge: gstar_copy.link_prediction_scores(method=method, edge=edge) for edge in unknown_edges}
    predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

    additions = 0
    i = 0

    already_selected_nodes = set()

    pbar = tqdm(total = k, desc="Similarity based completion with metric " + method)
    while additions < k and i < len(predictions):
        edge, score = predictions[i]
        if edge[0] in already_selected_nodes or edge[1] in already_selected_nodes:
            i += 1
            continue

        n1, n2 = edge
        neighbors_n1 = gstar_copy.neighbors(n1)
        neighbors_n2 = gstar_copy.neighbors(n2)
        
        ok = True

        for n in neighbors_n1:
            if len(gstar_copy.common_neighbors((n, n2))) == A[n, n2]:
                ok = False
                break
        for n in neighbors_n2:
            if len(gstar_copy.common_neighbors((n, n1))) == A[n, n1]:
                ok = False
                break

        if len(gstar_copy.neighbors(n1)) == A[n1, n1] or len(gstar_copy.neighbors(n2)) == A[n2, n2]:
            ok = False

        if ok:
            gstar_copy.add_edge(edge)
            additions += 1
            already_selected_nodes.add(n1)
            already_selected_nodes.add(n2)
            pbar.update(1)


        i+=1
        
    return gstar_copy, additions



def A_based_completion(gstar, number_of_additions, A, DEBUG=False):
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


def model_based_completion(gstar, number_of_additions, verbose=False, DEBUG=False):
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
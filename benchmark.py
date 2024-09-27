from deterministic_attack.deterministic_attack import *
from probabilistic_attack.probabilistic_attack import *
from helpers.helpers import *
from helpers.Graph import Graph
from itertools import product
import numpy as np
import argparse


argparser = argparse.ArgumentParser()
argparser.add_argument("--dataset", type=str, default="polblogs")
argparser.add_argument("--type", type=str, default="preferential_attachment", choices=["D", "P", "DP", "DPD"])
argparser.add_argument("--number_of_experiments", type=int, default=5)
argparser.add_argument("--graph1_props", type=float, nargs='+', default=[0.1])


args = argparser.parse_args()
DEBUG = True
number_of_experiments = args.number_of_experiments
dataset_name = args.dataset
heuristic = args.heuristic
prediction_type = args.prediction_type

if dataset_name != "block":
    dataset = Graph.from_txt(f"datasets/{dataset_name}.txt")
else:
    dataset = Graph.block_from_txt(f"datasets/{dataset_name}.txt")


graph1_props = args.graph1_props
common_props = [0]
addition_proportion = args.addition_proportion

for expe in range(number_of_experiments):
    for graph1_prop, common_prop in product(graph1_props, common_props):
        graph1, graph2 = dataset.split_dataset(common_prop=common_prop, graph1_prop=graph1_prop)
        A = np.dot(dataset.adj_matrix, dataset.adj_matrix)

        print("\n######## General information ########")

        
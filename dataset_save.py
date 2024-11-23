from numpy import save, load
import pandas as pd
import numpy as np


def convert_cora_dataset():
    
    content = pd.read_csv(f'/Users/aubinbirba/Desktop/graph-reconstruction-2/datasets/cora_erdos.content', sep='\t', header=None)

    partition_1 = set(content[0])
    partition_1_nodes = {node: i+1 for i, node in enumerate(partition_1)}

    dataset = pd.DataFrame(columns=['source', 'target', 'label'])
    for i, row in content.iterrows():
        for j in range(len(row)):
            if row[j] == 1:
                dataset = pd.concat([dataset, pd.DataFrame([[partition_1_nodes[row[0]], j, 1]], columns=['source', 'target', 'label'])])

    dataset.to_csv(f'/Users/aubinbirba/Desktop/graph-reconstruction-2/datasets/cora_erdos.txt', index=False, header=False, sep='\t')
    


def convert_cora_dataset_2():
    content = pd.read_csv(f'/Users/aubinbirba/Desktop/graph-reconstruction-2/datasets/cora_erdos.content', sep='\t', header=None)

    partition_1 = set(content[0])
    partition_1_nodes = {node: i+1 for i, node in enumerate(partition_1)}
    max_node = max(partition_1_nodes.values())


    dataset = pd.DataFrame(columns=['source', 'target', 'label'])
    for i, row in content.iterrows():
        for j in range(1, len(row)):
            if row[j] == 1:
                dataset = pd.concat([dataset, pd.DataFrame([[partition_1_nodes[row[0]], max_node+j, 1]], columns=['source', 'target', 'label'])])

    dataset.to_csv(f'/Users/aubinbirba/Desktop/graph-reconstruction-2/datasets/cora_erdos_2.txt', index=False, header=False, sep='\t')


def convert_cora_cites_dataset():
    cites = pd.read_csv(f'/Users/aubinbirba/Desktop/graph-reconstruction-2/datasets/cora_erdos.cites', sep='\t', header=None)

    dataset = pd.DataFrame(columns=['source', 'target', 'label'])
    nodes = np.union1d(np.unique(cites[0]), np.unique(cites[1]))

    node_mapping = {node : i+1 for i, node in enumerate(nodes)}

    for i, row in cites.iterrows():
        dataset = pd.concat([dataset, pd.DataFrame([[node_mapping[row[0].item()], node_mapping[row[1].item()], 1]], columns=['source', 'target', 'label'])])


    dataset.to_csv(f'/Users/aubinbirba/Desktop/graph-reconstruction-2/datasets/cora_erdos_cites.txt', index=False, header=False, sep='\t')

convert_cora_cites_dataset()


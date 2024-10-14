from deeprobust.graph.data import Dataset
from numpy import save, load
import pandas as pd


# dataset_name = "polblogs"
# dataset = Dataset(root='/tmp/', name=dataset_name)
# save(f'/Users/aubinbirba/Documents/PhD/Graph attack/code/attack/datasets/{dataset_name}.adj.npy', dataset.adj.A)

data = pd.read_csv(f'/Users/aubinbirba/Desktop/graph-reconstruction-2/datasets/email.mtx', sep=' ', header=None)
print(len(data))
# add a column to the dataframe with the value 1
data[2] = 1

for row in data.iterrows():
    data = pd.concat([data, pd.DataFrame([[row[1][1], row[1][0], 1]])])

print(len(data))

# save the dataframe to a csv file
data.to_csv(f'/Users/aubinbirba/Desktop/graph-reconstruction-2/datasets/email.txt', index=False, header=False, sep='\t')
    



                    


# Graph reconstruction

This code implements the graph reconstruction algorithms described in the the ongoing paper about graph reconstruction from common neighbors information and a subgraph of the original graph.

## Requirements
- Python 
- numpy
- matplotlib
- tqdm
- prettytable
- pandas
- torch
- torch_geometric

To install the requirements, make sure to have [python](https://www.python.org) installed , run:
```bash
pip install -r requirements.txt
```

## Running the attacks
The main attack functions are implemented in the deterministic_attack, probabilistic_attack and hybrid_attack modules. 
To run the attacks, you can use the following command:
```bash
python main.py
```

## Exploring and experimenting
For exploration purposes, you can use the notebook `scratc.ipynb` to run custom code. The notebook is already set up with (most of) the necessary imports. It also contains an example of computation of the eigenvalues of the adjacency matrix of two cospectral graphs, and the computation of the common neighbors matrix of a graph.

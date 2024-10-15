# Graph reconstruction

This code implements the graph reconstruction algorithms described in the the ongoing paper about graph reconstruction from common neighbors information and a subgraph of the original graph.

## Requirements
- Python 
- numpy
- matplotlib
- tqdm
- prettytable
- pandas


To install the requirements, make sure to have [python](https://www.python.org) installed , run:
```bash
pip install -r requirements.txt
```

## Benchmarking the algorithms
The `benchmark.py` script runs the different attacks. Sample commands are provided in the `launcher.sh` script. To run the benchmark, execute the following command:
```bash
./launcher.sh
```

The benchmark script runs with the following parameters:

- `--dataset` : the dataset to use. The datasets are stored in the `datasets` folder.
- `--type` : the type of attack to run. The possible values are:
  - `D` : deterministic attack
  - `P` : probabilistic attack
  - `DP` : deterministic attack followed by a probabilistic attack
  - `DPD` : deterministic attack followed by a probabilistic attack followed by a deterministic attack
- `--n_experiments` : the number of experiments to run for each attack
- `--graph1_props` : the proportion of edges in the known graph.
- `--proba_params` : the parameters of the probabilistic attack. The first, second and third parametesr are respectively alpha, beta and gamma.
- `--sanity_check_optim` : whether to run the sanity check optimization. 
- `--log_deterministic` : whether to log the deterministic attack results.

## Exploring and experimenting
For exploration purposes, you can use the notebook `scratch.ipynb` to run custom code. The notebook is already set up with (most of) the necessary imports. It also contains an example of computation of the eigenvalues of the adjacency matrix of two cospectral graphs, and the computation of the common neighbors matrix of a graph.

15/10/2024 : The `scratch.ipynb` notebook is up to date with experiments on the components that generate cosquare graphs in the Netscience dataset. The notebook contains the code to generate the cosquare graphs and explore the context of the non reconstructible components.

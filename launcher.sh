python_exe=python3.9
# $python_exe -m pip install -r requirements.txt

# $python_exe benchmark.py --dataset cora --type D  --n_experiments 1
# $python_exe benchmark.py --dataset cora --type P --n_experiments 1

# $python_exe benchmark.py --dataset cora --type DP --proba_params  0 0 0 --n_experiments 1
# $python_exe benchmark.py --dataset cora --type DP --proba_params  0 0 1 --n_experiments 1
# $python_exe benchmark.py --dataset cora --type DP --proba_params  0.05 0.05 0.9 --n_experiments 1
# $python_exe benchmark.py --dataset cora --type DP --proba_params  0.125 0.125 0.75 --n_experiments 1
# $python_exe benchmark.py --dataset cora --type DP --proba_params  0.25 0.25 0.5 --n_experiments 1
# $python_exe benchmark.py --dataset cora --type DP --proba_params  0.75 0.125 0.125 --n_experiments 1
$python_exe benchmark.py --dataset cora --type DP --proba_params  0.75 0.25 0 --n_experiments 1

$python_exe benchmark.py --dataset cora --type DPD --proba_params  0 0 0 --n_experiments 1
# $python_exe benchmark.py --dataset cora --type DPD --proba_params  0 0 1 --n_experiments 1
# $python_exe benchmark.py --dataset cora --type DPD --proba_params  0.05 0.05 0.9 --n_experiments 1
# $python_exe benchmark.py --dataset cora --type DPD --proba_params  0.125 0.125 0.75 --n_experiments 1
# $python_exe benchmark.py --dataset cora --type DPD --proba_params  0.25 0.25 0.5 --n_experiments 1
# $python_exe benchmark.py --dataset cora --type DPD --proba_params  0.75 0.125 0.125 --n_experiments 1
$python_exe benchmark.py --dataset cora --type DPD --proba_params  0.75 0.25 0 --n_experiments 1





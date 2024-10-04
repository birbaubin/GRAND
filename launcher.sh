python_exe=python3.9
# $python_exe -m pip install -r requirements.txt

$python_exe benchmark.py --dataset netscience --type D  --n_experiments 1

$python_exe benchmark.py --dataset netscience --type P --n_experiments 1

$python_exe benchmark.py --dataset netscience --type DP --proba_params 0.0 1 --n_experiments 1
$python_exe benchmark.py --dataset netscience --type DP --proba_params 0.25 0.75 --n_experiments 1
$python_exe benchmark.py --dataset netscience --type DP --proba_params 0.5 0.5 --n_experiments 1
$python_exe benchmark.py --dataset netscience --type DP --proba_params 0.75 0.25 --n_experiments 1
$python_exe benchmark.py --dataset netscience --type DP --proba_params 1.0 0.0 --n_experiments 1


$python_exe benchmark.py --dataset netscience --type DPD --proba_params 0.0 1.0 --n_experiments 1
$python_exe benchmark.py --dataset netscience --type DPD --proba_params 0.25 0.75 --n_experiments 1
$python_exe benchmark.py --dataset netscience --type DPD --proba_params 0.5 0.5 --n_experiments 1
$python_exe benchmark.py --dataset netscience --type DPD --proba_params 0.75 0.25 --n_experiments 1
$python_exe benchmark.py --dataset netscience --type DPD --proba_params 1.0 0.0 --n_experiments 1
